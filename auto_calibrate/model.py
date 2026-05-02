"""
04_model.py
-----------
CrosswalkTrimapNet : EfficientNet-B4 기반 트라이맵 세그멘테이션 모델

구성:
  Encoder  : EfficientNet-B4 (frozen)
  Skip     : GatedLaplacianUnit + CBAM (각 스킵 커넥션마다)
  Decoder  : ConvTranspose2d 업샘플링 + skip concat
  Special  : BoundaryAttentionModule (얕은 디코더 직후)
  Final    : LaplacianFusion → Conv1×1 → 3클래스 출력
  AuxHead  : Deep Supervision (학습 시만 활성화)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import config


# ── Attention ──────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 8):
        super().__init__()
        mid = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Channel + Spatial Attention Module"""
    def __init__(self, in_planes: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ── Laplacian ──────────────────────────────────────────────────────────────────

class MultiScaleLaplacianFilter(nn.Module):
    """
    고정 가중치 3×3 + 5×5 Laplacian → 채널 압축.
    횡단보도 줄무늬 엣지(가는 것·넓은 것 동시)를 강조.
    """
    def __init__(self, channels: int):
        super().__init__()
        lap3 = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32
        )
        lap5 = torch.tensor([
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 24, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ], dtype=torch.float32)

        w3 = lap3.expand(channels, 1, 3, 3).clone()
        w5 = lap5.expand(channels, 1, 5, 5).clone()

        self.filter3 = nn.Conv2d(channels, channels, 3, padding=1,
                                  groups=channels, bias=False)
        self.filter5 = nn.Conv2d(channels, channels, 5, padding=2,
                                  groups=channels, bias=False)
        self.filter3.weight = nn.Parameter(w3, requires_grad=False)
        self.filter5.weight = nn.Parameter(w5, requires_grad=False)
        self.compress = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(torch.cat([self.filter3(x), self.filter5(x)], dim=1))


class LaplacianFilter(nn.Module):
    """Final 단계용 단일 스케일 Laplacian (고정 가중치)"""
    def __init__(self, channels: int,
                 scale_center: float = 8.0, scale_surround: float = -1.0):
        super().__init__()
        lap = torch.tensor([
            [scale_surround, scale_surround, scale_surround],
            [scale_surround, scale_center,   scale_surround],
            [scale_surround, scale_surround, scale_surround],
        ], dtype=torch.float32)
        weight = lap.expand(channels, 1, 3, 3).clone()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                               groups=channels, bias=False)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── GatedLaplacianUnit ─────────────────────────────────────────────────────────

class GatedLaplacianUnit(nn.Module):
    """
    x + Gate(x) * Laplacian(x)
    Gate: 학습 가능한 Sigmoid → 엣지 강조 강도를 데이터에서 학습
    """
    def __init__(self, channels: int):
        super().__init__()
        self.lap_filter = MultiScaleLaplacianFilter(channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate(x) * self.lap_filter(x)


# ── Decoder block ──────────────────────────────────────────────────────────────

def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


# ── Main model ─────────────────────────────────────────────────────────────────

class CrosswalkTrimapNet(nn.Module):
    """
    Parameters
    ----------
    encoder_name    : smp encoder 이름 (default: efficientnet-b4)
    encoder_weights : 사전학습 가중치 (default: imagenet)
    in_channels     : 입력 채널 수 (default: 3)
    out_channels    : 출력 클래스 수 (default: 3 — 배경/내부/경계)
    freeze_backbone : Encoder 가중치 동결 여부 (default: True)
    """

    def __init__(self,
                 encoder_name: str    = config.ENCODER_NAME,
                 encoder_weights: str = config.ENCODER_WEIGHTS,
                 in_channels: int     = 3,
                 out_channels: int    = config.N_CLASSES,
                 freeze_backbone: bool = config.FREEZE_BACKBONE):
        super().__init__()

        # ── Encoder (EfficientNet-B4) ──────────────────────────────────────────
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        enc_ch = self.encoder.out_channels
        # enc_ch 예: (3, 48, 24, 32, 56, 160)  — depth=5 기준

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ── Skip connection processors ─────────────────────────────────────────
        # encoder 출력 인덱스: [stem, skip0, skip1, skip2, skip3, bottleneck]
        self.glu3 = GatedLaplacianUnit(enc_ch[-2])
        self.glu2 = GatedLaplacianUnit(enc_ch[-3])
        self.glu1 = GatedLaplacianUnit(enc_ch[-4])
        self.glu0 = GatedLaplacianUnit(enc_ch[-5])

        self.cbam3 = CBAM(enc_ch[-2])
        self.cbam2 = CBAM(enc_ch[-3])
        self.cbam1 = CBAM(enc_ch[-4])
        self.cbam0 = CBAM(enc_ch[-5])

        # ── Decoder ────────────────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(enc_ch[-1],  enc_ch[-2], 2, stride=2)
        self.dec4 = conv_block(enc_ch[-2] * 2,      enc_ch[-2])

        self.up3  = nn.ConvTranspose2d(enc_ch[-2],  enc_ch[-3], 2, stride=2)
        self.dec3 = conv_block(enc_ch[-3] * 2,      enc_ch[-3])

        self.up2  = nn.ConvTranspose2d(enc_ch[-3],  enc_ch[-4], 2, stride=2)
        self.dec2 = conv_block(enc_ch[-4] * 2,      enc_ch[-4])

        self.up1  = nn.ConvTranspose2d(enc_ch[-4],  enc_ch[-5], 2, stride=2)
        self.dec1 = conv_block(enc_ch[-5] * 2,      enc_ch[-5])

        self.up0  = nn.ConvTranspose2d(enc_ch[-5],  enc_ch[0],  2, stride=2)
        self.dec0 = conv_block(enc_ch[0]  * 2,      enc_ch[0])

        # ── BoundaryAttentionModule ────────────────────────────────────────────
        # dec1 출력(얕은 디코더) 직후 → 경계 픽셀 강조
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(enc_ch[-5], enc_ch[-5] // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_ch[-5] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch[-5] // 4, 1, 1),
            nn.Sigmoid(),
        )

        # ── Final Laplacian Fusion ─────────────────────────────────────────────
        self.final_lap   = LaplacianFilter(enc_ch[0])
        self.final_fusion = conv_block(enc_ch[0] * 2, enc_ch[0])

        # ── Output heads ───────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(enc_ch[0], out_channels, 1)

        # AuxHead: dec2 출력(중간 디코더)에서 Deep Supervision
        # 학습 시에만 활성화 (forward에서 self.training 체크)
        self.aux_head = nn.Conv2d(enc_ch[-4], out_channels, 1)

        # ── Weight init (decoder only) ─────────────────────────────────────────
        self._init_decoder_weights()

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        학습 시 : {"main": (B,C,H,W), "aux": (B,C,H,W)} 반환
        추론 시 : (B,C,H,W) 반환
        """
        features = self.encoder(x)
        # features: [stem, skip0, skip1, skip2, skip3, bottleneck]
        stem, s0, s1, s2, s3, bottleneck = features

        # Skip connection: GatedLaplacian → CBAM
        s3 = self.cbam3(self.glu3(s3))
        s2 = self.cbam2(self.glu2(s2))
        s1 = self.cbam1(self.glu1(s1))
        s0 = self.cbam0(self.glu0(s0))

        # Decoder
        d = self.dec4(torch.cat([self.up4(bottleneck), s3], dim=1))
        d = self.dec3(torch.cat([self.up3(d), s2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d), s1], dim=1))   # AuxHead 입력

        # BoundaryAttentionModule
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))
        d1 = d1 * self.boundary_attention(d1)

        d = self.dec0(torch.cat([self.up0(d1), stem], dim=1))

        # Final Laplacian Fusion
        d = self.final_fusion(torch.cat([d, self.final_lap(d)], dim=1))
        main_out = self.out_conv(d)

        if self.training:
            # AuxHead 출력을 main과 동일한 해상도로 업샘플링
            aux_out = F.interpolate(
                self.aux_head(d2),
                size=main_out.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            return {"main": main_out, "aux": aux_out}

        return main_out

    # ── Init ───────────────────────────────────────────────────────────────────

    def _init_decoder_weights(self) -> None:
        """Encoder를 제외한 Decoder 전체 가중치 Kaiming 초기화"""
        for name, m in self.named_modules():
            if "encoder" in name:
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = CrosswalkTrimapNet().to(config.DEVICE)
    dummy = torch.randn(2, 3, *config.IMAGE_SIZE).to(config.DEVICE)

    model.train()
    out = model(dummy)
    print("[Train mode]")
    print(f"  main : {out['main'].shape}")
    print(f"  aux  : {out['aux'].shape}")

    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print("[Eval mode]")
    print(f"  output : {out.shape}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  전체 파라미터   : {total:,}")
    print(f"  학습 가능 파라미터 : {trainable:,}")
    print(f"  동결 파라미터    : {total - trainable:,}")
