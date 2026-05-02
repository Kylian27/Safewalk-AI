"""
utils.py
--------
공용 유틸리티 함수.
IoU 계산, 샤프니스 측정, 히스토리 저장/로드 등.
"""

import pickle
import cv2
import torch
import numpy as np
from pathlib import Path
from config import N_CLASSES


# ── IoU ────────────────────────────────────────────────────────────────────────

def compute_iou_tensor(pred_map: torch.Tensor,
                       target_map: torch.Tensor) -> torch.Tensor:
    """
    pred_map   : (B, H, W) bool or uint8
    target_map : (B, H, W) bool or uint8
    returns    : (B,) float  — 샘플별 IoU
    """
    intersection = (pred_map & target_map).sum(dim=(1, 2)).float()
    union        = (pred_map | target_map).sum(dim=(1, 2)).float()
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return iou


def compute_miou(pred_logits: torch.Tensor,
                 targets: torch.Tensor,
                 n_classes: int = N_CLASSES) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pred_logits : (B, C, H, W)
    targets     : (B, H, W)  long
    returns     : (mean_iou, per_class_iou)  — 배치 평균
    """
    pred_map = torch.argmax(pred_logits, dim=1)   # (B, H, W)
    iou_per_class = torch.zeros(n_classes, device=pred_logits.device)

    for cls in range(n_classes):
        pred_mask   = (pred_map == cls)
        target_mask = (targets  == cls)
        iou = compute_iou_tensor(pred_mask, target_mask)
        iou_per_class[cls] = iou.mean()

    return iou_per_class.mean(), iou_per_class


# ── Sharpness ──────────────────────────────────────────────────────────────────

def laplacian_sharpness(frame: np.ndarray) -> float:
    """
    Laplacian 분산으로 프레임 샤프니스 측정.
    높을수록 선명. 캘리브레이션 최적 프레임 선택에 사용.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
        if frame.ndim == 3 else frame
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_best_frame(video_path: str, sample_count: int = 10) -> np.ndarray | None:
    """
    영상에서 샤프니스·밝기 기준으로 가장 좋은 프레임 반환.
    밝기가 [80, 200] 범위 밖이면 가중치 0.3 페널티.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    best_score, best_frame = 0.0, None

    for i in range(sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / sample_count))
        ret, frame = cap.read()
        if not ret:
            continue

        sharpness  = laplacian_sharpness(frame)
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        weight     = 1.0 if 80 < brightness < 200 else 0.3
        score      = sharpness * weight

        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    cap.release()
    return best_frame


# ── History ───────────────────────────────────────────────────────────────────

def save_history(history: dict, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(history, f)


def load_history(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Logging ───────────────────────────────────────────────────────────────────

def print_epoch(epoch: int, total: int, train_loss: float,
                val_loss: float, miou: float,
                per_class_iou: torch.Tensor, lr: float) -> None:
    class_names = ["Background", "Interior", "Boundary"]
    iou_str = " | ".join(
        f"{n}: {v:.4f}" for n, v in zip(class_names, per_class_iou)
    )
    print(f"[{epoch:>3}/{total}]  "
          f"train={train_loss:.4f}  val={val_loss:.4f}  "
          f"mIoU={miou:.4f}  lr={lr:.2e}")
    print(f"         Per-class → {iou_str}")
