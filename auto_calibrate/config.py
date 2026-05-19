"""
config.py
---------
경로, 하이퍼파라미터 중앙 관리.
모든 파일에서 이 파일만 import해서 사용.
"""

from pathlib import Path

# ── Base ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_ROOT = BASE_DIR / "FPVCrosswalk2025"
TRIMAP_DIR   = BASE_DIR / "trimaps"
CKPT_DIR     = BASE_DIR / "checkpoints"
LOG_DIR      = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
BASE_CKPT    = MODELS_DIR / "base_best.pt"
FINETUNE_CKPT = CKPT_DIR / "finetune_best.pt"
HISTORY_PATH  = LOG_DIR  / "history.pkl"

# ── Trimap ─────────────────────────────────────────────────────────────────────
SE_SIZE   = 19          # SE 탐색으로 결정된 최적값
N_CLASSES = 3           # 0=배경  1=내부  2=경계

# ── Model ──────────────────────────────────────────────────────────────────────
ENCODER_NAME    = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
FREEZE_BACKBONE = True

# ── Training ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (512, 512)   # (H, W)
BATCH_SIZE   = 32
EPOCHS       = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-2
ETA_MIN       = 1e-6        # CosineAnnealingLR 최소 LR
AUX_WEIGHT    = 0.4         # Deep Supervision 보조 손실 가중치
PATIENCE      = 10          # Early Stopping patience

# Fine-tuning 시 덮어쓸 값
FINETUNE_LR      = 1e-4
FINETUNE_EPOCHS  = 50

# ── Data split ─────────────────────────────────────────────────────────────────
VAL_RATIO    = 0.2
RANDOM_STATE = 42

# ── Augmentation ───────────────────────────────────────────────────────────────
HFLIP_PROB        = 0.5
COLOR_JITTER      = dict(brightness=0.2, contrast=0.2, saturation=0.2)
# Fine-tuning 시 야간·역광 대응을 위해 밝기 범위 확대
FINETUNE_COLOR_JITTER = dict(brightness=0.4, contrast=0.3, saturation=0.3)

# ── Checkpoint filenames ────────────────────────────────────────────────────────
FINETUNE_CKPT = MODELS_DIR / "finetune_best.pt"
HISTORY_PATH   = LOG_DIR  / "history.pkl"

# ── Device ─────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
