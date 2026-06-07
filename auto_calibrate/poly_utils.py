"""
poly_utils.py
---------------
폴리곤/OBB 관련 유틸을 한 곳에 모아둔 모듈.
calibrate.py와 polygon_batch_calibrate.py에서 사용하던 함수들을 통합합니다.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import torch

import config


def load_model(ckpt_path: str | Path) -> torch.nn.Module:
    from model import CrosswalkTrimapNet
    model = CrosswalkTrimapNet().to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE,
                                     weights_only=False))
    model.eval()
    return model


def preprocess(image: np.ndarray) -> torch.Tensor:
    """BGR 이미지 → (1, 3, H, W) 정규화 텐서.
    학습과 동일하게 ImageNet mean/std 정규화 적용."""
    h, w = config.IMAGE_SIZE
    img  = cv2.resize(image, (w, h))
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = img.astype(np.float32) / 255.0

    # 학습 시 적용한 ImageNet 정규화 — 반드시 맞춰야 성능 유지됨
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(config.DEVICE)


def predict_trimap(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    """TTA(좌우반전) 평균으로 trimap 예측. 원본 해상도로 반환."""
    orig_h, orig_w = image.shape[:2]
    with torch.no_grad():
        # 원본
        logits = model(preprocess(image))
        # 좌우반전 후 다시 뒤집어 원본 좌표계로 복원
        flipped      = cv2.flip(image, 1)
        logits_flip  = model(preprocess(flipped))
        logits_flip  = torch.flip(logits_flip, dims=[3])
        # 두 예측의 평균
        logits_avg = (logits + logits_flip) / 2
        pred = torch.argmax(logits_avg, dim=1)
        trimap = pred[0].cpu().numpy().astype(np.uint8)

    # 원본 해상도로 복원 (NEAREST: 클래스 레이블 보존)
    trimap = cv2.resize(trimap, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST)
    return trimap


def _morph_mask(mask: np.ndarray, image_h: int, image_w: int,
                ratio: float = 0.04) -> np.ndarray:
    """이미지 크기 대비 비율로 morphology kernel 크기를 결정.
    하드코딩된 kernel_size 대신 해상도에 적응적으로 대응."""
    k = max(3, int(min(image_h, image_w) * ratio))
    k = k if k % 2 == 1 else k + 1   # 홀수 보정
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_obb(trimap: np.ndarray) -> dict | None:
    """Interior(1) 픽셀로 OBB(Oriented Bounding Box)와 hull 교집합 반환."""
    h, w   = trimap.shape
    interior = np.where(trimap == 1, 255, 0).astype(np.uint8)

    # 크기 비례 morphology (열기→닫기로 줄무늬 노이즈 제거)
    k = max(3, int(min(h, w) * 0.01))
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN,  kernel)
    interior = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 최소 면적 기준도 해상도 비례로 설정
    min_area = (h * w) * 0.002
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None

    all_points = np.vstack(valid)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(all_points)

    if rw < rh:
        rw, rh = rh, rw
        angle  = angle - 90 if angle > 0 else angle + 90
    angle   = float(np.clip(angle, -90.0, 0.0))
    obb_pts = cv2.boxPoints(((cx, cy), (rw, rh), angle))
    obb_pts = np.round(obb_pts).astype(np.int32)

    hull = cv2.convexHull(all_points).reshape(-1, 2)
    M    = cv2.moments(hull)
    if M["m00"] != 0:
        cxh = M["m10"] / M["m00"]
        cyh = M["m01"] / M["m00"]
    else:
        cxh = float(np.mean(hull[:, 0]))
        cyh = float(np.mean(hull[:, 1]))

    return {
        "cx":    round(float(cx),    2),
        "cy":    round(float(cy),    2),
        "w":     round(float(rw),    2),
        "h":     round(float(rh),    2),
        "angle": round(angle,        2),
        "obb":   obb_pts,
        "hull":  hull.astype(int),
    }


def extract_polygon(trimap: np.ndarray,
                    approx_epsilon: float = 3.0,
                    min_area: int | None = None) -> np.ndarray | None:
    """Trimap(0/1/2)에서 polygon 추출 → (N, 2) ndarray 반환.

    interior(1) + boundary(2)를 합쳐 마스크를 구성하고 morphology close로
    줄무늬 틈을 메운 뒤 가장 큰 컨투어의 convex hull을 approxPolyDP로 단순화.
    """
    h, w = trimap.shape

    # interior(1)와 boundary(2)를 모두 포함 — 외곽선이 더 정확
    mask = np.where(trimap >= 1, 255, 0).astype(np.uint8)

    # 이미지 크기 비례 morphology
    mask = _morph_mask(mask, h, w, ratio=0.04)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # 최소 면적 기준을 해상도 비례로 설정
    _min_area = min_area if min_area is not None else max(500, (h * w) * 0.002)
    valid = [c for c in contours if cv2.contourArea(c) > _min_area]
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)
    hull    = cv2.convexHull(largest)
    poly    = cv2.approxPolyDP(hull, epsilon=approx_epsilon, closed=True)
    return poly.reshape(-1, 2)


def visualize(image: np.ndarray, trimap: np.ndarray,
              obb: dict | None, out_path: str) -> None:
    h, w       = image.shape[:2]
    font_scale = max(0.5, min(w, h) / 800)
    thick_box  = max(2, min(w, h) // 300)

    vis_trimap               = np.zeros((h, w, 3), dtype=np.uint8)
    vis_trimap[trimap == 0]  = [0,   0,   0]
    vis_trimap[trimap == 1]  = [0, 200,   0]
    vis_trimap[trimap == 2]  = [0,   0, 255]

    orig_obb = image.copy()
    overlay  = image.copy()

    if obb is not None:
        try:
            from shapely.geometry import Polygon
            obb_poly  = Polygon(obb["obb"])
            hull_poly = Polygon(obb["hull"])
            inter     = obb_poly.intersection(hull_poly)
            inter_pts = None
            if not inter.is_empty and inter.geom_type == "Polygon":
                inter_pts = np.array(inter.exterior.coords,
                                     dtype=np.int32).reshape(-1, 1, 2)
        except Exception:
            inter_pts = None

        if inter_pts is not None:
            cv2.polylines(orig_obb, [inter_pts], isClosed=True,
                          color=(0, 215, 255), thickness=thick_box)
            cv2.fillPoly(overlay, [inter_pts], color=(0, 255, 0))
            blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
            cv2.polylines(blended, [inter_pts], isClosed=True,
                          color=(0, 215, 255), thickness=thick_box)
        else:
            blended = image.copy()

        cx, cy = int(obb["cx"]), int(obb["cy"])
        cv2.circle(orig_obb, (cx, cy), max(6, thick_box * 3), (0, 0, 255), -1)
        label = (f"OBB∩Hull  {int(obb['w'])}x{int(obb['h'])}"
                 f"  {obb['angle']:.1f}deg")
        cv2.putText(orig_obb, label, (10, int(40 * font_scale * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 215, 255), thick_box)
    else:
        blended = image.copy()
        cv2.putText(orig_obb, "NO CROSSWALK DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    combined = np.hstack([orig_obb, vis_trimap, blended])
    cv2.imwrite(out_path, combined)


def visualize_polygon(image: np.ndarray, trimap: np.ndarray,
                      polygon: np.ndarray | None) -> np.ndarray:
    h, w = image.shape[:2]

    vis_trimap              = np.zeros((h, w, 3), dtype=np.uint8)
    vis_trimap[trimap == 0] = [0,   0,   0]
    vis_trimap[trimap == 1] = [0, 200,   0]
    vis_trimap[trimap == 2] = [0,   0, 255]

    img_poly = image.copy()
    overlay  = image.copy()

    if polygon is not None:
        poly_pts = polygon.reshape(-1, 1, 2)
        cv2.polylines(img_poly, [poly_pts], isClosed=True,
                      color=(0, 0, 255), thickness=3)
        cv2.fillPoly(overlay, [poly_pts], color=(0, 255, 0))
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    else:
        cv2.putText(img_poly, "NO CROSSWALK DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        blended = image.copy()

    return np.hstack([img_poly, vis_trimap, blended])


def save_calibration(obb: dict | None, source: str, out_path: str) -> None:
    result = {
        "source":   source,
        "detected": obb is not None,
        "obb": {
            "cx":     obb["cx"],
            "cy":     obb["cy"],
            "w":      obb["w"],
            "h":      obb["h"],
            "angle":  obb["angle"],
            "points": obb["obb"].tolist(),
        } if obb is not None else None,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


def save_polygon(polygon: np.ndarray | None,
                 source: str, out_path: str) -> None:
    result = {
        "source":   source,
        "detected": polygon is not None,
        "polygon":  polygon.tolist() if polygon is not None else None,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
