import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import pick_best_frame
from poly_utils import (
    load_model,
    predict_trimap,
    extract_polygon,
    save_polygon,
    visualize_polygon,
)
import cv2


def main():
    import argparse
    p = argparse.ArgumentParser(description="Batch crosswalk polygon segmentation extractor")
    p.add_argument("--batch", help="Process all images/videos in a folder")
    p.add_argument("--out_dir", default="polygon_output_batch", help="Output directory")
    p.add_argument("--ckpt", default=None, help="Checkpoint path (default: ./base_best.pt)")
    args = p.parse_args()

    if args.ckpt:
        config.BASE_CKPT = Path(args.ckpt)
    # otherwise use default config.BASE_CKPT (now pointing to models/base_best.pt)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    batch_dir = Path(args.batch)
    if not batch_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {batch_dir}")

    files = sorted(
        p for p in batch_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    )
    if not files:
        print(f"[Warning] No image/video files to process: {batch_dir}")
        sys.exit(0)

    out_root = Path(args.out_dir)
    print(f"[Batch] Processing {len(files)} files → {out_root}")
    model = load_model(config.BASE_CKPT)
    summary = []

    for idx, fpath in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] {fpath.name}")
        out_sub = out_root / fpath.stem
        out_sub.mkdir(parents=True, exist_ok=True)

        if fpath.suffix.lower() in VIDEO_EXTS:
            print(f"  Selecting best frame from video...")
            image = pick_best_frame(str(fpath))
            if image is None:
                print(f"  [Error] Failed to extract frame: {fpath}")
                summary.append({"file": fpath.name, "detected": False, "error": "Frame extraction failed"})
                continue
        else:
            image = cv2.imread(str(fpath))
            if image is None:
                print(f"  [Error] Failed to read image: {fpath}")
                summary.append({"file": fpath.name, "detected": False, "error": "Image read failed"})
                continue

        trimap = predict_trimap(model, image)
        polygon = extract_polygon(trimap)

        save_polygon(polygon, str(fpath), str(out_sub / "polygon_calibration.json"))
        vis = visualize_polygon(image, trimap, polygon)
        cv2.imwrite(str(out_sub / "polygon_calibration_vis.jpg"), vis)
        print(f"  [Saved] Visualization image: {out_sub / 'polygon_calibration_vis.jpg'}")

        summary.append({"file": fpath.name, "detected": polygon is not None})

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Detection success: {sum(1 for s in summary if s.get('detected'))}/{len(files)} | Summary: {summary_path}")


if __name__ == "__main__":
    main()