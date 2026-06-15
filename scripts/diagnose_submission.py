import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Profile and evaluate a Software Cup submission predict.py locally.")
    parser.add_argument("--submission-dir", default="submissions/submission_picodet_xs192_build")
    parser.add_argument("--val-list", default="submissions/val_data.txt")
    parser.add_argument("--anno", default="datasets/softcup/annotations/val.json")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--mkldnn", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def load_predict_module(submission_dir):
    predict_path = Path(submission_dir) / "predict.py"
    spec = importlib.util.spec_from_file_location("softcup_predict", predict_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.cv2.imread = imread_unicode
    return module


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_ground_truth(anno_path):
    coco = json.loads(Path(anno_path).read_text(encoding="utf-8"))
    id_to_name = {item["id"]: Path(item["file_name"]).stem for item in coco["images"]}
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_image[id_to_name[ann["image_id"]]].append(
            {
                "type": int(ann["category_id"]),
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "matched": False,
            }
        )
    return gt_by_image


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def evaluate(result_items, gt_by_image, iou_threshold):
    gt_copy = defaultdict(list)
    for image_id, items in gt_by_image.items():
        gt_copy[image_id] = [dict(item, matched=False) for item in items]

    preds_by_image = defaultdict(list)
    for item in result_items:
        x = float(item["x"])
        y = float(item["y"])
        w = float(item["width"])
        h = float(item["height"])
        preds_by_image[item["image_id"]].append(
            {
                "type": int(item["type"]),
                "bbox": [x, y, x + w, y + h],
                "score": float(item.get("score", 1.0)),
            }
        )

    tp = fp = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for image_id, preds in preds_by_image.items():
        preds.sort(key=lambda item: item["score"], reverse=True)
        for pred in preds:
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(gt_copy.get(image_id, [])):
                if gt["matched"] or gt["type"] != pred["type"]:
                    continue
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            cls = pred["type"]
            if best_idx >= 0 and best_iou >= iou_threshold:
                gt_copy[image_id][best_idx]["matched"] = True
                tp += 1
                per_class[cls]["tp"] += 1
            else:
                fp += 1
                per_class[cls]["fp"] += 1

    fn = 0
    for items in gt_copy.values():
        for gt in items:
            if not gt["matched"]:
                fn += 1
                per_class[gt["type"]]["fn"] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for stats in per_class.values():
        p = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
        stats["precision"] = p
        stats["recall"] = r
        stats["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "per_class": dict(per_class)}


def profile_predict(module, image_list, model_dir, threshold, batch_size):
    predictor = module.load_predictor(str(model_dir))
    results = {"result": []}
    timings = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    valid_images = 0

    for start in range(0, len(image_list), batch_size):
        batch_paths = image_list[start : start + batch_size]
        t0 = time.perf_counter()
        preprocessed = module.preprocess_batch(batch_paths)
        valid_paths, image_tensor, im_shape, scale_factor = preprocessed[:4]
        origin_shapes = preprocessed[4] if len(preprocessed) > 4 else None
        timings["preprocess"] += time.perf_counter() - t0
        if image_tensor is None:
            continue
        valid_images += len(valid_paths)

        t0 = time.perf_counter()
        boxes, boxes_num = module.run_predictor(predictor, image_tensor, im_shape, scale_factor)
        timings["inference"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        offset = 0
        for batch_idx, path in enumerate(valid_paths):
            image_id = Path(path).stem
            num = int(boxes_num[batch_idx])
            if num > 0:
                for box in boxes[offset : offset + num]:
                    score = float(box[1])
                    if score < threshold:
                        continue
                    cls_id = int(box[0])
                    x1, y1, x2, y2 = [float(v) for v in box[2:6]]
                    if origin_shapes is not None and hasattr(module, "clip_box"):
                        origin_h, origin_w = origin_shapes[batch_idx]
                        x1, y1, x2, y2 = module.clip_box(x1, y1, x2, y2, origin_h, origin_w)
                        if x2 <= x1 or y2 <= y1:
                            continue
                    obj_type = module.CLASS_ID_TO_TYPE.get(cls_id, cls_id + 1)
                    results["result"].append(
                        {
                            "image_id": image_id,
                            "type": int(obj_type),
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "segmentation": [],
                            "score": score,
                        }
                    )
            offset += num
        timings["postprocess"] += time.perf_counter() - t0
    return results, timings, valid_images


def resolve_local_images(module, val_list):
    image_list = module.get_test_images(val_list)
    fixed = []
    raw_lines = [
        line.strip().replace("\\", "/")
        for line in Path(val_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for path, raw in zip(image_list, raw_lines):
        if Path(path).exists():
            fixed.append(path)
            continue
        cwd_path = Path(raw)
        if not cwd_path.is_absolute():
            cwd_path = Path.cwd() / cwd_path
        fixed.append(str(cwd_path))
    return fixed


def main():
    args = parse_args()
    os.environ["CPU_THREADS"] = str(args.cpu_threads)
    os.environ["USE_MKLDNN"] = "1" if args.mkldnn else "0"
    os.environ["IR_OPTIM"] = "1"

    submission_dir = Path(args.submission_dir)
    module = load_predict_module(submission_dir)
    image_list = resolve_local_images(module, args.val_list)
    gt_by_image = load_ground_truth(args.anno)

    total_start = time.perf_counter()
    results, timings, valid_images = profile_predict(
        module,
        image_list,
        submission_dir / "model",
        args.threshold,
        args.batch_size,
    )
    total_time = time.perf_counter() - total_start
    metrics = evaluate(results["result"], gt_by_image, args.iou_threshold)
    summary = {
        "submission_dir": str(submission_dir),
        "images": valid_images,
        "predictions": len(results["result"]),
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "mkldnn": args.mkldnn,
        "total_time_sec": total_time,
        "fps": valid_images / total_time if total_time > 0 else 0.0,
        "timings_sec": timings,
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
