import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


DEFAULT_CLASSES = ["battery", "fire", "board"]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Software Cup LabelMe data to COCO.")
    parser.add_argument("--raw-dir", default="A_train", help="Raw dataset directory.")
    parser.add_argument("--out-dir", default="datasets/softcup", help="Output COCO dataset directory.")
    parser.add_argument("--classes", default="configs/classes.txt", help="Class list, one class per line.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic split.")
    parser.add_argument("--no-copy-images", action="store_true", help="Do not copy images into out-dir/images.")
    return parser.parse_args()


def load_classes(path: Path):
    if not path.exists():
        return DEFAULT_CLASSES
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return classes or DEFAULT_CLASSES


def rectangle_to_bbox(points, width, height):
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def build_coco(records, categories):
    images = []
    annotations = []
    ann_id = 1
    class_to_id = {item["name"]: item["id"] for item in categories}

    for image_id, record in enumerate(records, start=1):
        data = record["label"]
        file_name = record["image"].name
        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in class_to_id:
                continue
            bbox = rectangle_to_bbox(shape.get("points", []), width, height)
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": class_to_id[label],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    image_dir = raw_dir / "Image"
    label_dir = raw_dir / "label"
    out_dir = Path(args.out_dir)

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    classes = load_classes(Path(args.classes))
    categories = [{"id": idx + 1, "name": name, "supercategory": "object"} for idx, name in enumerate(classes)]

    records = []
    for label_path in sorted(label_dir.glob("*.json")):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        image_name = data.get("imagePath") or f"{label_path.stem}.jpg"
        image_path = image_dir / Path(image_name).name
        if not image_path.exists():
            image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            print(f"[WARN] missing image for {label_path.name}, skipped")
            continue
        records.append({"image": image_path, "label_path": label_path, "label": data})

    random.Random(args.seed).shuffle(records)
    val_count = max(1, int(round(len(records) * args.val_ratio)))
    val_records = sorted(records[:val_count], key=lambda item: item["image"].name)
    train_records = sorted(records[val_count:], key=lambda item: item["image"].name)

    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    if not args.no_copy_images:
        for record in tqdm(records, desc="Copy images"):
            target = out_dir / "images" / record["image"].name
            if not target.exists():
                shutil.copy2(record["image"], target)

    (out_dir / "annotations" / "train.json").write_text(
        json.dumps(build_coco(train_records, categories), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "annotations" / "val.json").write_text(
        json.dumps(build_coco(val_records, categories), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "label_list.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    print(f"Converted {len(records)} images")
    print(f"Train: {len(train_records)} images")
    print(f"Val:   {len(val_records)} images")
    print(f"Classes: {', '.join(classes)}")


if __name__ == "__main__":
    main()
