import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Create PaddleDetection config for the Software Cup baseline.")
    parser.add_argument("--paddledet-dir", default="external/PaddleDetection", help="PaddleDetection checkout path.")
    parser.add_argument("--dataset-dir", default="datasets/softcup", help="Prepared COCO dataset path.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of detection classes.")
    parser.add_argument("--epoch", type=int, default=80, help="Training epochs.")
    parser.add_argument("--snapshot-epoch", type=int, default=5, help="Checkpoint interval.")
    return parser.parse_args()


def main():
    args = parse_args()
    paddledet_dir = Path(args.paddledet_dir)
    dataset_dir = Path(args.dataset_dir).resolve()
    config_dir = paddledet_dir / "configs" / "softcup"
    config_dir.mkdir(parents=True, exist_ok=True)

    if not paddledet_dir.exists():
        raise FileNotFoundError(f"{paddledet_dir} does not exist. Clone PaddleDetection first.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} does not exist. Run scripts/prepare_dataset.py first.")

    dataset_config = {
        "metric": "COCO",
        "num_classes": args.num_classes,
        "TrainDataset": {
            "name": "COCODataSet",
            "image_dir": "images",
            "anno_path": "annotations/train.json",
            "dataset_dir": str(dataset_dir).replace("\\", "/"),
            "data_fields": ["image", "gt_bbox", "gt_class", "is_crowd"],
        },
        "EvalDataset": {
            "name": "COCODataSet",
            "image_dir": "images",
            "anno_path": "annotations/val.json",
            "dataset_dir": str(dataset_dir).replace("\\", "/"),
        },
        "TestDataset": {
            "name": "ImageFolder",
            "anno_path": str((dataset_dir / "annotations" / "val.json").resolve()).replace("\\", "/"),
        },
    }

    model_config = {
        "_BASE_": [
            "../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml",
            "./softcup_coco.yml",
        ],
        "epoch": args.epoch,
        "snapshot_epoch": args.snapshot_epoch,
        "weights": "https://bj.bcebos.com/v1/paddledet/models/ppyoloe_plus_crn_s_80e_coco.pdparams",
        "PPYOLOEHead": {"num_classes": args.num_classes},
    }

    dataset_path = config_dir / "softcup_coco.yml"
    model_path = config_dir / "ppyoloe_plus_crn_s_80e_softcup.yml"
    dataset_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model_path.write_text(yaml.safe_dump(model_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Wrote {dataset_path}")
    print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()

