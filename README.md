# PP-YOLO+ Baseline for Software Cup

This repository contains a reproducible preliminary-round baseline for the Software Cup PaddlePaddle object detection task. The provided raw dataset is LabelMe-style JSON annotations plus JPG images. The baseline converts the data to COCO, creates a train/validation split, and trains a PaddleDetection PP-YOLOE+ model.

## Dataset

Raw files are expected locally under:

```text
A_train/
  Image/*.jpg
  label/*.json
```

Detected classes:

```text
battery
fire
board
```

The repository intentionally ignores `A_train/`, generated datasets, checkpoints, and downloaded PaddleDetection sources.

## Quick Start

```powershell
python scripts/prepare_dataset.py --raw-dir A_train --out-dir datasets/softcup --val-ratio 0.2 --seed 2026
git clone https://github.com/PaddlePaddle/PaddleDetection.git external/PaddleDetection
pip install -r requirements.txt
pip install -r external/PaddleDetection/requirements.txt
python scripts/create_paddledet_config.py --paddledet-dir external/PaddleDetection --dataset-dir datasets/softcup
python external/PaddleDetection/tools/train.py -c external/PaddleDetection/configs/softcup/ppyoloe_plus_crn_s_80e_softcup.yml --eval
```

After training, evaluate the best checkpoint:

```powershell
python external/PaddleDetection/tools/eval.py -c external/PaddleDetection/configs/softcup/ppyoloe_plus_crn_s_80e_softcup.yml -o weights=output/ppyoloe_plus_crn_s_80e_softcup/best_model.pdparams
```

## Project Layout

```text
scripts/prepare_dataset.py              Convert LabelMe JSON to COCO and split data.
scripts/create_paddledet_config.py      Create PaddleDetection dataset and model config.
configs/classes.txt                     Stable class list for conversion and training.
configs/softcup_coco.yml                PaddleDetection dataset config template.
configs/ppyoloe_plus_crn_s_80e_softcup.yml  PaddleDetection PP-YOLOE+ config template.
```

If the PaddleDetection repository is slow to clone, keep using the scripts and copy the two YAML templates into `external/PaddleDetection/configs/softcup/` after the clone finishes.

## Notes

- The split keeps empty-label images, because they are useful negative samples for object detection.
- Default split is deterministic with seed `2026`.
- The default model is `ppyoloe_plus_crn_s_80e`, a practical starting point for a small dataset.
- Increase epochs, image size, or switch to a larger PP-YOLOE+ backbone only after the baseline trains cleanly.
