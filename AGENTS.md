# AGENTS.md

## Project Goal

This repository is the preliminary-round baseline for the 2026 China Software Cup "Wenxin Xiaowang" PaddlePaddle competition. The local baseline uses PaddleDetection PP-YOLO/PP-YOLOE-style object detection on the provided LabelMe-format dataset.

## Current Dataset

- Raw data lives in `A_train/`.
- Images: `A_train/Image/*.jpg`
- Labels: `A_train/label/*.json`
- Annotation format: LabelMe-style rectangles.
- Known classes, in stable order:
  1. `battery`
  2. `fire`
  3. `board`

Do not commit raw images, generated COCO datasets, model checkpoints, inference results, or downloaded PaddleDetection sources unless the user explicitly asks for that.

## Baseline Workflow

1. Convert and split the raw dataset:

   ```powershell
   python scripts/prepare_dataset.py --raw-dir A_train --out-dir datasets/softcup --val-ratio 0.2 --seed 2026
   ```

2. Install/clone PaddleDetection separately:

   ```powershell
   git clone https://github.com/PaddlePaddle/PaddleDetection.git external/PaddleDetection
   pip install -r requirements.txt
   pip install -r external/PaddleDetection/requirements.txt
   ```

3. Generate the PaddleDetection config:

   ```powershell
   python scripts/create_paddledet_config.py --paddledet-dir external/PaddleDetection --dataset-dir datasets/softcup
   ```

4. Train:

   ```powershell
   python external/PaddleDetection/tools/train.py -c external/PaddleDetection/configs/softcup/ppyoloe_plus_crn_s_80e_softcup.yml --eval
   ```

5. Evaluate:

   ```powershell
   python external/PaddleDetection/tools/eval.py -c external/PaddleDetection/configs/softcup/ppyoloe_plus_crn_s_80e_softcup.yml -o weights=output/ppyoloe_plus_crn_s_80e_softcup/best_model.pdparams
   ```

6. Predict or export as needed by the competition submission instructions.

## Editing Rules For Future Agents

- Prefer small, reproducible scripts over notebook-only steps.
- Keep class order fixed unless the dataset analysis proves the competition requires another order.
- Keep generated files under ignored directories such as `datasets/`, `output/`, `inference_model/`, and `runs/`.
- Use `rg` for repository search when available.
- Use `apply_patch` for source edits.
- Never delete or rewrite raw data in `A_train/`.
- Before changing training hyperparameters, record the reason in `README.md` or a new experiment note.

