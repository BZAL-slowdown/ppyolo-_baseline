# -*- coding: utf-8 -*-
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import paddle
import yaml
from paddle.inference import Config, create_predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "PaddleDetection"))

from PaddleDetection.deploy.python.preprocess import NormalizeImage, PadStride, Permute, Resize, preprocess


CLASS_ID_TO_TYPE = {0: 1, 1: 2, 2: 3}


def get_test_images(infer_file):
    infer_dir = os.path.dirname(os.path.abspath(infer_file))
    images = []
    with open(infer_file, "r") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if not line:
                continue
            if not os.path.isabs(line):
                line = os.path.join(infer_dir, line)
            images.append(line)
    return images


def frame_no(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


class PredictConfig:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "infer_cfg.yml")) as f:
            yml_conf = yaml.safe_load(f)
        self.preprocess_infos = yml_conf["Preprocess"]


def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams"),
    )
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(int(os.environ.get("CPU_THREADS", "4")))
    if os.environ.get("USE_MKLDNN", "0") == "1":
        config.enable_mkldnn()
    config.switch_ir_optim(False)
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    return create_predictor(config)


class Detector:
    def __init__(self, model_dir):
        self.pred_config = PredictConfig(model_dir)
        self.predictor = load_predictor(model_dir)
        self.preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            info = op_info.copy()
            op_type = info.pop("type")
            self.preprocess_ops.append(eval(op_type)(**info))

    def detect(self, image_path, threshold):
        im, im_info = preprocess(image_path, self.preprocess_ops)
        inputs = {
            "image": np.expand_dims(im, axis=0).astype("float32"),
            "im_shape": np.array([im_info["im_shape"]], dtype="float32"),
            "scale_factor": np.array([im_info["scale_factor"]], dtype="float32"),
        }
        for name in self.predictor.get_input_names():
            self.predictor.get_input_handle(name).copy_from_cpu(inputs[name])
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        boxes = self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(output_names[num_outs]).copy_to_cpu()
        out = []
        for i in range(int(boxes_num[0])):
            cls_id = int(boxes[i][0])
            score = float(boxes[i][1])
            if score < threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes[i][2:6]]
            out.append(
                {
                    "type": CLASS_ID_TO_TYPE.get(cls_id, cls_id + 1),
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
        return out


def clamp_box(box, width, height):
    x = max(0.0, min(float(box["x"]), width - 1.0))
    y = max(0.0, min(float(box["y"]), height - 1.0))
    w = max(2.0, min(float(box["width"]), width - x))
    h = max(2.0, min(float(box["height"]), height - y))
    return {"type": box["type"], "x": x, "y": y, "width": w, "height": h}


def track_boxes(prev_img, cur_img, prev_boxes):
    if prev_img is None or cur_img is None or not prev_boxes:
        return []
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    height, width = cur_gray.shape[:2]
    tracked = []
    for box in prev_boxes:
        b = clamp_box(box, width, height)
        x, y, w, h = [int(round(b[k])) for k in ("x", "y", "width", "height")]
        if w < 8 or h < 8:
            continue
        tpl = prev_gray[y : y + h, x : x + w]
        if tpl.size == 0:
            continue
        margin = int(max(24, min(90, max(w, h) * 0.45)))
        sx1 = max(0, x - margin)
        sy1 = max(0, y - margin)
        sx2 = min(width, x + w + margin)
        sy2 = min(height, y + h + margin)
        search = cur_gray[sy1:sy2, sx1:sx2]
        if search.shape[0] < h or search.shape[1] < w:
            tracked.append(b)
            continue
        res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < 0.25:
            tracked.append(b)
            continue
        nx = sx1 + loc[0]
        ny = sy1 + loc[1]
        tracked.append(clamp_box({"type": b["type"], "x": nx, "y": ny, "width": w, "height": h}, width, height))
    return tracked


def to_json(image_id, box):
    return {
        "image_id": image_id,
        "type": int(box["type"]),
        "x": float(box["x"]),
        "y": float(box["y"]),
        "width": float(box["width"]),
        "height": float(box["height"]),
        "segmentation": [],
    }


def main(infer_txt, result_path):
    image_list = get_test_images(infer_txt)
    detector = Detector(os.path.join(BASE_DIR, "model"))
    threshold = float(os.environ.get("THRESHOLD", "0.5"))
    interval = int(os.environ.get("DETECT_INTERVAL", "8"))
    max_gap = int(os.environ.get("MAX_TRACK_GAP", "50"))

    results = {"result": []}
    prev_img = None
    prev_boxes = []
    last_det_index = -10**9
    last_frame = -10**9

    for idx, image_path in enumerate(image_list):
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        cur_img = cv2.imread(image_path)
        cur_frame = frame_no(image_path)
        should_detect = (
            idx == 0
            or idx - last_det_index >= interval
            or cur_frame < 0
            or last_frame < 0
            or cur_frame - last_frame > max_gap
            or not prev_boxes
        )
        if should_detect:
            boxes = detector.detect(image_path, threshold)
            last_det_index = idx
        else:
            boxes = track_boxes(prev_img, cur_img, prev_boxes)
        for box in boxes:
            results["result"].append(to_json(image_id, box))
        prev_img = cur_img
        prev_boxes = boxes
        last_frame = cur_frame

    with open(result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    start = time.time()
    paddle.enable_static()
    main(sys.argv[1], sys.argv[2])
    print("total time:", time.time() - start)
