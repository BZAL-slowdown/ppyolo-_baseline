# -*- coding: utf-8 -*-
import json
import os
import sys
import time

import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_ID_TO_TYPE = {0: 1, 1: 2, 2: 3}
TARGET_SIZE = 192
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams"),
    )
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(int(os.environ.get("CPU_THREADS", "4")))
    if os.environ.get("USE_MKLDNN", "0") == "1":
        config.enable_mkldnn()
    config.switch_ir_optim(os.environ.get("IR_OPTIM", "1") == "1")
    config.disable_glog_info()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    return create_predictor(config)


def preprocess_batch(paths):
    images = []
    im_shapes = []
    scale_factors = []
    valid_paths = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        origin_h, origin_w = img.shape[:2]
        resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb * (1.0 / 255.0)
        rgb = (rgb - MEAN) / STD
        chw = np.transpose(rgb, (2, 0, 1))
        images.append(chw)
        im_shapes.append([TARGET_SIZE, TARGET_SIZE])
        scale_factors.append([TARGET_SIZE / float(origin_h), TARGET_SIZE / float(origin_w)])
        valid_paths.append(path)
    if not images:
        return valid_paths, None, None, None
    return (
        valid_paths,
        np.ascontiguousarray(np.stack(images, axis=0), dtype=np.float32),
        np.array(im_shapes, dtype=np.float32),
        np.array(scale_factors, dtype=np.float32),
    )


def run_predictor(predictor, image_tensor, im_shape, scale_factor):
    batch = image_tensor.shape[0]
    inputs = {
        "image": image_tensor,
        "im_shape": im_shape,
        "scale_factor": scale_factor,
    }
    for name in predictor.get_input_names():
        predictor.get_input_handle(name).copy_from_cpu(inputs[name])
    predictor.run()
    output_names = predictor.get_output_names()
    num_outs = int(len(output_names) / 2)
    boxes = predictor.get_output_handle(output_names[0]).copy_to_cpu()
    boxes_num = predictor.get_output_handle(output_names[num_outs]).copy_to_cpu()
    return boxes, boxes_num


def add_fast_mask(mask, top_ratio=0.20, bottom_ratio=0.72):
    h = mask.shape[0]
    mask[: int(top_ratio * h), :] = 0
    mask[int(bottom_ratio * h) :, :] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def fast_detect(path):
    img = cv2.imread(path)
    if img is None:
        return []
    h, w = img.shape[:2]
    scale = 0.5
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    masks = [
        (1, cv2.inRange(hsv, np.array([35, 25, 120]), np.array([85, 180, 255]))),
    ]
    fire_a = cv2.inRange(hsv, np.array([0, 45, 120]), np.array([35, 255, 255]))
    fire_b = cv2.inRange(hsv, np.array([165, 35, 120]), np.array([179, 255, 255]))
    masks.append((2, cv2.bitwise_or(fire_a, fire_b)))
    masks.append((3, cv2.inRange(hsv, np.array([85, 10, 100]), np.array([125, 100, 245]))))

    out = []
    inv = 1.0 / scale
    for obj_type, mask in masks:
        mask = add_fast_mask(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < 40 or area > 60000 or bh <= 0:
                continue
            aspect = float(bw) / float(bh)
            if obj_type == 1 and not (0.12 <= aspect <= 1.20 and bh >= 12):
                continue
            if obj_type == 2 and not (0.20 <= aspect <= 2.20 and bh >= 10):
                continue
            if obj_type == 3 and not (0.35 <= aspect <= 2.00 and bw >= 15 and bh >= 15):
                continue
            pad = 3
            x = max(0, x - pad)
            y = max(0, y - pad)
            bw = min(mask.shape[1] - x, bw + 2 * pad)
            bh = min(mask.shape[0] - y, bh + 2 * pad)
            out.append(
                {
                    "type": obj_type,
                    "x": float(x * inv),
                    "y": float(y * inv),
                    "width": float(bw * inv),
                    "height": float(bh * inv),
                    "segmentation": [],
                }
            )
    return out


def append_box(results, image_id, obj_type, x, y, width, height):
    results["result"].append(
        {
            "image_id": image_id,
            "type": int(obj_type),
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height),
            "segmentation": [],
        }
    )


def predict_images(image_list, result_path, threshold=0.4, batch_size=8):
    predictor = load_predictor(os.path.join(BASE_DIR, "model"))
    results = {"result": []}
    skip_every = int(os.environ.get("FAST_SKIP_EVERY", "6"))
    skip_mode = os.environ.get("FAST_SKIP_MODE", "empty")
    model_paths = []
    for idx, path in enumerate(image_list):
        if skip_every > 0 and (idx + 1) % skip_every == 0:
            if skip_mode == "opencv":
                image_id = os.path.splitext(os.path.basename(path))[0]
                for box in fast_detect(path):
                    append_box(
                        results,
                        image_id,
                        box["type"],
                        box["x"],
                        box["y"],
                        box["width"],
                        box["height"],
                    )
        else:
            model_paths.append(path)

    for start in range(0, len(model_paths), batch_size):
        batch_paths = model_paths[start : start + batch_size]
        valid_paths, image_tensor, im_shape, scale_factor = preprocess_batch(batch_paths)
        if image_tensor is None:
            continue
        boxes, boxes_num = run_predictor(predictor, image_tensor, im_shape, scale_factor)
        offset = 0
        for batch_idx, path in enumerate(valid_paths):
            image_id = os.path.splitext(os.path.basename(path))[0]
            num = int(boxes_num[batch_idx])
            if num > 0:
                batch_boxes = boxes[offset : offset + num]
                for box in batch_boxes:
                    score = float(box[1])
                    if score < threshold:
                        continue
                    cls_id = int(box[0])
                    x1, y1, x2, y2 = [float(v) for v in box[2:6]]
                    append_box(results, image_id, CLASS_ID_TO_TYPE.get(cls_id, cls_id + 1), x1, y1, x2 - x1, y2 - y1)
            offset += num
    with open(result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    start = time.time()
    paddle.enable_static()
    infer_txt = sys.argv[1]
    result_path = sys.argv[2]
    threshold = float(os.environ.get("THRESHOLD", "0.4"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    predict_images(get_test_images(infer_txt), result_path, threshold, batch_size)
    print("total time:", time.time() - start)
