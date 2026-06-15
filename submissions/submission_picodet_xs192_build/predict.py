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


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
                if not os.path.exists(line):
                    cwd_line = os.path.abspath(line.replace(infer_dir, os.getcwd(), 1))
                    if os.path.exists(cwd_line):
                        line = cwd_line
            images.append(line)
    return images


def load_predictor(model_dir):
    model_file = os.path.join(model_dir, "model.pdmodel")
    params_file = os.path.join(model_dir, "model.pdiparams")
    try:
        model_file = os.path.relpath(model_file, os.getcwd())
        params_file = os.path.relpath(params_file, os.getcwd())
    except ValueError:
        pass
    config = Config(
        model_file,
        params_file,
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
    origin_shapes = []
    valid_paths = []
    for path in paths:
        img = read_image(path)
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
        origin_shapes.append([origin_h, origin_w])
        valid_paths.append(path)
    if not images:
        return valid_paths, None, None, None, None
    return (
        valid_paths,
        np.ascontiguousarray(np.stack(images, axis=0), dtype=np.float32),
        np.array(im_shapes, dtype=np.float32),
        np.array(scale_factors, dtype=np.float32),
        origin_shapes,
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


def clip_box(x1, y1, x2, y2, origin_h, origin_w):
    x1 = min(max(float(x1), 0.0), float(origin_w))
    y1 = min(max(float(y1), 0.0), float(origin_h))
    x2 = min(max(float(x2), 0.0), float(origin_w))
    y2 = min(max(float(y2), 0.0), float(origin_h))
    return x1, y1, x2, y2


def predict_images(image_list, result_path, threshold=0.4, batch_size=8):
    predictor = load_predictor(os.path.join(BASE_DIR, "model"))
    results = {"result": []}
    model_paths = image_list
    for start in range(0, len(model_paths), batch_size):
        batch_paths = model_paths[start : start + batch_size]
        valid_paths, image_tensor, im_shape, scale_factor, origin_shapes = preprocess_batch(batch_paths)
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
                    origin_h, origin_w = origin_shapes[batch_idx]
                    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, origin_h, origin_w)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    append_box(results, image_id, CLASS_ID_TO_TYPE.get(cls_id, cls_id + 1), x1, y1, x2 - x1, y2 - y1)
            offset += num
    with open(result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    start = time.time()
    paddle.enable_static()
    infer_txt = sys.argv[1]
    result_path = sys.argv[2]
    threshold = float(os.environ.get("THRESHOLD", "0.45"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    predict_images(get_test_images(infer_txt), result_path, threshold, batch_size)
    print("total time:", time.time() - start)
