# -*- coding: utf-8 -*-
import json
import os
import sys
import time

import cv2
import numpy as np


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


def add_mask(mask, top_ratio=0.20, bottom_ratio=0.72):
    h = mask.shape[0]
    mask[:int(top_ratio * h), :] = 0
    mask[int(bottom_ratio * h):, :] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_image(path):
    img = cv2.imread(path)
    if img is None:
        return []
    height, width = img.shape[:2]
    scale = 0.5
    small = cv2.resize(
        img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    masks = []
    masks.append(
        (
            1,
            cv2.inRange(hsv, np.array([35, 25, 120]), np.array([85, 180, 255])),
        )
    )
    fire_a = cv2.inRange(hsv, np.array([0, 45, 120]), np.array([35, 255, 255]))
    fire_b = cv2.inRange(hsv, np.array([165, 35, 120]), np.array([179, 255, 255]))
    masks.append((2, cv2.bitwise_or(fire_a, fire_b)))
    masks.append(
        (
            3,
            cv2.inRange(hsv, np.array([85, 10, 100]), np.array([125, 100, 245])),
        )
    )

    results = []
    inv = 1.0 / scale
    for obj_type, mask in masks:
        mask = add_mask(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 40 or area > 60000 or h <= 0:
                continue
            aspect = float(w) / float(h)
            if obj_type == 1 and not (0.12 <= aspect <= 1.20 and h >= 12):
                continue
            if obj_type == 2 and not (0.20 <= aspect <= 2.20 and h >= 10):
                continue
            if obj_type == 3 and not (0.35 <= aspect <= 2.00 and w >= 15 and h >= 15):
                continue
            pad = 3
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(mask.shape[1] - x, w + 2 * pad)
            h = min(mask.shape[0] - y, h + 2 * pad)
            results.append(
                {
                    "type": obj_type,
                    "x": float(x * inv),
                    "y": float(y * inv),
                    "width": float(w * inv),
                    "height": float(h * inv),
                }
            )
    return results


def main(infer_txt, result_path):
    output = {"result": []}
    for image_path in get_test_images(infer_txt):
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        for det in detect_image(image_path):
            output["result"].append(
                {
                    "image_id": image_id,
                    "type": det["type"],
                    "x": det["x"],
                    "y": det["y"],
                    "width": det["width"],
                    "height": det["height"],
                    "segmentation": [],
                }
            )
    with open(result_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1], sys.argv[2])
    print("total time:", time.time() - start)
