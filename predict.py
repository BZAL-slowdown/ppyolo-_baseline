import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def existing_path(value):
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def find_image_input(root):
    candidates = [
        "test",
        "images",
        "image",
        "input",
        "data",
        "A_test/Image",
        "A_test/images",
    ]
    for item in candidates:
        path = root / item
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return path
        if path.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir()):
            return path
    return root


def normalize_output(path):
    if not path:
        return Path("output")
    output = Path(path)
    if output.suffix.lower() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("positional", nargs="*")
    parser.add_argument("--image_dir")
    parser.add_argument("--image_file")
    parser.add_argument("--input")
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--output")
    parser.add_argument("--result_path")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "CPU"))
    parser.add_argument("--threshold", default="0.5")
    args, unknown = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    model_dir = root / "model"
    infer_py = root / "PaddleDetection" / "deploy" / "python" / "infer.py"

    image_file = existing_path(args.image_file)
    image_dir = existing_path(args.image_dir or args.input_dir or args.input)
    result_target = args.result_path or args.output or args.output_dir

    if args.positional:
        first = existing_path(args.positional[0])
        if first:
            if first.is_file():
                image_file = first
            else:
                image_dir = first
        if len(args.positional) > 1 and not result_target:
            result_target = args.positional[1]

    if image_file is None and image_dir is None:
        inferred = find_image_input(Path.cwd())
        if inferred.is_file():
            image_file = inferred
        else:
            image_dir = inferred

    output = normalize_output(result_target)
    output_dir = output.parent if output.suffix.lower() == ".json" else output

    cmd = [
        sys.executable,
        str(infer_py),
        "--model_dir",
        str(model_dir),
        "--device",
        args.device.upper(),
        "--threshold",
        str(args.threshold),
        "--save_results",
        "--save_images",
        "False",
        "--output_dir",
        str(output_dir),
    ]
    if image_file:
        cmd.extend(["--image_file", str(image_file)])
    else:
        cmd.extend(["--image_dir", str(image_dir)])
    cmd.extend(unknown)

    subprocess.run(cmd, check=True)

    bbox_json = output_dir / "bbox.json"
    if output.suffix.lower() == ".json" and bbox_json.exists() and bbox_json != output:
        shutil.copyfile(bbox_json, output)


if __name__ == "__main__":
    main()
