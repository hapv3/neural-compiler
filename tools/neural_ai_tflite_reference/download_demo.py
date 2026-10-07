"""Download 100 validation images + labels from detection-datasets/coco, not the full dataset."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = "detection-datasets/coco"


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise


def main():
    destination = ROOT / "data/coco_val100"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for sample in manifest["samples"]:
            assert hashlib.sha256((destination / sample["file"]).read_bytes()).hexdigest() == sample["sha256"]
        print(f"Verified existing {len(manifest['samples'])} images: {destination}")
        return
    url = f"https://datasets-server.huggingface.co/first-rows?dataset={DATASET}&config=default&split=val"
    data = json.loads(fetch(url))
    rows = data["rows"][:100]
    if len(rows) != 100 or any(row.get("truncated_cells") for row in rows):
        raise RuntimeError("Need 100 complete image/annotation rows; refusing partial benchmark")
    classes = next(f for f in data["features"] if f["name"] == "objects")["type"]["feature"]["category"]["names"]
    info = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET}"))

    def download(item):
        row = item["row"]
        filename = f"{row['image_id']:012d}.jpg"
        raw = fetch(row["image"]["src"])
        with Image.open(BytesIO(raw)) as image:
            if image.size != (row["width"], row["height"]):
                raise RuntimeError(f"Viewer thumbnail is not full resolution: {filename}")
            image.verify()
        objects = row["objects"]
        if len({len(objects[key]) for key in ("bbox_id", "category", "bbox", "area")}) != 1:
            raise ValueError(f"Incomplete labels: {filename}")
        (destination / filename).write_bytes(raw)
        return {"row_index": item["row_idx"], "image_id": row["image_id"], "file": filename,
                "width": row["width"], "height": row["height"], "objects": objects,
                "sha256": hashlib.sha256(raw).hexdigest()}

    samples = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for sample in pool.map(download, rows):
            samples.append(sample)
            if len(samples) % 10 == 0:
                print(f"Downloaded {len(samples)}/100", flush=True)
    manifest = {"dataset": DATASET, "source": f"https://huggingface.co/datasets/{DATASET}",
                "revision_at_download": info["sha"], "split": "val", "selection": "first 100 viewer rows",
                "bbox_format": "xyxy absolute original-image coordinates", "category_ids": "contiguous 0..79",
                "limitations": "Dataset omits iscrowd; subset evaluation treats all supplied boxes as non-crowd.",
                "classes": classes, "samples": samples}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(samples)} images, {sum(len(s['objects']['bbox']) for s in samples)} boxes: {destination}")


if __name__ == "__main__":
    main()
