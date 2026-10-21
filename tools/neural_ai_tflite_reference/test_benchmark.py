"""Check output comparison and the dataset evaluation adapter independently."""

from pathlib import Path
import tempfile
import unittest
import hashlib

import numpy as np

from benchmark import decode_yolo, difference, evaluate_coco
from benchmark_dfl import aligned_iou, require_cached_output


class BenchmarkTest(unittest.TestCase):
    def test_difference_signed_bytes(self):
        result = difference(np.array([-128, 127], np.int8), np.array([127, -128], np.int8))
        self.assertEqual(result["mismatches"], 2)
        self.assertEqual(result["max_abs_lsb"], 255)
        self.assertEqual(result["mae_lsb"], 255)

    def test_normalized_boxes_and_class_aware_nms(self):
        values = np.zeros((1, 84, 3))
        values[0, :4, :] = np.array([0.5, 0.5, 0.4, 0.4])[:, None]
        values[0, 4, :2] = [0.9, 0.8]
        values[0, 5, 2] = 0.7
        output = (np.rint(values / 0.01) - 128).astype(np.int8)
        transform = {"width": 320, "height": 320, "original_width": 640, "original_height": 640,
                     "left": 0, "top": 0, "scale_x": 0.5, "scale_y": 0.5}
        detections = decode_yolo(output, {"quantization": (0.01, -128)}, transform, 139)
        self.assertEqual(len(detections), 2)
        self.assertEqual([d["category_id"] for d in detections], [0, 1])
        np.testing.assert_allclose(detections[0]["bbox"], [192, 192, 256, 256])
        anchored = decode_yolo(output, {"quantization": (0.01, -128)}, transform, 139, include_anchor=True)
        self.assertEqual([d['anchor_index'] for d in anchored], [0, 2])

    def test_aligned_box_iou_and_hashed_baseline(self):
        output = np.zeros((1, 84, 2), np.int8)
        output[0, :4] = np.array([50, 50, 40, 40])[:, None]
        np.testing.assert_array_equal(aligned_iou(output, output, (0.01, 0)), [1, 1])
        moved = output.copy()
        moved[0, 0, 1] += 40
        np.testing.assert_array_equal(aligned_iou(moved, output, (0.01, 0)), [1, 0])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'outputs').mkdir()
            np.savez_compressed(folder / 'outputs/000000000001.npz', tflite_reference=output)
            rows = {(1, 'tflite_reference'): dict(input_sha256='same input',
                                                 output_sha256=hashlib.sha256(output.tobytes()).hexdigest())}
            np.testing.assert_array_equal(require_cached_output(folder, rows, 1, 'tflite_reference', 'same input'), output)
            with self.assertRaises(ValueError):
                require_cached_output(folder, rows, 1, 'tflite_reference', 'different input')
            rows[(1, 'tflite_reference')]['output_sha256'] = 'corrupt output'
            with self.assertRaises(ValueError):
                require_cached_output(folder, rows, 1, 'tflite_reference', 'same input')

    def test_coco_perfect_and_empty_predictions(self):
        manifest = {"classes": ["person"]}
        samples = [{"image_id": 1, "width": 100, "height": 100,
                    "objects": {"bbox": [[10, 20, 40, 60]], "category": [0], "area": [1200]}}]
        predictions = {"perfect": [{"image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40], "score": 0.9}],
                       "empty": []}
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_coco(manifest, samples, predictions, Path(directory), detailed=True)
        self.assertAlmostEqual(result["perfect"]["AP_50_95"], 1)
        self.assertEqual(result["empty"]["AP_50_95"], 0)
        self.assertAlmostEqual(result['perfect']['per_category']['person']['AP_50_95'], 1)


if __name__ == "__main__":
    unittest.main()
