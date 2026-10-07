"""Check output comparison and the dataset evaluation adapter independently."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark import decode_yolo, difference, evaluate_coco


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

    def test_coco_perfect_and_empty_predictions(self):
        manifest = {"classes": ["person"]}
        samples = [{"image_id": 1, "width": 100, "height": 100,
                    "objects": {"bbox": [[10, 20, 40, 60]], "category": [0], "area": [1200]}}]
        predictions = {"perfect": [{"image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40], "score": 0.9}],
                       "empty": []}
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_coco(manifest, samples, predictions, Path(directory))
        self.assertAlmostEqual(result["perfect"]["AP_50_95"], 1)
        self.assertEqual(result["empty"]["AP_50_95"], 0)


if __name__ == "__main__":
    unittest.main()
