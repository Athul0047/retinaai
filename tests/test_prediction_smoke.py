"""Offline smoke test for local RetinaAI inference using a synthetic image."""

import unittest

from PIL import Image

from src.prediction import NUM_CLASSES, RetinaPredictor


class PredictionSmokeTest(unittest.TestCase):
    def test_synthetic_rgb_image_returns_five_class_probabilities(self) -> None:
        predictor = RetinaPredictor(device="cpu")
        image = Image.new("RGB", (320, 200), color=(80, 120, 160))

        result = predictor.predict(image)

        self.assertIn(result["grade_index"], range(NUM_CLASSES))
        self.assertEqual(result["class_name"], predictor.config["class_names"][result["grade_index"]])
        self.assertEqual(len(result["probabilities"]), NUM_CLASSES)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
