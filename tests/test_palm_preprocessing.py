import unittest
import sys
import types

import cv2
import numpy as np

from palm_preprocessing import (
    PROFILE_CAPTURE_V2,
    PROFILE_DATASET_V3,
    PalmPreprocessingConfig,
    assess_palm_vein_quality,
    apply_clahe,
    extract_adaptive_roi,
    extract_roi,
    normalize_and_resize,
    preprocess_palm_image,
)


def make_synthetic_palm() -> np.ndarray:
    image = np.zeros((720, 1280), dtype=np.uint8)
    cv2.ellipse(image, (640, 395), (310, 250), 0, 0, 360, 88, -1)
    cv2.rectangle(image, (470, 25), (540, 210), 82, -1)
    cv2.rectangle(image, (600, 0), (675, 210), 86, -1)
    cv2.rectangle(image, (725, 35), (795, 220), 84, -1)
    cv2.polylines(
        image,
        [np.array([(520, 440), (610, 395), (710, 360), (815, 315)], np.int32)],
        False,
        54,
        10,
        cv2.LINE_AA,
    )
    cv2.polylines(
        image,
        [np.array([(590, 570), (645, 500), (705, 440), (785, 390)], np.int32)],
        False,
        58,
        8,
        cv2.LINE_AA,
    )
    gradient = np.linspace(0, 12, image.shape[1], dtype=np.uint8)
    gradient = np.tile(gradient[np.newaxis, :], (image.shape[0], 1))
    return cv2.add(image, gradient)


def make_usable_final() -> np.ndarray:
    image = np.full((224, 224), 160, dtype=np.uint8)
    cv2.line(image, (35, 170), (190, 70), 75, 5, cv2.LINE_AA)
    cv2.line(image, (55, 130), (170, 120), 90, 4, cv2.LINE_AA)
    cv2.line(image, (90, 200), (140, 45), 95, 4, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (223, 18), 220, -1)
    noise = np.random.default_rng(7).normal(0, 8, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class PalmPreprocessingTest(unittest.TestCase):
    def test_legacy_profile_matches_original_pipeline(self) -> None:
        image = make_synthetic_palm()
        config = PalmPreprocessingConfig()

        result = preprocess_palm_image(image, config)
        expected_roi, _ = extract_roi(image, config.roi_size, config.centroid_window)
        expected_clahe = apply_clahe(
            expected_roi,
            config.clahe_clip,
            config.clahe_tile,
        )
        expected_final = normalize_and_resize(expected_clahe, config.final_size)

        np.testing.assert_array_equal(result["roi"], expected_roi)
        np.testing.assert_array_equal(result["clahe"], expected_clahe)
        np.testing.assert_array_equal(result["final"], expected_final)
        self.assertEqual(result["debug"]["profile"], "legacy")
        self.assertEqual(result["debug"]["center_mode"], "weighted")

    def test_capture_v2_outputs_expected_debug_images_and_quality(self) -> None:
        image = make_synthetic_palm()
        config = PalmPreprocessingConfig(
            profile=PROFILE_CAPTURE_V2,
            clahe_clip=1.2,
            clahe_tile=(12, 12),
            denoise_h=5.0,
        )

        result = preprocess_palm_image(image, config)
        debug = result["debug"]

        self.assertEqual(result["roi"].shape, (384, 384))
        self.assertEqual(result["mask"].shape, image.shape)
        self.assertEqual(result["clahe"].shape, (384, 384))
        self.assertEqual(result["vessel_preview"].shape, (384, 384))
        self.assertEqual(result["final"].shape, (224, 224))
        self.assertEqual(debug["profile"], PROFILE_CAPTURE_V2)
        self.assertEqual(debug["center_mode"], "contour")
        self.assertEqual(debug["refined_center"], debug["rough_center"])
        self.assertIn("quality", debug)
        self.assertIn("roi", debug["quality"])
        self.assertIn("p95", debug["quality"]["roi"])
        self.assertIn("p99", debug["quality"]["roi"])
        self.assertIn("dark_fraction", debug["quality"]["roi"])
        self.assertIn("saturated_fraction", debug["quality"]["roi"])
        self.assertIn("sharpness", debug["quality"]["roi"])

    def test_dataset_v3_uses_shifted_large_roi_and_percentile_stretch(self) -> None:
        image = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.ellipse(image, (960, 560), (430, 360), 0, 0, 360, 95, -1)
        cv2.rectangle(image, (770, 40), (860, 330), 88, -1)
        cv2.rectangle(image, (930, 0), (1020, 330), 92, -1)
        cv2.rectangle(image, (1100, 40), (1190, 340), 90, -1)
        cv2.ellipse(image, (1370, 650), (240, 115), -25, 0, 360, 90, -1)
        cv2.line(image, (780, 720), (1280, 420), 52, 12, cv2.LINE_AA)
        cv2.line(image, (860, 850), (1220, 520), 58, 10, cv2.LINE_AA)

        config = PalmPreprocessingConfig(
            profile=PROFILE_DATASET_V3,
            clahe_clip=2.4,
            adaptive_roi=True,
            adaptive_roi_scale=0.95,
            palm_core_width_ratio=0.45,
        )

        result = preprocess_palm_image(image, config)
        debug = result["debug"]
        expected_roi, expected_debug = extract_adaptive_roi(
            image,
            roi_scale=config.adaptive_roi_scale,
            width_ratio=config.palm_core_width_ratio,
            centroid_window=config.centroid_window,
            center_offset=(config.center_offset_x, config.center_offset_y),
        )

        np.testing.assert_array_equal(result["roi"], expected_roi)
        self.assertEqual(result["final"].shape, (224, 224))
        self.assertEqual(result["clahe"].shape, result["roi"].shape)
        self.assertEqual(debug["profile"], PROFILE_DATASET_V3)
        self.assertEqual(debug["center_mode"], "adaptive_weighted")
        self.assertTrue(debug["adaptive_roi"])
        self.assertEqual(debug["adaptive_roi_scale"], config.adaptive_roi_scale)
        self.assertEqual(debug["palm_core_width_ratio"], config.palm_core_width_ratio)
        self.assertEqual(debug["roi_box"], expected_debug["roi_box"])
        self.assertIn("final_source", debug["quality"])

    def test_quality_filter_rejects_edge_heavy_low_texture_final(self) -> None:
        final = np.full((224, 224), 168, dtype=np.uint8)
        cv2.rectangle(final, (0, 0), (223, 28), 240, -1)
        cv2.rectangle(final, (0, 196), (223, 223), 90, -1)
        cv2.line(final, (18, 112), (206, 112), 118, 11, cv2.LINE_AA)
        cv2.line(final, (112, 18), (112, 206), 118, 11, cv2.LINE_AA)
        final = cv2.GaussianBlur(final, (0, 0), 5.5)

        result = assess_palm_vein_quality(final)

        self.assertFalse(result["usable"])
        self.assertIn("edges dominate fine vessel texture", result["reasons"])

    def test_invalid_profile_is_rejected(self) -> None:
        image = make_synthetic_palm()
        config = PalmPreprocessingConfig(profile="unknown")

        with self.assertRaises(ValueError):
            preprocess_palm_image(image, config)

    def test_quality_filter_rejects_dark_flat_final(self) -> None:
        final = np.full((224, 224), 85, dtype=np.uint8)
        result = assess_palm_vein_quality(final)

        self.assertFalse(result["usable"])
        self.assertIn("final too dark", result["reasons"])
        self.assertIn("mostly dark final image", result["reasons"])

    def test_quality_filter_accepts_structured_final(self) -> None:
        final = make_usable_final()
        result = assess_palm_vein_quality(final)

        self.assertTrue(result["usable"], result["reasons"])
        self.assertGreaterEqual(result["metrics"]["mean"], 110)
        self.assertGreaterEqual(result["metrics"]["p95"], 125)
        self.assertGreaterEqual(result["metrics"]["gradient_p95"], 40)

    def test_quality_filter_cli_is_opt_in(self) -> None:
        sys.modules["picamera2"] = types.SimpleNamespace(Picamera2=object)
        import capture_on_hand_detect

        args = capture_on_hand_detect.parse_args(["--preprocess"])

        self.assertTrue(args.preprocess)
        self.assertFalse(args.quality_filter)
        self.assertFalse(args.save_rejected)

    def test_dataset_v3_cli_defaults_are_applied(self) -> None:
        sys.modules["picamera2"] = types.SimpleNamespace(Picamera2=object)
        import capture_on_hand_detect

        argv = ["--preprocess", "--preprocess-profile", PROFILE_DATASET_V3]
        args = capture_on_hand_detect.parse_args(argv)
        explicit_options = capture_on_hand_detect.collect_explicit_options(
            argv,
            {
                "--preprocess-roi-size",
                "--preprocess-clahe-clip",
                "--preprocess-clahe-tile",
                "--preprocess-denoise-h",
                "--preprocess-center-offset-x",
                "--preprocess-center-offset-y",
                "--preprocess-stretch-percentiles",
                "--preprocess-adaptive-roi",
                "--preprocess-adaptive-roi-scale",
                "--preprocess-palm-core-width-ratio",
            },
        )
        config = capture_on_hand_detect.build_preprocessing_config(
            args,
            explicit_options,
        )

        self.assertEqual(config.profile, PROFILE_DATASET_V3)
        self.assertEqual(config.roi_size, 760)
        self.assertEqual(config.final_size, 224)
        self.assertEqual(config.clahe_clip, 2.4)
        self.assertEqual(config.clahe_tile, (8, 8))
        self.assertEqual(config.center_offset_x, 0)
        self.assertEqual(config.center_offset_y, 0)
        self.assertIsNone(config.stretch_percentiles)
        self.assertTrue(config.adaptive_roi)
        self.assertEqual(config.adaptive_roi_scale, 0.95)
        self.assertEqual(config.palm_core_width_ratio, 0.45)


if __name__ == "__main__":
    unittest.main()
