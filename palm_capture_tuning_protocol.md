# Palm Capture Tuning Protocol

Use this protocol to compare palm-vein capture settings while keeping the physical setup stable.

## Goal

Find the best balance between:

- no overexposure or clipping
- visible structure in the center of the palm
- repeatable illumination across multiple captures

## Keep Constant During One Sweep

- same camera position
- same hand-to-camera distance
- same IR position and brightness
- same palm pose and angle
- same output directory for one exposure condition

Only change one variable at a time.

## Recommended First Sweep

Keep:

- `gain = 1.0`
- IR moderately far, not close to the palm
- no preview autostretch when judging final capture quality

Test this exposure sequence:

1. `1000`
2. `1100`
3. `1200`
4. `1300`

Suggested commands:

```bash
python3 capture_on_hand_detect.py --fps 30 --exposure-us 1000 --gain 1.0 --out-dir captures/captures_e1000 --burst-frames 5 --preprocess
python3 capture_on_hand_detect.py --fps 30 --exposure-us 1100 --gain 1.0 --out-dir captures/captures_e1100 --burst-frames 5 --preprocess
python3 capture_on_hand_detect.py --fps 30 --exposure-us 1200 --gain 1.0 --out-dir captures/captures_e1200 --burst-frames 5 --preprocess
python3 capture_on_hand_detect.py --fps 30 --exposure-us 1300 --gain 1.0 --out-dir captures/captures_e1300 --burst-frames 5 --preprocess
```

For each setting, collect at least `5` captures.

After `1200us` is confirmed as the best baseline, use the automatic quality gate for larger collection:

```bash
python3 capture_on_hand_detect.py --fps 30 --exposure-us 1200 --gain 1.0 --out-dir captures/captures_e1200_filtered --burst-frames 5 --preprocess --quality-filter
```

For debugging rejected samples:

```bash
python3 capture_on_hand_detect.py --fps 30 --exposure-us 1200 --gain 1.0 --out-dir captures/captures_e1200_filtered_debug --burst-frames 5 --preprocess --quality-filter --save-rejected
```

## What To Look For

Best images should have:

- dark background
- palm not clipped to white
- central palm structure visible
- finger and palm edges still separated from the background
- similar appearance across repeated captures

Watch for failure modes:

- palm looks flat and bright: IR too strong or too close
- palm very dark and featureless: exposure too low or IR too weak
- only surface creases appear, no deeper structure: illumination geometry still needs work
- hand fills the frame vertically: move the hand slightly farther from the camera

## Metadata To Compare

Each capture JSON now records:

- `brightness.mean_gray`
- `brightness.std_gray`
- `brightness.p95_gray`
- `brightness.p99_gray`
- `brightness.saturated_fraction`
- `preprocessing.profile`
- `preprocessing.quality.roi.p95`
- `preprocessing.quality.roi.p99`
- `preprocessing.quality.roi.saturated_fraction`
- `quality_filter.usable`
- `quality_filter.score`
- `quality_filter.reasons`
- `camera_settings`

Use these as sanity checks:

- lower `saturated_fraction` is better than clipped images
- very low `std_gray` can indicate the palm is too flat and low-contrast
- for the current setup, prefer the visually stronger `1200us` result unless a nearby exposure gives clearer veins without bright patches
- compare images visually first, then use metadata to confirm the trend

## Recommended Decision Order

1. Reject any setup that clips or washes out the palm.
2. Prefer the setup that shows the strongest central palm structure while staying stable across repeats.
3. Only after exposure is stable, tune IR distance, diffuser, or side-lighting geometry.

## After the First Sweep

If `1000` to `1300` all look safe but still weak for vein contrast:

- keep the best exposure from the sweep
- keep `gain = 1.0`
- improve IR geometry with diffuser or side illumination
- repeat a smaller sweep around the current best exposure

Use the default `legacy` preprocessing for the old model and as the baseline retraining dataset. Keep `capture_v2` as an experiment only if it is visibly better on the same physical capture setup. Use `_vessel_preview.png` only for visual inspection, not as a training input.
