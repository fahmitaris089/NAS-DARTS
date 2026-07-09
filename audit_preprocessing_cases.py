"""
Audit preprocessing/crop/illumination for hard classification cases.

This is a non-training diagnostic tool. It reads error rows from
analyze_prediction_overlap.py, compares each error image against its true class
and confusing top-k classes, and writes visual/contact-sheet plus image-quality
statistics to an output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from palm_vein_dataset import build_label_map, load_split


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit image preprocessing/crop/illumination for C12 error cases."
    )
    parser.add_argument("--data_dir", default="preprocessed_results")
    parser.add_argument("--split_path", default="split_info.json")
    parser.add_argument(
        "--errors_csv",
        default="analysis/prediction_overlap_L005_C12_cells10_HintonKD_9976/c12_errors.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/preprocessing_audit_L005_C12_cells10_9976",
    )
    parser.add_argument(
        "--focus_filenames",
        default="",
        help="Optional comma-separated filenames, e.g. 277_6.bmp,504_4.bmp. Empty means all error rows.",
    )
    parser.add_argument("--top_confusions", type=int, default=3)
    parser.add_argument("--thumb_size", type=int, default=160)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_text: str, data_dir: Path) -> Path:
    normalized = path_text.replace("\\", "/")
    p = Path(normalized)
    if p.exists():
        return p
    if normalized.startswith(str(data_dir).replace("\\", "/")):
        candidate = Path(normalized)
    else:
        candidate = data_dir / p.name if p.parent.name == data_dir.name else Path(normalized)
    if candidate.exists():
        return candidate
    # Common case: CSV has preprocessed_results\277\277_6.bmp.
    parts = normalized.split("/")
    if len(parts) >= 3:
        candidate = data_dir / parts[-2] / parts[-1]
        if candidate.exists():
            return candidate
    return p


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def otsu_threshold(img: np.ndarray) -> float:
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))
    total = img.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    weight_b = 0
    max_var = -1.0
    threshold = 0
    for t in range(256):
        weight_b += int(hist[t])
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += float(t * hist[t])
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return float(threshold)


def image_stats(path: Path) -> dict[str, float]:
    img = load_gray(path)
    h, w = img.shape
    thr = otsu_threshold(img)
    # Palm foreground is bright relative to black background.
    mask = img > max(8.0, thr)
    if mask.sum() < img.size * 0.05:
        mask = img > 8.0

    if mask.any():
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        fg = img[mask]
        margin_left = x1 / max(1, w)
        margin_right = (w - 1 - x2) / max(1, w)
        margin_top = y1 / max(1, h)
        margin_bottom = (h - 1 - y2) / max(1, h)
        bbox_w = (x2 - x1 + 1) / max(1, w)
        bbox_h = (y2 - y1 + 1) / max(1, h)
        center_x = ((x1 + x2) / 2.0) / max(1, w)
        center_y = ((y1 + y2) / 2.0) / max(1, h)
    else:
        fg = img.ravel()
        margin_left = margin_right = margin_top = margin_bottom = math.nan
        bbox_w = bbox_h = center_x = center_y = math.nan

    gy, gx = np.gradient(img / 255.0)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_fg = grad[mask] if mask.any() else grad.ravel()
    p1, p99 = np.percentile(img, [1, 99])

    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "p01": float(p1),
        "p99": float(p99),
        "foreground_mean": float(fg.mean()),
        "foreground_std": float(fg.std()),
        "foreground_ratio": float(mask.mean()),
        "otsu_threshold": thr,
        "high_pixel_ratio": float((img > 245).mean()),
        "low_pixel_ratio": float((img < 10).mean()),
        "bbox_w": float(bbox_w),
        "bbox_h": float(bbox_h),
        "bbox_area": float(bbox_w * bbox_h),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "margin_left": float(margin_left),
        "margin_right": float(margin_right),
        "margin_top": float(margin_top),
        "margin_bottom": float(margin_bottom),
        "gradient_mean": float(grad_fg.mean()),
        "gradient_std": float(grad_fg.std()),
    }


def summarize_class(paths: list[Path]) -> dict[str, dict[str, float]]:
    stats = [image_stats(p) for p in paths]
    keys = stats[0].keys() if stats else []
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.asarray([s[key] for s in stats], dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    return summary


def zscore(value: float, mean: float, std: float) -> float:
    if not np.isfinite(value) or not np.isfinite(mean) or not np.isfinite(std) or std < 1e-9:
        return 0.0
    return float((value - mean) / std)


def parse_top_labels(row: dict[str, str], max_items: int) -> list[int]:
    labels = []
    for item in row.get("c12_top5_labels", "").split("|"):
        item = item.strip()
        if not item:
            continue
        try:
            labels.append(int(item))
        except ValueError:
            pass
    return labels[:max_items]


def class_image_paths(data_dir: Path, subject_id: str) -> list[Path]:
    return sorted((data_dir / subject_id).glob("*.bmp"), key=lambda p: p.name)


def make_contact_sheet(
    output_path: Path,
    rows: list[tuple[str, list[Path]]],
    focus_path: Path,
    thumb_size: int,
) -> None:
    pad = 18
    label_h = 28
    cols = max((len(paths) for _, paths in rows), default=1)
    cols = min(cols, 10)
    width = cols * (thumb_size + pad) + pad
    height = len(rows) * (thumb_size + label_h + pad) + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except Exception:
        font = font_bold = None

    for row_idx, (label, paths) in enumerate(rows):
        y = pad + row_idx * (thumb_size + label_h + pad)
        draw.text((pad, y), label, fill=(0, 0, 0), font=font_bold)
        for col_idx, path in enumerate(paths[:cols]):
            x = pad + col_idx * (thumb_size + pad)
            yy = y + label_h
            img = Image.open(path).convert("RGB").resize((thumb_size, thumb_size))
            canvas.paste(img, (x, yy))
            is_focus = path.resolve() == focus_path.resolve()
            outline = (220, 0, 0) if is_focus else (150, 150, 150)
            draw.rectangle(
                [x, yy, x + thumb_size - 1, yy + thumb_size - 1],
                outline=outline,
                width=4 if is_focus else 1,
            )
            draw.rectangle([x, yy + thumb_size - 22, x + thumb_size, yy + thumb_size], fill=(255, 255, 255))
            draw.text(
                (x + 4, yy + thumb_size - 19),
                path.name,
                fill=(220, 0, 0) if is_focus else (0, 0, 0),
                font=font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def metric_diagnosis(focus: dict[str, float], true_summary: dict[str, dict[str, float]]) -> list[str]:
    checks = {
        "foreground_mean": "illumination",
        "foreground_std": "contrast",
        "high_pixel_ratio": "overexposure",
        "bbox_area": "crop_scale",
        "center_x": "horizontal_crop_shift",
        "center_y": "vertical_crop_shift",
        "margin_top": "top_crop_margin",
        "gradient_mean": "vein_edge_strength",
    }
    tags = []
    for metric, tag in checks.items():
        ref = true_summary.get(metric)
        if not ref:
            continue
        z = zscore(focus[metric], ref["mean"], ref["std"])
        if abs(z) >= 1.5:
            direction = "high" if z > 0 else "low"
            tags.append(f"{tag}_{direction}_z{z:.2f}")
    return tags


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_path = Path(args.split_path)
    errors_csv = Path(args.errors_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    inv_label_map = {label: subject for subject, label in label_map.items()}

    rows = read_csv(errors_csv)
    focus_names = {x.strip() for x in args.focus_filenames.split(",") if x.strip()}
    if focus_names:
        rows = [row for row in rows if row.get("filename") in focus_names]
    if not rows:
        raise SystemExit("No error rows selected for audit.")

    stats_rows: list[dict[str, Any]] = []
    summary_cases: list[dict[str, Any]] = []
    md_lines = [
        "# Preprocessing / Illumination Audit",
        "",
        f"- Errors CSV: `{errors_csv}`",
        f"- Data dir: `{data_dir}`",
        "",
    ]

    for row in rows:
        filename = row["filename"]
        true_subject = row["subject_id"]
        focus_path = resolve_path(row["path"], data_dir)
        top_labels = parse_top_labels(row, args.top_confusions)
        top_subjects = [inv_label_map[label] for label in top_labels if label in inv_label_map]
        class_subjects = []
        for subject in [true_subject, *top_subjects]:
            if subject not in class_subjects:
                class_subjects.append(subject)

        focus_stats = image_stats(focus_path)
        true_paths = class_image_paths(data_dir, true_subject)
        true_summary = summarize_class(true_paths)
        tags = metric_diagnosis(focus_stats, true_summary)

        contact_rows = []
        for subject in class_subjects:
            role = "true" if subject == true_subject else "confusion"
            contact_rows.append((f"{role} class {subject}", class_image_paths(data_dir, subject)))
        sheet_name = f"contact_{Path(filename).stem}_true_{true_subject}_vs_" + "_".join(class_subjects[1:]) + ".png"
        make_contact_sheet(output_dir / sheet_name, contact_rows, focus_path, args.thumb_size)

        for metric, value in focus_stats.items():
            ref = true_summary.get(metric, {"mean": math.nan, "std": math.nan})
            stats_rows.append(
                {
                    "filename": filename,
                    "subject_id": true_subject,
                    "metric": metric,
                    "focus_value": round(value, 6),
                    "true_class_mean": round(ref["mean"], 6),
                    "true_class_std": round(ref["std"], 6),
                    "z_vs_true_class": round(zscore(value, ref["mean"], ref["std"]), 6),
                }
            )

        case_summary = {
            "filename": filename,
            "true_subject": true_subject,
            "c12_pred_label": row.get("c12_pred", row.get("c12_top1")),
            "c12_top5_labels": row.get("c12_top5_labels", ""),
            "compared_subjects": class_subjects,
            "diagnostic_tags": tags,
            "contact_sheet": sheet_name,
            "focus_stats": focus_stats,
        }
        summary_cases.append(case_summary)

        md_lines.extend(
            [
                f"## {filename}",
                "",
                f"- True subject: `{true_subject}`",
                f"- C12 top-5 labels: `{row.get('c12_top5_labels', '')}`",
                f"- Compared subjects: `{', '.join(class_subjects)}`",
                f"- Contact sheet: `{sheet_name}`",
                f"- Diagnostic tags: `{', '.join(tags) if tags else 'no strong image-stat outlier vs true class'}`",
                "",
                "| Metric | Focus | True class mean | z-score |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in [
            "foreground_mean",
            "foreground_std",
            "high_pixel_ratio",
            "foreground_ratio",
            "bbox_area",
            "center_x",
            "center_y",
            "margin_top",
            "gradient_mean",
        ]:
            ref = true_summary[metric]
            md_lines.append(
                f"| {metric} | {focus_stats[metric]:.4f} | {ref['mean']:.4f} | "
                f"{zscore(focus_stats[metric], ref['mean'], ref['std']):.2f} |"
            )
        md_lines.append("")

    write_csv(output_dir / "image_quality_stats.csv", stats_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "errors_csv": str(errors_csv),
                "data_dir": str(data_dir),
                "num_cases": len(summary_cases),
                "cases": summary_cases,
            },
            f,
            indent=2,
            default=str,
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Audit cases : {len(summary_cases)}")
    print(f"Output      : {output_dir}")
    print(f"Summary     : {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
