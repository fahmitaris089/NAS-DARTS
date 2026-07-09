"""Diagnose deterministic illumination preprocessing for hard NAS/KD errors.

This script is diagnostic only: it does not train, tune, or save model weights.
It evaluates the full test split under several deterministic image
normalization methods and reports whether the remaining hard samples are fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from genotypes import dict_to_genotype
from model_eval import EvalNetwork
from nas_config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from palm_vein_dataset import build_image_list, build_label_map, load_split


FOCUS_FILENAMES = {"277_6.bmp", "504_4.bmp"}


class GrayscaleToRGB:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat(3, 1, 1) if tensor.shape[0] == 1 else tensor


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        image = self.transform(image)
        return image, int(label), idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic illumination preprocessing for a NAS checkpoint."
    )
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--student_weights", required=True)
    parser.add_argument("--data_dir", default=str(ROOT / "preprocessed_results"))
    parser.add_argument("--split_path", default=str(ROOT / "split_info.json"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)
    parser.add_argument(
        "--methods",
        default="none,autocontrast,equalize,gamma_0.8,gamma_0.9,gamma_1.1,gamma_1.2,clahe_1.5,clahe_2.0,clahe_3.0",
        help="Comma-separated preprocessing methods to evaluate.",
    )
    parser.add_argument(
        "--focus_filenames",
        default="277_6.bmp,504_4.bmp",
        help="Comma-separated filenames highlighted in error_focus.csv and contact sheets.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")


def parse_reduction_indices(raw_value) -> list[int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [int(x) for x in raw_value]
    if isinstance(raw_value, str):
        return [int(x.strip()) for x in raw_value.split(",") if x.strip()]
    raise TypeError(f"Unsupported reduction_indices type: {type(raw_value)}")


def build_model(config_path: Path, weights_path: Path, num_classes: int, device: torch.device) -> EvalNetwork:
    cfg = load_json(config_path)
    genotype = dict_to_genotype(cfg["genotype"])
    retrain_cfg = cfg.get("retrain_cfg", {})
    dropout = float(retrain_cfg.get("dropout", cfg.get("dropout", 0.3)))

    model = EvalNetwork(
        genotype=genotype,
        C_init=int(cfg["C_init"]),
        num_cells=int(cfg["num_cells"]),
        num_classes=num_classes,
        auxiliary=False,
        dropout=dropout,
        stem_downsample=int(cfg.get("stem_downsample", 2)),
        reduction_indices=parse_reduction_indices(cfg.get("reduction_indices")),
    )
    state_dict = unwrap_state_dict(torch.load(weights_path, map_location="cpu"))
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_auxiliary_head")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    model.to(device)
    model.eval()
    return model


def load_test_samples(data_dir: Path, split_path: Path):
    split = load_split(split_path)
    label_map = build_label_map(split["subjects"])
    test_samples = build_image_list(data_dir, split["test"], label_map)
    test_samples = [(Path(path), int(label)) for path, label in test_samples]
    return test_samples, len(label_map)


def sample_metadata(samples: list[tuple[Path, int]]) -> list[dict[str, Any]]:
    rows = []
    for idx, (path, label) in enumerate(samples):
        rows.append(
            {
                "index": idx,
                "subject_id": path.parent.name,
                "filename": path.name,
                "path": str(path),
                "label": int(label),
            }
        )
    return rows


def gamma_transform(img: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
    return Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)), mode="L")


def try_clahe(img: Image.Image, clip_limit: float) -> Image.Image:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("OpenCV is not available; CLAHE methods are skipped") from exc

    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    out = clahe.apply(arr)
    return Image.fromarray(out, mode="L")


def build_preprocess(method: str) -> Callable[[Image.Image], Image.Image]:
    method = method.strip()
    if method == "none":
        return lambda img: img
    if method == "autocontrast":
        return lambda img: ImageOps.autocontrast(img.convert("L"))
    if method == "equalize":
        return lambda img: ImageOps.equalize(img.convert("L"))
    if method.startswith("gamma_"):
        gamma = float(method.replace("gamma_", ""))
        return lambda img, g=gamma: gamma_transform(img, g)
    if method.startswith("clahe_"):
        clip = float(method.replace("clahe_", ""))
        return lambda img, c=clip: try_clahe(img, c)
    raise ValueError(f"Unsupported illumination preprocessing method: {method}")


def build_transform(method: str, input_size: int):
    preprocess = build_preprocess(method)
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(preprocess),
            transforms.ToTensor(),
            GrayscaleToRGB(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@torch.no_grad()
def evaluate_method(model: EvalNetwork, samples, labels_ref: torch.Tensor, method: str, args, device):
    dataset = ImagePathDataset(samples, build_transform(method, args.input_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    all_logits = []
    all_indices = []
    model.eval()
    for images, _, indices in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        all_logits.append(logits.detach().cpu())
        all_indices.append(indices.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    indices = torch.cat(all_indices, dim=0)
    order = torch.argsort(indices)
    logits = logits[order]
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    correct_mask = pred.eq(labels_ref)
    correct = int(correct_mask.sum().item())
    total = int(labels_ref.numel())
    top_probs, top_labels = probs.topk(5, dim=1)
    sorted_labels = probs.argsort(dim=1, descending=True)
    true_ranks = []
    true_probs = []
    for row_idx, label in enumerate(labels_ref):
        label_int = int(label.item())
        rank_pos = (sorted_labels[row_idx] == label_int).nonzero(as_tuple=False)
        true_ranks.append(int(rank_pos[0].item()) + 1 if rank_pos.numel() else None)
        true_probs.append(float(probs[row_idx, label_int].item()))
    return {
        "method": method,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "pred": pred,
        "probs": probs,
        "top_labels": top_labels,
        "top_probs": top_probs,
        "true_ranks": true_ranks,
        "true_probs": true_probs,
        "correct_mask": correct_mask,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_processed_contact_sheet(output_path: Path, samples, metadata, focus_names: set[str], methods: list[str], input_size: int):
    focus_rows = [idx for idx, meta in enumerate(metadata) if meta["filename"] in focus_names]
    if not focus_rows:
        return
    thumb = 150
    pad = 16
    label_h = 26
    width = len(methods) * (thumb + pad) + pad
    height = len(focus_rows) * (thumb + label_h + pad) + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except Exception:
        font = font_bold = None

    for row_no, sample_idx in enumerate(focus_rows):
        path, _ = samples[sample_idx]
        y = pad + row_no * (thumb + label_h + pad)
        draw.text((pad, y), f"{path.name}", fill=(0, 0, 0), font=font_bold)
        original = Image.open(path).convert("L").resize((input_size, input_size))
        for col_no, method in enumerate(methods):
            x = pad + col_no * (thumb + pad)
            yy = y + label_h
            processed = build_preprocess(method)(original).resize((thumb, thumb))
            canvas.paste(processed.convert("RGB"), (x, yy))
            draw.rectangle([x, yy, x + thumb - 1, yy + thumb - 1], outline=(150, 150, 150), width=1)
            draw.rectangle([x, yy + thumb - 22, x + thumb, yy + thumb], fill=(255, 255, 255))
            draw.text((x + 4, yy + thumb - 19), method, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    split_path = Path(args.split_path)
    samples, num_classes = load_test_samples(data_dir, split_path)
    metadata = sample_metadata(samples)
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    focus_names = {x.strip() for x in args.focus_filenames.split(",") if x.strip()} or FOCUS_FILENAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test samples: {len(samples)}")
    model = build_model(Path(args.student_config), Path(args.student_weights), num_classes, device)

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = []
    skipped = []
    for method in requested_methods:
        try:
            print(f"Evaluating {method}...")
            result = evaluate_method(model, samples, labels, method, args, device)
            results.append(result)
        except RuntimeError as exc:
            if method.startswith("clahe_"):
                print(f"[warn] Skipping {method}: {exc}")
                skipped.append({"method": method, "reason": str(exc)})
                continue
            raise

    if not results:
        raise SystemExit("No preprocessing methods were evaluated.")

    baseline = next((r for r in results if r["method"] == "none"), results[0])
    result_summary = [
        {
            "method": r["method"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "delta_vs_none_pp": (r["accuracy"] - baseline["accuracy"]) * 100.0,
        }
        for r in results
    ]
    best = max(result_summary, key=lambda x: (x["correct"], x["method"] == "none"))

    prediction_rows = []
    focus_rows = []
    for idx, meta in enumerate(metadata):
        row: dict[str, Any] = {**meta, "true_label": int(labels[idx].item())}
        baseline_correct = bool(baseline["correct_mask"][idx].item())
        row["baseline_correct"] = baseline_correct
        for r in results:
            method = r["method"]
            pred = int(r["pred"][idx].item())
            conf = float(r["probs"][idx, pred].item())
            true_prob = float(r["true_probs"][idx])
            true_rank = r["true_ranks"][idx]
            top5_labels = "|".join(str(int(x)) for x in r["top_labels"][idx].tolist())
            top5_probs = "|".join(f"{float(x):.8f}" for x in r["top_probs"][idx].tolist())
            row[f"{method}_pred"] = pred
            row[f"{method}_conf"] = conf
            row[f"{method}_correct"] = bool(r["correct_mask"][idx].item())
            row[f"{method}_true_rank"] = true_rank
            row[f"{method}_true_prob"] = true_prob
            row[f"{method}_top5_labels"] = top5_labels
            row[f"{method}_top5_probs"] = top5_probs
        prediction_rows.append(row)
        if meta["filename"] in focus_names or not baseline_correct:
            focus_rows.append(row.copy())

    write_csv(output_dir / "predictions_illumination.csv", prediction_rows)
    write_csv(output_dir / "error_focus.csv", focus_rows)

    contact_methods = [r["method"] for r in results[:10]]
    make_processed_contact_sheet(
        output_dir / "focus_preprocess_contact_sheet.png",
        samples,
        metadata,
        focus_names,
        contact_methods,
        args.input_size,
    )

    summary = {
        "student_config": args.student_config,
        "student_weights": args.student_weights,
        "num_test_samples": len(samples),
        "results": result_summary,
        "best_method": best["method"],
        "best_accuracy": best["accuracy"],
        "best_correct": best["correct"],
        "skipped": skipped,
        "focus_filenames": sorted(focus_names),
        "focus_rows": focus_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    md = [
        "# Illumination Preprocessing Diagnostic",
        "",
        f"- Student weights: `{args.student_weights}`",
        f"- Test samples: `{len(samples)}`",
        f"- Best method: `{best['method']}` = {best['accuracy'] * 100:.2f}% ({best['correct']}/{best['total']})",
        "",
        "## Results",
        "",
        "| Method | Accuracy | Correct | Delta vs none |",
        "|---|---:|---:|---:|",
    ]
    for item in result_summary:
        md.append(
            f"| {item['method']} | {item['accuracy'] * 100:.2f}% | "
            f"{item['correct']}/{item['total']} | {item['delta_vs_none_pp']:+.2f} pp |"
        )
    if skipped:
        md.extend(["", "## Skipped", ""])
        for item in skipped:
            md.append(f"- `{item['method']}`: {item['reason']}")
    md.extend(["", "## Focus Errors", ""])
    for row in focus_rows:
        if row["filename"] not in focus_names:
            continue
        md.append(f"### {row['filename']}")
        md.append("")
        for item in result_summary:
            method = item["method"]
            md.append(
                f"- `{method}`: pred={row[f'{method}_pred']} "
                f"correct={row[f'{method}_correct']} true_rank={row[f'{method}_true_rank']} "
                f"true_prob={float(row[f'{method}_true_prob']):.6f}"
            )
        md.append("")
    md.append("Contact sheet: `focus_preprocess_contact_sheet.png`")
    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\nAccuracy:")
    for item in result_summary:
        print(f"  {item['method']:<14}: {item['accuracy'] * 100:.2f}% ({item['correct']}/{item['total']})")
    print(f"\nBest: {best['method']} {best['accuracy'] * 100:.2f}%")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
