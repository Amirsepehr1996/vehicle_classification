import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIDENCE_THRESHOLD = 0.8


class ResizeWithPadding:
    def __init__(self, size, color):
        self.size = size
        self.color = tuple(color)

    def __call__(self, image):
        resized = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.color)
        canvas.paste(
            resized,
            (
                (self.size - resized.width) // 2,
                (self.size - resized.height) // 2,
            ),
        )
        return canvas


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]

    if config["architecture"] != "resnet18":
        raise ValueError("Expected a ResNet18 checkpoint.")

    if config["loss"].lower() != "bce":
        raise ValueError("Expected the selected BCE model.")

    if checkpoint.get("stage") != "neysan_train_validation_refit":
        raise ValueError(
            "Use resnet18_neysan_trainval_final.pt "
            "from the Neysan experiment."
        )

    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError("Invalid class mapping.")

    if "vanet" not in class_to_idx or "neysan" in class_to_idx:
        raise ValueError(
            "The model must predict Vanet, with no separate Neysan class."
        )

    classes = sorted(class_to_idx, key=class_to_idx.get)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=config["dropout"]),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    transform = transforms.Compose([
        ResizeWithPadding(config["image_size"], config["padding_color"]),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])

    return model, transform, class_to_idx, classes, checkpoint


def calculate_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_neysan_images(data_dir):
    """The supplied folder must contain only Neysan test images."""
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {data_dir}")

    rows = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()

        except Exception as exc:
            raise ValueError(f"Unreadable image: {path}: {exc}") from exc

        rows.append({
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(data_dir)),
            "source_label": "neysan",
            "expected_class": "vanet",
            "sha256": calculate_sha256(path),
        })

    if not rows:
        raise ValueError("No supported images found.")

    return pd.DataFrame(rows)


def exclude_training_overlap(test_df, checkpoint, output_dir):
    training_hashes = checkpoint.get("training_sha256")

    if not training_hashes:
        raise ValueError("Training hashes are missing from the checkpoint.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlap_mask = test_df["sha256"].isin(set(training_hashes))

    excluded_df = test_df.loc[overlap_mask].copy()
    evaluation_df = test_df.loc[~overlap_mask].copy().reset_index(drop=True)

    excluded_df.to_csv(output_dir / "excluded_training_overlap.csv", index=False)

    summary = {
        "supplied_images": int(len(test_df)),
        "excluded_training_overlaps": int(len(excluded_df)),
        "evaluation_images": int(len(evaluation_df)),
        "extra_duplicate_copies_in_evaluation": int(
            len(evaluation_df) - evaluation_df["sha256"].nunique()
        ),
        "overlap_check_method": "exact_file_sha256",
    }

    with (output_dir / "overlap_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Images supplied:", summary["supplied_images"])
    print("Training overlaps excluded:", summary["excluded_training_overlaps"])
    print("Images remaining:", summary["evaluation_images"])

    if evaluation_df.empty:
        raise ValueError("No images remain after excluding training overlaps.")

    return evaluation_df, summary


class NeysanTestDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image)


def predict_images(model, transform, evaluation_df, classes, device, batch_size=16):
    loader = DataLoader(
        NeysanTestDataset(evaluation_df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    predicted_indices = []
    confidence_scores = []
    all_scores = []

    with torch.inference_mode():
        for images in loader:
            logits = model(images.to(device))

            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced non-finite scores.")

            predictions = logits.argmax(dim=1)
            scores = torch.sigmoid(logits)
            confidence = scores.gather(1, predictions.unsqueeze(1)).squeeze(1)

            predicted_indices.extend(predictions.cpu().tolist())
            confidence_scores.extend(confidence.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())

    if len(predicted_indices) != len(evaluation_df):
        raise RuntimeError("Prediction count does not match image count.")

    results = evaluation_df.copy().reset_index(drop=True)

    results["predicted_class"] = [classes[i] for i in predicted_indices]
    results["confidence"] = confidence_scores
    results["confidence_percent"] = results["confidence"] * 100
    results["correct"] = results["predicted_class"] == results["expected_class"]
    results["needs_human_check"] = results["confidence"] < CONFIDENCE_THRESHOLD

    for index, name in enumerate(classes):
        results[f"sigmoid_score_{name}"] = [s[index] for s in all_scores]

    return results


def save_evaluation(
    results,
    class_to_idx,
    classes,
    overlap_summary,
    checkpoint_path,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_labels = [class_to_idx[name] for name in results["expected_class"]]
    predicted_labels = [class_to_idx[name] for name in results["predicted_class"]]
    all_labels = list(range(len(classes)))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=[class_to_idx["vanet"]],
        average=None,
        zero_division=0,
    )

    review_mask = results["confidence"] < CONFIDENCE_THRESHOLD
    correct_mask = results["correct"]

    metrics = {
        **overlap_summary,
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_sha256": calculate_sha256(checkpoint_path),
        "correct": int(correct_mask.sum()),
        "errors": int((~correct_mask).sum()),
        "neysan_as_vanet_accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "vanet_recall_on_neysan": float(recall[0]),
        "vanet_precision_on_neysan_only": float(precision[0]),
        "vanet_f1_on_neysan_only": float(f1[0]),
        "confidence_method": "uncalibrated_sigmoid_score",
        "human_check_threshold": CONFIDENCE_THRESHOLD,
        "below_threshold_images": int(review_mask.sum()),
        "errors_below_threshold": int((review_mask & ~correct_mask).sum()),
        "errors_at_or_above_threshold": int((~review_mask & ~correct_mask).sum()),
        "evaluation_scope": (
            "Neysan-only test with expected class vanet. "
            "All eligible images included regardless of confidence."
        ),
    }

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=all_labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

    results.to_csv(output_dir / "predictions.csv", index=False)
    results.loc[~correct_mask].to_csv(output_dir / "errors.csv", index=False)
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        output_dir / "confusion_matrix.csv",
        index_label="expected_class",
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Neysan predicted as Vanet:", f"{metrics['neysan_as_vanet_accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['evaluation_images']}")
    print("Confidence below 80%:", metrics["below_threshold_images"])
    print("Evaluation results saved:", output_dir.resolve())

    return metrics


def copy_images_for_human_check(results, output_dir):
    output_dir = Path(output_dir)

    if not (output_dir / "metrics.json").is_file():
        raise RuntimeError("Save evaluation results before copying review images.")

    review_dir = output_dir / "need human check"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_df = results.loc[results["confidence"] < CONFIDENCE_THRESHOLD].copy()

    copied_paths = []

    for number, (_, row) in enumerate(review_df.iterrows(), start=1):
        source_path = Path(row["path"])
        destination = review_dir / (
            f"{number:05d}_{row['sha256']}{source_path.suffix.lower()}"
        )
        shutil.copy2(source_path, destination)
        copied_paths.append(str(destination.resolve()))

    review_df["review_copy_path"] = copied_paths
    review_df.to_csv(output_dir / "human_check_manifest.csv", index=False)

    print("Images copied for human check:", len(review_df))
    print("Review folder:", review_dir.resolve())
    print("Original images preserved.")
    print("Evaluation scores include low-confidence images.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test the Neysan-holdout ResNet18 model. "
            "The input folder must contain only Neysan images; "
            "their expected class is vanet."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to resnet18_neysan_trainval_final.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing only Neysan images, including subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neysan_mentor_results"),
        help="An empty output folder outside the input dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir == data_dir or data_dir in output_dir.parents:
        parser.error("Output folder must be outside the input dataset.")

    if output_dir.exists():
        if not output_dir.is_dir():
            parser.error("--output-dir must be a directory.")
        if any(output_dir.iterdir()):
            parser.error("Output folder is not empty. Choose a new folder.")

    device = torch.device("cpu")
    print("Device:", device)

    model, transform, class_to_idx, classes, checkpoint = load_model(
        args.checkpoint, device
    )

    test_df = read_neysan_images(data_dir)
    evaluation_df, overlap_summary = exclude_training_overlap(
        test_df, checkpoint, output_dir
    )

    results = predict_images(
        model=model,
        transform=transform,
        evaluation_df=evaluation_df,
        classes=classes,
        device=device,
        batch_size=args.batch_size,
    )

    save_evaluation(
        results=results,
        class_to_idx=class_to_idx,
        classes=classes,
        overlap_summary=overlap_summary,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
    )

    copy_images_for_human_check(results, output_dir)

    print("Confidence percentages are uncalibrated sigmoid scores.")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()