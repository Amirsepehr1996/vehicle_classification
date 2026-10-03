from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageOps

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Image Processing Step
class ResizeWithPadding:
    """Resize while preserving proportions, then pad to a square."""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        resized = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            color=(124, 116, 104),
        )

        position = (
            (self.size - resized.width) // 2,
            (self.size - resized.height) // 2,
        )
        canvas.paste(resized, position)

        return canvas


def build_test_transform(config):
    """Use the same preprocessing as our previous evaluation."""

    return transforms.Compose([
        ResizeWithPadding(size=config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"],
            std=config["std"],
        ),
    ])


# Load the saved model
def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]

    if config["architecture"] != "resnet18":
        raise ValueError("This evaluator expects a ResNet18 model.")

    if config["loss"].lower() != "bce":
        raise ValueError("This evaluator expects the selected BCE model.")

    if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
        raise ValueError("Invalid class mapping in the checkpoint.")

    classes = sorted(class_to_idx, key=class_to_idx.get)

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=config["dropout"]),
        nn.Linear(model.fc.in_features, len(classes)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_test_transform(config)

    return model, transform, class_to_idx, classes, checkpoint



# Read mentor"s images and their class labels.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_ALIASES = {
    "neysan": "vanet",
    "nysan": "vanet",
    "nissan": "vanet",
}


def calculate_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_test_images(data_dir, class_to_idx):
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}"
        )

    rows = []

    for path in sorted(data_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        relative_path = path.relative_to(data_dir)

        if len(relative_path.parts) < 2:
            raise ValueError(
                f"Image must be inside a class folder: {path}"
            )

        original_label = relative_path.parts[0].strip().lower()
        label = LABEL_ALIASES.get(original_label, original_label)

        if label not in class_to_idx:
            raise ValueError(
                f"Unknown class folder: {original_label}. "
                f"Expected classes: {list(class_to_idx)}"
            )

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()

        except Exception as exc:
            raise ValueError(
                f"Unreadable image: {path}: {exc}"
            ) from exc

        rows.append({
            "path": str(path.resolve()),
            "original_label": original_label,
            "true_class": label,
            "sha256": calculate_sha256(path),
        })

    if not rows:
        raise ValueError("No supported images found in the dataset.")

    return pd.DataFrame(rows)


# check training overlap
def check_training_overlap(test_df, checkpoint, output_dir):
    training_hashes = checkpoint.get("training_sha256")

    if not training_hashes:
        raise ValueError(
            "Training hashes are missing from the checkpoint. "
            "Use the exported mentor checkpoint."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlap_mask = test_df["sha256"].isin(set(training_hashes))

    excluded_df = test_df.loc[overlap_mask].copy()
    evaluation_df = test_df.loc[~overlap_mask].copy()
    evaluation_df = evaluation_df.reset_index(drop=True)

    excluded_df.to_csv(
        output_dir / "excluded_training_overlap.csv",
        index=False,
    )

    summary = {
        "supplied_images": int(len(test_df)),
        "excluded_training_overlaps": int(len(excluded_df)),
        "evaluation_images": int(len(evaluation_df)),
        "overlap_check_method": "exact_file_sha256",
        "extra_duplicate_copies_in_evaluation": int(
            len(evaluation_df) - evaluation_df["sha256"].nunique()
        ),
    }

    with (output_dir / "overlap_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("Images supplied:", summary["supplied_images"])
    print("Training overlaps excluded:", summary["excluded_training_overlaps"])
    print("Images remaining for evaluation:", summary["evaluation_images"])

    if evaluation_df.empty:
        raise ValueError(
            "No images remain after excluding training overlaps."
        )

    return evaluation_df, summary


# Prepare the image loader
class MentorTestDataset(Dataset):
    def __init__(self, dataframe, transform, class_to_idx):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image_tensor = self.transform(image)

        label = self.class_to_idx[row["true_class"]]

        return image_tensor, label


def build_test_loader(
    evaluation_df, transform, class_to_idx, batch_size=16
):
    dataset = MentorTestDataset(
        dataframe=evaluation_df,
        transform=transform,
        class_to_idx=class_to_idx,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )



# Generate predictions and calculate metrics
def evaluate_model(model, loader, evaluation_df, classes, device):
    true_labels = []
    predicted_labels = []
    confidence_scores = []

    model.eval()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)

            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite scores.")

            predicted = logits.argmax(dim=1)

            # Sigmoid scores match the BCE training objective.
            scores = torch.sigmoid(logits)
            confidence = scores.gather(
                1, predicted.unsqueeze(1)
            ).squeeze(1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.cpu().tolist())
            confidence_scores.extend(confidence.cpu().tolist())

    if len(predicted_labels) != len(evaluation_df):
        raise ValueError("Prediction count does not match image count.")

    predictions_df = evaluation_df.copy()
    predictions_df["predicted_class"] = [
        classes[index] for index in predicted_labels
    ]
    predictions_df["confidence_percent"] = [
        score * 100 for score in confidence_scores
    ]
    predictions_df["correct"] = [
        true == predicted
        for true, predicted in zip(true_labels, predicted_labels)
    ]

    class_ids = list(range(len(classes)))

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=class_ids,
            average="macro",
            zero_division=0,
        )
    )

    present_classes = set(predictions_df["true_class"])
    missing_classes = [
        name for name in classes if name not in present_classes
    ]

    metrics = {
        "evaluated_images": int(len(predictions_df)),
        "correct": int(predictions_df["correct"].sum()),
        "errors": int((~predictions_df["correct"]).sum()),
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "macro_classes": classes,
        "missing_classes": missing_classes,
        "confidence_method": "uncalibrated_sigmoid_score",
    }

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_ids,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_ids,
    )

    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(
        f"Correct: {metrics['correct']}/"
        f"{metrics['evaluated_images']}"
    )

    if missing_classes:
        print("Classes absent from evaluation:", missing_classes)
        print("Macro metrics include all eight model classes.")

    return predictions_df, metrics, report, matrix



#Save the evaluation results
def save_results(
    predictions_df,
    metrics,
    report,
    matrix,
    classes,
    overlap_summary,
    checkpoint_path,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record the supplied and excluded image counts.
    final_metrics = {
        **metrics,
        **overlap_summary,
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_sha256": calculate_sha256(checkpoint_path),
    }

    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    predictions_df.loc[~predictions_df["correct"]].to_csv(
        output_dir / "errors.csv",
        index=False,
    )

    pd.DataFrame(report).T.to_csv(
        output_dir / "classification_report.csv",
        index_label="class_or_metric",
    )

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    ).to_csv(
        output_dir / "confusion_matrix.csv",
        index_label="true_class",
    )

    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(final_metrics, file, indent=2)

    print("Results saved:", output_dir.resolve())



# Connect the functions into a runnable script
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the vehicle classifier on labeled images."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the exported model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder directly containing the class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mentor_results"),
        help="An empty folder for evaluation results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            parser.error("--output-dir must be a directory.")
        if any(args.output_dir.iterdir()):
            parser.error(
                "Output folder is not empty. Choose a new folder."
            )

    device = torch.device("cpu")
    print("Device:", device)

    model, transform, class_to_idx, classes, checkpoint = load_model(
        args.checkpoint, device
    )

    test_df = read_test_images(args.data_dir, class_to_idx)

    evaluation_df, overlap_summary = check_training_overlap(
        test_df, checkpoint, args.output_dir
    )

    loader = build_test_loader(
        evaluation_df,
        transform,
        class_to_idx,
        batch_size=args.batch_size,
    )

    predictions_df, metrics, report, matrix = evaluate_model(
        model, loader, evaluation_df, classes, device
    )

    save_results(
        predictions_df=predictions_df,
        metrics=metrics,
        report=report,
        matrix=matrix,
        classes=classes,
        overlap_summary=overlap_summary,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()