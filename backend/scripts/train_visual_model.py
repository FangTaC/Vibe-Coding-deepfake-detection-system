from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a torchvision visual backbone for Deepfake face classification.")
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet_b0",
        choices=("efficientnet_b0", "resnet50", "mobilenet_v3_large"),
        help="Torchvision backbone architecture.",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with train/ and val/ subfolders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for weights and metadata.")
    parser.add_argument("--dataset-version", type=str, required=True, help="Frozen dataset version string for metadata.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    return parser.parse_args()


def model_name_for_arch(arch: str) -> str:
    return f"{arch}_deepfake"


def build_model(arch: str):
    from torch import nn  # type: ignore
    from torchvision import models  # type: ignore

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        return model
    raise ValueError(f"unsupported architecture: {arch}")


def build_dataloaders(data_root: Path, batch_size: int):
    from torch.utils.data import DataLoader  # type: ignore
    from torchvision import datasets, transforms  # type: ignore

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_set = datasets.ImageFolder(data_root / "train", transform=transform)
    val_set = datasets.ImageFolder(data_root / "val", transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def resolve_device(requested_device: str) -> str:
    import torch  # type: ignore

    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_auc(labels: list[int], scores: list[float]) -> float:
    from sklearn.metrics import roc_auc_score  # type: ignore

    if len(set(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def dataset_class_distribution(data_root: Path, split: str) -> dict[str, int]:
    distribution: dict[str, int] = {}
    split_root = data_root / split
    for label_name in ("real", "fake"):
        class_dir = split_root / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"expected class directory: {class_dir}")
        distribution[label_name] = sum(
            1 for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return distribution


def calibrate_thresholds(
    scores: list[float],
    labels: list[int],
    default_true_threshold: float,
    default_fake_threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not scores or not labels or len(scores) != len(labels):
        return (
            {"true_threshold": default_true_threshold, "fake_threshold": default_fake_threshold},
            {"best_accuracy": 0.0, "review_rate": 0.0},
        )

    best_state: tuple[float, float, float, float] | None = None
    for true_step in range(5, 51):
        true_threshold = round(true_step / 100.0, 2)
        for fake_step in range(max(true_step + 5, 55), 96):
            fake_threshold = round(fake_step / 100.0, 2)
            predicted = []
            review_count = 0
            for score, truth in zip(scores, labels):
                if score <= true_threshold:
                    predicted.append(0)
                elif score >= fake_threshold:
                    predicted.append(1)
                else:
                    predicted.append(1 - truth)
                    review_count += 1

            accuracy = sum(int(pred == truth) for pred, truth in zip(predicted, labels)) / max(len(labels), 1)
            review_rate = review_count / max(len(labels), 1)
            state = (accuracy, -review_rate, true_threshold, fake_threshold)
            if best_state is None or state > best_state:
                best_state = state

    if best_state is None:
        return (
            {"true_threshold": default_true_threshold, "fake_threshold": default_fake_threshold},
            {"best_accuracy": 0.0, "review_rate": 0.0},
        )
    accuracy, negative_review_rate, true_threshold, fake_threshold = best_state
    return (
        {"true_threshold": true_threshold, "fake_threshold": fake_threshold},
        {"best_accuracy": round(accuracy, 4), "review_rate": round(-negative_review_rate, 4)},
    )


def main() -> None:
    args = parse_args()
    if not (args.data_root / "train").exists() or not (args.data_root / "val").exists():
        raise FileNotFoundError("expected data_root/train and data_root/val with real/ and fake/ classes")

    import torch  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score  # type: ignore

    actual_device = resolve_device(args.device)
    train_loader, val_loader = build_dataloaders(args.data_root, args.batch_size)
    model = build_model(args.arch)
    model.to(actual_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_scores: list[float] = []
    best_labels: list[int] = []
    model_output_dir = args.output_dir / "visual" / args.arch
    model_output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = model_output_dir / "model.pt"

    print(
        f"[train] arch={args.arch} device={actual_device} epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} train_batches={len(train_loader)} val_batches={len(val_loader)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_examples = 0
        for images, labels in train_loader:
            images = images.to(actual_device)
            labels = labels.to(actual_device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.size(0))
            train_loss += float(loss.item()) * batch_size
            train_examples += batch_size

        model.eval()
        val_scores: list[float] = []
        val_labels: list[int] = []
        val_predictions: list[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(actual_device)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                batch_scores = probabilities[:, 1].tolist()
                batch_predictions = probabilities.argmax(axis=1).tolist()
                val_scores.extend(float(score) for score in batch_scores)
                val_predictions.extend(int(prediction) for prediction in batch_predictions)
                val_labels.extend(int(value) for value in labels.cpu().numpy().tolist())

        train_loss_mean = train_loss / max(train_examples, 1)
        val_acc = float(accuracy_score(val_labels, val_predictions)) if val_labels else 0.0
        val_f1_epoch = (
            float(f1_score(val_labels, val_predictions, zero_division=0)) if val_labels else 0.0
        )
        val_auc_epoch = safe_auc(val_labels, val_scores) if val_labels else 0.0
        if val_acc >= best_acc:
            best_acc = val_acc
            best_scores = list(val_scores)
            best_labels = list(val_labels)
            torch.save(model.state_dict(), weight_path)
            best_tag = "yes"
        else:
            best_tag = "no"

        elapsed = time.time() - epoch_start
        print(
            f"[epoch {epoch + 1}/{args.epochs}] loss={train_loss_mean:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1_epoch:.4f} val_auc={val_auc_epoch:.4f} "
            f"best={best_tag} elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    val_f1 = float(f1_score(best_labels, [1 if score >= 0.5 else 0 for score in best_scores], zero_division=0)) if best_labels else 0.0
    val_auc = safe_auc(best_labels, best_scores) if best_labels else 0.0
    thresholds, calibration = calibrate_thresholds(
        best_scores,
        best_labels,
        default_true_threshold=0.30,
        default_fake_threshold=0.70,
    )

    metadata = {
        "model_name": model_name_for_arch(args.arch),
        "architecture": args.arch,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": args.dataset_version,
        "input_size": [256, 256],
        "class_distribution": {
            "train": dataset_class_distribution(args.data_root, "train"),
            "val": dataset_class_distribution(args.data_root, "val"),
        },
        "metrics": {
            "split": "val",
            "accuracy": round(best_acc, 4),
            "f1": round(val_f1, 4),
            "auc": round(val_auc, 4),
        },
        "thresholds": thresholds,
        "threshold_source": "validated_grid_search" if best_labels else "default_config_no_validation",
        "calibration": calibration,
        "training_config": {
            "arch": args.arch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": actual_device,
        },
    }
    metadata_path = model_output_dir / "meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.arch == "efficientnet_b0":
        legacy_weight_path = args.output_dir / "efficientnet_b0_deepfake.pt"
        legacy_metadata_path = args.output_dir / "efficientnet_b0_deepfake.meta.json"
        legacy_weight_path.write_bytes(weight_path.read_bytes())
        legacy_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] Saved legacy EfficientNet weights to {legacy_weight_path}", flush=True)
        print(f"[done] Saved legacy EfficientNet metadata to {legacy_metadata_path}", flush=True)
    print(f"[done] Saved weights to {weight_path}", flush=True)
    print(f"[done] Saved metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
