import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import MNIST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MNIST and preprocess into train/val/test tensor files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/mnist"),
        help="Directory for raw MNIST download/cache (default: data/mnist).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/mnist"),
        help="Directory to write processed dataset tensors (default: datasets/mnist).",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Validation set size sampled from MNIST train split (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42).",
    )
    return parser.parse_args()


def save_split(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    payload = {"images": images.contiguous(), "labels": labels.contiguous()}
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MNIST(root=str(raw_dir), train=True, download=True)
    test_ds = MNIST(root=str(raw_dir), train=False, download=True)

    if args.val_size <= 0 or args.val_size >= len(train_ds):
        raise ValueError(
            f"--val-size must be in [1, {len(train_ds) - 1}], got {args.val_size}."
        )

    # Convert uint8 [0, 255] -> float32 [0.0, 1.0], add channel dim (N, 1, 28, 28)
    train_images = train_ds.data.unsqueeze(1).float() / 255.0
    train_labels = train_ds.targets.long()

    test_images = test_ds.data.unsqueeze(1).float() / 255.0
    test_labels = test_ds.targets.long()

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(train_images.shape[0], generator=generator)
    val_idx = perm[: args.val_size]
    train_idx = perm[args.val_size :]

    final_train_images = train_images[train_idx]
    final_train_labels = train_labels[train_idx]
    val_images = train_images[val_idx]
    val_labels = train_labels[val_idx]

    train_path = out_dir / "train.pt"
    val_path = out_dir / "val.pt"
    test_path = out_dir / "test.pt"
    meta_path = out_dir / "metadata.json"

    save_split(final_train_images, final_train_labels, train_path)
    save_split(val_images, val_labels, val_path)
    save_split(test_images, test_labels, test_path)

    metadata = {
        "raw_dir": str(raw_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "seed": args.seed,
        "val_size": args.val_size,
        "shapes": {
            "train_images": list(final_train_images.shape),
            "train_labels": list(final_train_labels.shape),
            "val_images": list(val_images.shape),
            "val_labels": list(val_labels.shape),
            "test_images": list(test_images.shape),
            "test_labels": list(test_labels.shape),
        },
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("MNIST preprocessing complete.")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"  Meta:  {meta_path}")


if __name__ == "__main__":
    main()
