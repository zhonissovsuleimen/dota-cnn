from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 256
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
OUTPUT = ROOT / "checkpoints" / "best.pt"


class HeroCNN(nn.Sequential):
    """A CNN that accepts (batch, 3, 256, 256) and returns hero logits.

    Inheriting nn.Sequential preserves the layer keys of existing checkpoints.
    """

    def __init__(self, num_classes):
        super().__init__(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(1.0, 10.0),
        ),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
        ),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(ROOT / "train", transform=train_transform)
    eval_data = datasets.ImageFolder(ROOT / "eval", transform=eval_transform)

    if train_data.class_to_idx != eval_data.class_to_idx:
        raise ValueError("train/ and eval/ must contain the same hero folders")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE)

    model = HeroCNN(len(train_data.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    def run_epoch(loader, training):
        model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = loss_fn(logits, labels)

                if training:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")
    print(f"Heroes: {train_data.class_to_idx}")

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(train_loader, True)
        eval_loss, eval_acc = run_epoch(eval_loader, False)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train: {train_acc:.1%} (loss {train_loss:.4f}) | "
            f"Eval: {eval_acc:.1%} (loss {eval_loss:.4f}) | "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,
        )

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_data.class_to_idx,
                "num_classes": len(train_data.classes),
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "eval_accuracy": eval_acc,
                "eval_loss": eval_loss,
            }, OUTPUT)
            print(f"  Saved best model -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
