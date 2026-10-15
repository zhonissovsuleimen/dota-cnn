from pathlib import Path
import sys

from PIL import Image, ImageOps
import torch
from torchvision import transforms

from train import HeroCNN

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best.pt"
TOP_K = 5


def main():
    image_path = Path(sys.argv[1] if len(sys.argv) > 1 else input("Image path: ").strip())
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    image_size = checkpoint.get("image_size", 256)

    model = HeroCNN(len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
        values, indices = probabilities.topk(min(TOP_K, len(class_to_idx)))

    print(f"\nImage: {image_path}")
    print(f"Prediction: {idx_to_class[indices[0].item()]}")
    print("Top predictions:")
    for rank, (probability, index) in enumerate(zip(values, indices), 1):
        print(f"  {rank}. {idx_to_class[index.item()]}: {probability.item():.2%}")


if __name__ == "__main__":
    main()

