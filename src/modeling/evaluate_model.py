import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, accuracy_score

from src.config import SUBSET_DATA_DIR, MODEL_DIR


def main():
    data_dir = str(SUBSET_DATA_DIR)
    model_dir = str(MODEL_DIR)
    max_samples_per_class = 50

    print(f"Loading model from {model_dir}...")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    print(f"Evaluating on {device}. Maximum {max_samples_per_class} samples per class...")

    for category in classes:
        cat_path = os.path.join(data_dir, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        images = images[:max_samples_per_class]

        for img_name in images:
            img_path = os.path.join(cat_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_class_id = outputs.logits.argmax(-1).item()

                predicted_label = model.config.id2label[predicted_class_id]

                y_true.append(category)
                y_pred.append(predicted_label)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    print("Evaluation Results:")
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))


if __name__ == "__main__":
    main()
