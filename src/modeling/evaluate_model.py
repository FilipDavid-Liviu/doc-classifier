import os
import time
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.config import MODEL_DIR, SUBSET_VALIDATION_DIR


def main():
    data_dir = str(SUBSET_VALIDATION_DIR)
    model_dir = str(MODEL_DIR)
    max_samples_per_class = 50

    print(f"Loading model from {model_dir}...")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0.0

    print(f"Loading data from {data_dir}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    total_inference_time = 0.0
    total_samples = 0

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    print(f"Evaluating on {device}. Maximum {max_samples_per_class} samples per class...")

    for category in classes:
        cat_path = os.path.join(data_dir, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith('.tif')]
        images = images[:max_samples_per_class]

        for img_name in images:
            img_path = os.path.join(cat_path, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                start_time = time.time()
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_class_id = outputs.logits.argmax(-1).item()
                end_time = time.time()

                total_inference_time += (end_time - start_time)
                total_samples += 1

                predicted_label = model.config.id2label[predicted_class_id]

                y_true.append(category)
                y_pred.append(predicted_label)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    latency_ms = (total_inference_time / total_samples) * 1000 if total_samples > 0 else 0.0

    eval_output = {
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "latency_ms": latency_ms,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct
    }

    output_path = Path(model_dir) / "eval_script_results.json"
    with open(output_path, "w") as f:
        json.dump(eval_output, f, indent=4)

    print(f"Evaluation Results saved to {output_path}")
    print(f"Overall Accuracy: {eval_output['classification_report']['accuracy']:.4f}")
    print(f"Latency: {latency_ms:.2f} ms/doc")


if __name__ == "__main__":
    main()
