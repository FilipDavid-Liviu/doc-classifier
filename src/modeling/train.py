import os
import time

import evaluate
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import torch

from src.config import SUBSET_DATA_DIR, MODEL_DIR, CHECKPOINT_DIR
from src.data import load_local_data

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    print("Loading data...")
    dataset, label_info = load_local_data(str(SUBSET_DATA_DIR), test_size=0.1)
    
    id2label = {i: name for i, name in enumerate(label_info.names)}
    label2id = {name: i for i, name in enumerate(label_info.names)}

    model_name = "microsoft/dit-base"
    print(f"Loading processor: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)

    def transforms(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    dataset["train"].set_transform(transforms)
    dataset["test"].set_transform(transforms)

    print("Initializing model...")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=label_info.num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    print("Setting up training...")
    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adafactor",
        num_train_epochs=3,
        warmup_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Starting training...")
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)

    print("Running evaluation and extracting custom metrics...")

    start_time = time.time()
    predict_results = trainer.predict(dataset["test"])
    end_time = time.time()

    num_samples = len(dataset["test"])
    latency_ms = ((end_time - start_time) / num_samples) * 1000

    preds = np.argmax(predict_results.predictions, axis=1)
    labels = predict_results.label_ids
    target_names = [id2label[i] for i in range(len(id2label))]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0.0

    eval_output = {
        "classification_report": classification_report(
            labels,
            preds,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "latency_ms": latency_ms,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct
    }

    with open(Path(MODEL_DIR) / "eval_results.json", "w") as f:
        json.dump(eval_output, f, indent=4)

    history = trainer.state.log_history
    with open(Path(MODEL_DIR) / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    train_loss = [x["loss"] for x in history if "loss" in x]
    epochs = [x["epoch"] for x in history if "loss" in x]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.savefig(Path(MODEL_DIR) / "training_curves.png")

    print(f"Saving model to {MODEL_DIR}")
    model.save_pretrained(str(MODEL_DIR))
    processor.save_pretrained(str(MODEL_DIR))

    trainer.save_state()
    torch.save(args, Path(MODEL_DIR) / "training_args.bin")

if __name__ == "__main__":
    main()
