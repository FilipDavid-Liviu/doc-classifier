import os
import evaluate
import numpy as np

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
    trainer.train()

    print(f"Saving model to {MODEL_DIR}")
    model.save_pretrained(str(MODEL_DIR))
    processor.save_pretrained(str(MODEL_DIR))

if __name__ == "__main__":
    main()
