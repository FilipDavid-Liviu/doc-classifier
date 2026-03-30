# doc-classifier

This repository contains code for fine-tuning an image classification model for document classification. The RVL-CDIP dataset is utilized.

Outputs and saved models are located in `models/`.

## 1. Model and adaptation

The selected backbone is `microsoft/dit-base`. 

Training hyperparameters:
* Epochs: 6
* Learning rate: 5e-5
* Optimizer: adafactor
* Batch size: 4 (per device)
* Gradient accumulation steps: 4
* Mixed precision: fp16

| Run directory        | Backbone | Finetune Strategy | Trainable params | Total params | Trainable % |
|----------------------| -------- | ----------------- | ---------------- | ------------ | ----------- |
| `models/final_model` | `microsoft/dit-base` | Full fine-tune | 85820944  | 85820944  | 100.0  |

## 2. Test-set accuracy and inference latency

Evaluation is performed using `evaluate_model.py`, calculating overall accuracy and generating a classification report. 

### Overall Metrics
| Run | Accuracy | Macro F1 | Weighted F1 | Latency (ms / doc) |
| --- | -------- | -------- | ----------- | ------------------ |
| `models/` | 0.816 | 0.816 | 0.816 | 17.29 |

### Category Performance
| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| advertisement | 0.857 | 0.840 | 0.848 | 50.0 |
| budget | 0.775 | 0.620 | 0.689 | 50.0 |
| email | 0.909 | 1.000 | 0.952 | 50.0 |
| file_folder | 0.852 | 0.920 | 0.885 | 50.0 |
| form | 0.611 | 0.660 | 0.635 | 50.0 |
| handwritten | 0.852 | 0.920 | 0.885 | 50.0 |
| invoice | 0.786 | 0.880 | 0.830 | 50.0 |
| letter | 0.833 | 0.900 | 0.865 | 50.0 |
| memo | 0.900 | 0.900 | 0.900 | 50.0 |
| news_article | 0.787 | 0.740 | 0.763 | 50.0 |
| presentation | 0.796 | 0.860 | 0.827 | 50.0 |
| questionnaire | 0.818 | 0.720 | 0.766 | 50.0 |
| resume | 0.930 | 0.800 | 0.860 | 50.0 |
| scientific_publication | 0.905 | 0.760 | 0.826 | 50.0 |
| scientific_report | 0.639 | 0.780 | 0.703 | 50.0 |
| specification | 0.884 | 0.760 | 0.817 | 50.0 |

## 3. Training time and hardware

| Run | Train time (s) | Train time (approx.) | Train throughput (samples/s) |
| --- | -------------- | -------------------- | ---------------------------- |
| `models/` | 667.5 | ~11.1 mins | 32.36 |

Consumer hardware -- 12th Gen Intel(R) Core(TM) i5-12500H CPU, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB Dedicated), 16.0 GB RAM.

## 4. Dataset Processing

Data preparation scripts handle `.tif` images.
* `create_subset.py`: Creates balanced training subsets and separate validation subsets.
* `preprocess.py`: Loads local data, verifies images, and applies a stratified train/test split.

## 5. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt