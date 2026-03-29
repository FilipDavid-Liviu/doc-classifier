import os

from PIL import Image
from datasets import Dataset, Features, Image as ImageFeature, ClassLabel

def load_local_data(data_path, test_size=0.2, seed=42):
    categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_label = ClassLabel(names=categories)

    data_list = []
    for category in categories:
        cat_path = os.path.join(data_path, category)
        for img_file in os.listdir(cat_path):
            if img_file.lower().endswith('.tif'):
                full_path = os.path.join(cat_path, img_file)

                try:
                    with Image.open(full_path) as img:
                        img.verify()
                    data_list.append({
                        "image": full_path,
                        "label": class_label.str2int(category)
                    })
                except Exception:
                    print(f"Skipping corrupt image: {full_path}")

    features = Features({"image": ImageFeature(), "label": class_label})
    full_dataset = Dataset.from_list(data_list, features=features)

    split_dataset = full_dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    
    return split_dataset, class_label
