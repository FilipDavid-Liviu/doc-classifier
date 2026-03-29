import os
import shutil
import random

from src.config import RAW_DATA_DIR, SUBSET_DATA_DIR


def create_balanced_subset(src_root, dst_root, images_per_class=500):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    categories = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    for category in categories:
        src_cat_path = os.path.join(src_root, category)
        dst_cat_path = os.path.join(dst_root, category)

        if not os.path.exists(dst_cat_path):
            os.makedirs(dst_cat_path)

        all_images = [f for f in os.listdir(src_cat_path)
                      if f.lower().endswith('.tif')]

        num_to_copy = min(len(all_images), images_per_class)
        selected_images = random.sample(all_images, num_to_copy)

        print(f"Copying {num_to_copy} images for category: {category}")

        for img_name in selected_images:
            shutil.copy2(
                os.path.join(src_cat_path, img_name),
                os.path.join(dst_cat_path, img_name)
            )


if __name__ == "__main__":
    create_balanced_subset(str(RAW_DATA_DIR), str(SUBSET_DATA_DIR))
    print("\nBalanced subset created successfully.")
