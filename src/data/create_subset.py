import os
import shutil
import random

from src.config import RAW_DATA_DIR, SUBSET_DATA_DIR, SUBSET_VALIDATION_DIR


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


def create_validation_subset(src_root, subset_root, dst_root, images_per_class=50):
    """
    Creates a validation set using only files from src_root
    that are NOT present in subset_root.
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    categories = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    for category in categories:
        src_cat_path = os.path.join(src_root, category)
        subset_cat_path = os.path.join(subset_root, category)
        dst_cat_path = os.path.join(dst_root, category)

        if not os.path.exists(dst_cat_path):
            os.makedirs(dst_cat_path)

        all_raw_images = set(f for f in os.listdir(src_cat_path) if f.lower().endswith('.tif'))

        used_images = set()
        if os.path.exists(subset_cat_path):
            used_images = set(os.listdir(subset_cat_path))

        available_images = list(all_raw_images - used_images)

        num_to_copy = min(len(available_images), images_per_class)
        selected_images = random.sample(available_images, num_to_copy)

        for img_name in selected_images:
            shutil.copy2(
                os.path.join(src_cat_path, img_name),
                os.path.join(dst_cat_path, img_name)
            )
    print(f"Validation subset created at: {dst_root}")


if __name__ == "__main__":
    # create_balanced_subset(str(RAW_DATA_DIR), str(SUBSET_DATA_DIR))
    create_validation_subset(str(RAW_DATA_DIR), str(SUBSET_DATA_DIR), str(SUBSET_VALIDATION_DIR))
    print("\nBalanced subset created successfully.")
