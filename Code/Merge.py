import os
import shutil

def copy_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        if os.path.isfile(src_path):
            if not os.path.exists(dst_path):  
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} → {dst_path}")

def safe_merge_class(src_dir, dst_dir, prefix):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i, filename in enumerate(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, filename)
        if not os.path.isfile(src_path):
            continue

        ext = os.path.splitext(filename)[1]
        new_filename = f"{prefix}_{i:04d}{ext}"
        dst_path = os.path.join(dst_dir, new_filename)

        counter = 0
        while os.path.exists(dst_path):
            new_filename = f"{prefix}_{i:04d}_{counter}{ext}"
            dst_path = os.path.join(dst_dir, new_filename)
            counter += 1

        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} → {dst_path}")

if __name__ == "__main__":
    src_dir = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/trashnet-master/data/dataset-resized'
    dst_dir = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/Data'
    copy_files(src_dir, dst_dir)
    kaggle_base = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/garbage_classification'
    final_base = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/Data'
    for glass_type in ["brown-glass", "green-glass", "white-glass"]:
        safe_merge_class(
            os.path.join(kaggle_base, glass_type),
            os.path.join(final_base, "glass"),
            prefix=glass_type.replace("-", "")
        )
    for glass_type in ["paper", "cardboard"]:
        safe_merge_class(
            os.path.join(kaggle_base, glass_type),
            os.path.join(final_base, "paper"),
            prefix=glass_type.replace("-", "")
        )
    for category in ["battery", "biological", "clothes", "metal", "plastic"]:
        safe_merge_class(
            os.path.join(kaggle_base, category),
            os.path.join(final_base, category),
            prefix=category
        )
