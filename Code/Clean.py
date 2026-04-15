import os
import cv2
import shutil

def clean_invalid_images(root_dir):
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not os.path.isfile(img_path):
                    continue
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[WARN] Immagine non leggibile, rimuovo: {img_path}")
                    os.remove(img_path)

if __name__ == "__main__":
    root_dir = '/Users/lorenzo/Desktop/Università/Sapienza/3° anno/AI Lab Computer Vision and NLP/Progetto/Classificatore rifiuti/Dataset/Data'
    clean_invalid_images(root_dir)