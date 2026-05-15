# Code to properly split the dataset (for it to have train and val)
# Run with the code "python backend/split_dataset.py"

import os
import random
import shutil

# paths (IMPORTANT: ensure matche repo)
image_dir = "backend/dataset/images"
label_dir = "backend/dataset/labels"

train_img_dir = "backend/dataset/images/train"
val_img_dir = "backend/dataset/images/val"
train_lbl_dir = "backend/dataset/labels/train"
val_lbl_dir = "backend/dataset/labels/val"

# create folders
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# get images
images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

random.shuffle(images)

split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

def move_files(img_list, img_out, lbl_out):
    for img in img_list:
        label = img.replace(".jpg", ".txt")

        shutil.move(os.path.join(image_dir, img),
                    os.path.join(img_out, img))

        shutil.move(os.path.join(label_dir, label),
                    os.path.join(lbl_out, label))

move_files(train_images, train_img_dir, train_lbl_dir)
move_files(val_images, val_img_dir, val_lbl_dir)

print("DONE: dataset split into train/val")