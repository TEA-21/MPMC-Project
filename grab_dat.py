import os
import random
import shutil

# 1. Your Kaggle dataset source
source_dir = r"D:\Downloads\archive\vehicle_data"

# 2. Your project destination
dest_dir = r"D:\Projects\MPMC"

target_folders = ['bicycle', 'motorcycle', 'scooter', 'car', 'van', 'bus', 'truck']
os.makedirs(dest_dir, exist_ok=True)
images_per_class = 45 

print("Extracting and auto-labeling images...")
counter = 1

for folder_name in target_folders:
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sampled = random.sample(files, min(len(files), images_per_class))
        
        for file in sampled:
            src = os.path.join(folder_path, file)
            # This forces the exact name the C++ code is looking for
            dst = os.path.join(dest_dir, f"vehicle.{counter}.jpg")
            shutil.copy(src, dst)
            counter += 1
            
print(f"Done! {counter-1} images perfectly formatted for Edge Impulse.")
