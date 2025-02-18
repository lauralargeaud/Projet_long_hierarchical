import os
import glob

def keep_only_first_image(root_folder):
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        
        if os.path.isdir(subdir_path):
            image_files = sorted(glob.glob(os.path.join(subdir_path, "*.*")))
            image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]
            
            if len(image_files) > 1:
                for img in image_files[1:]:
                    os.remove(img)
                    print(f"Deleted: {img}")
                print(f"Kept: {image_files[0]}")
            elif len(image_files) == 1:
                print(f"Only one image found, kept: {image_files[0]}")
            else:
                print(f"No images found in {subdir_path}")