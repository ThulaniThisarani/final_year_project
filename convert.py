import os
from PIL import Image
import pillow_heif

input_folder = "dataset"  # main dataset folder

# Loop through all folders and convert HEIC files
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".heic"):
            heic_path = os.path.join(root, file)
            jpg_path = os.path.join(root, file.rsplit(".", 1)[0] + ".jpg")
            
            heif_file = pillow_heif.read_heif(heic_path)
            img = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
            )
            img.save(jpg_path, "JPEG", quality=95)
            print(f"Converted: {heic_path} → {jpg_path}")

print("All HEIC files converted to JPG successfully.")
