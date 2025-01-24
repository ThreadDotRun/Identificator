import os
from PIL import Image

def convert_webp_to_png(root_dir="./testimages/"):
    """
    Converts all .webp files in the given directory and its subdirectories to .png,
    overwriting the original .webp files.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(".webp"):
                file_path = os.path.join(dirpath, file)
                png_path = os.path.splitext(file_path)[0] + ".png"
                
                # Open and convert the image
                try:
                    with Image.open(file_path) as img:
                        img.save(png_path, "PNG")
                    
                    # Remove the original .webp file
                    os.remove(file_path)
                    print(f"Converted and replaced: {file_path} -> {png_path}")
                except Exception as e:
                    print(f"Failed to convert {file_path}: {e}")

if __name__ == "__main__":
    convert_webp_to_png()
