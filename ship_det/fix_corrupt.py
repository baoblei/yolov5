import os
import cv2
import shutil

def fix_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    i = 0
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        # Check if the image is corrupted
        with open(img_path, 'rb') as im:
            im.seek(-2, 2)
            if im.read() == b'\xff\xd9':
                # print('Image OK:', img_name)
                pass
            else:
                # Fix the corrupted image
                img = cv2.imread(img_path)
                cv2.imwrite(output_path, img)
                i += 1
                # print('FIXED corrupted image:', img_name)
    print("fixed images:%d" % (i))            

    j = 0
    # Copy non-corrupted images to the output folder
    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        if not os.path.exists(output_path):
            shutil.copy2(img_path, output_path)
            j += 1
    print("no_corrupt images:%d" % (j))
            # print('Copied non-corrupted image:', img_name)

if __name__ == "__main__":
    input_folder = "images"
    output_folder = "images_fixed"

    fix_images(input_folder, output_folder)
