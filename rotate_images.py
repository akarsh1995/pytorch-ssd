import cv2
from pathlib import Path
  
# path
dir_path = Path('/home/akarsh/temp/pytorch-ssd/data/number_plate_labels/Pascal_voc/JPEGImages')
  
for image_path in dir_path.glob("*.jpg"):
    image_str = str(image_path)
    # Reading an image in default mode
    src = cv2.imread(image_str)
    image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(image_str, image)
    print("Image {} rotated successfully saved in place.".format(image_str))
