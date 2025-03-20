import cv2
import os
import numpy as np

print(os.getcwd())
# print(os.listdir())
os.chdir('Lecture_4')
print(os.getcwd())  
print(os.listdir())

image_path = 'uWu.jpg'  # Change if necessary
# Check if file exists before reading
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found. Check the path!")
    exit()

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read the image from {image_path}. Try opening it manually.")
    exit()

# Scaling Transformation
def scale_image(image, scale_factor):
    rows, cols = image.shape[:2]
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Rotation Transformation
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Translation Transformation
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, (cols, rows))

# Apply transformations
scaled_image = scale_image(image, 1.5)
rotated_image = rotate_image(image, 45)
translated_image = translate_image(image, 50, 30)

# Show results
cv2.imshow('Original', image)
cv2.imshow('Scaled', scaled_image)
cv2.imshow('Rotated', rotated_image)
cv2.imshow('Translated', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()