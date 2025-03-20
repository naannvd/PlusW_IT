#%%
import numpy as np
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

def arr_functions(arr1d, arr2d):
    print("1D Array: ", arr1d)
    print("2D Array: ", arr2d)

    print('Sum of 1D Array: ', np.sum(arr1d))
    print('Mean of 2D Array: ', np.mean(arr2d))
    print('Transpose of 2D Array: ', np.transpose(arr2d))

arr_functions(arr_1d, arr_2d)
# %%
def numpyImage():
    image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    print("Original Image:\n", image)
    # Slicing a portion of the image
    cropped = image[1:4, 1:4]
    print("Cropped Section:\n", cropped)
    # Inverting colors
    inverted_image = 255 - image
    print("Inverted Image:\n", inverted_image)

numpyImage()
# %%
import cv2
import os
import numpy as np

# print(os.getcwd())
# print(os.listdir())
# os.chdir('Lecture_4')
# print(os.getcwd())  
# print(os.listdir())

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

# %%
import cv2
import numpy as np
# Load an image and convert it to a NumPy array
image = cv2.imread('mamad.jpg') # Replace with your image path
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
# Detect faces in the image
faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1,
minNeighbors=5)
    # Loop through the detected faces and extract facial features (regions)
for (x, y, w, h) in faces:# Slice the image array to extract the face region
    face_region = image[y:y+h, x:x+w]
    # Optional: Display the face region
    cv2.imshow('Face Region', face_region)
    # Extract additional facial features if required (e.g., eyes, nose)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
    'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(face_region, scaleFactor=1.1,
    minNeighbors=5)
for (ex, ey, ew, eh) in eyes:
    eye_region = face_region[ey:ey+eh, ex:ex+ew]
    cv2.imshow('Eye Region', eye_region)

cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
