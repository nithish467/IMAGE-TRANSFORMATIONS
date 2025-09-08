# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
## Step 1: 
Start
## Step 2: Import necessary libraries
Import cv2 for image processing.
Import numpy for matrix operations.

## Step 3: Load the input image
Read the image using cv2.imread().
Convert the color from BGR to RGB if needed (for display with matplotlib).

## Step 4: Apply Transformations
a. Translation

Define a translation matrix to shift the image.

Apply cv2.warpAffine() with the translation matrix.

b. Scaling

Use cv2.resize() with scaling factors (fx, fy) to resize the image.

c. Shearing

Define a shearing transformation matrix.

Apply cv2.warpAffine() with the shearing matrix.

d. Reflection (Flipping)

Use cv2.flip() to reflect the image horizontally or vertically.

e. Rotation

Get the rotation matrix using cv2.getRotationMatrix2D().

Apply cv2.warpAffine() with the rotation matrix.

f. Cropping

Slice the image array to extract a specific region.

Step 5: Display or save the output images

Use matplotlib.pyplot or cv2.imshow() to display images.

(Optional) Save results using cv2.imwrite().
## Step 6: 
End

## Program:
```python
Developed By:
Register Number:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('nithish.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# 1. Image Translation
rows, cols, _ = image.shape
M_translate = np.float32([[1, 0, 50], [0, 1, 100]])  # Translate by (50, 100) pixels
translated_image = cv2.warpAffine(image_rgb, M_translate, (cols, rows))

# 2. Image Scaling
scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)  # Scale by 1.5x

# 3. Image Shearing
M_shear = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Shear with factor 0.5
sheared_image = cv2.warpAffine(image_rgb, M_shear, (int(cols * 1.5), int(rows * 1.5)))

# 4. Image Reflection (Flip)
reflected_image = cv2.flip(image_rgb, 1)  # Horizontal reflection (flip along y-axis)

# 5. Image Rotation
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotate by 45 degrees
rotated_image = cv2.warpAffine(image_rgb, M_rotate, (cols, rows))

# 6. Image Cropping
cropped_image = image_rgb[50:300, 100:400]  # Crop a portion of the image

# Plot the original and transformed images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(translated_image)
plt.title("Translated Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(scaled_image)
plt.title("Scaled Image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(reflected_image)
plt.title("Reflected Image")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis('off')

plt.tight_layout()
plt.show()

# Plot cropped image separately as its aspect ratio may be different
plt.figure(figsize=(4, 4))
plt.imshow(cropped_image)
plt.title("Cropped Image")
plt.axis('off')
plt.show()


```
## Output:
### i)Image Translation
<br>
<br>
<img width="287" height="377" alt="image" src="https://github.com/user-attachments/assets/d4351a37-7b83-4a97-9414-a979f28e6f11" />

<br>
<br>

### ii) Image Scaling
<br>
<br>
<img width="273" height="371" alt="image" src="https://github.com/user-attachments/assets/348d6a5e-f404-4fe5-a8bf-6b056d6dda65" />

<br>
<br>


### iii)Image shearing
<br>
<br>
<img width="253" height="377" alt="image" src="https://github.com/user-attachments/assets/5aa5776f-53d0-4cbf-9a8c-fe02bf2fb65e" />

<br>
<br>


### iv)Image Reflection
<br>
<br>
<img width="277" height="371" alt="image" src="https://github.com/user-attachments/assets/4fdd6dd8-87e0-427f-a237-482ccc269e78" />

<br>
<br>



### v)Image Rotation
<br>
<br>
<img width="252" height="362" alt="image" src="https://github.com/user-attachments/assets/0717ce3d-9bd5-4e6f-9743-26d8ed88e3f3" />

<br>
<br>



### vi)Image Cropping
<br>
<br>
<img width="315" height="291" alt="image" src="https://github.com/user-attachments/assets/c39b2714-8ce6-45aa-aff8-67ad0c0227af" />

<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
