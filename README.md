# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm :
### Step 1 :

Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

### Step 2 :


Read the input image using cv2.imread() and store it in a variable for further processing.

### Step 3 :

Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis. 2.Scaling resizes the image by scaling factors. 3.Shearing distorts the image along one axis. 4.Reflection flips the image horizontally or vertically. 5.Rotation rotates the image by a given angle.

### Step 4 :


Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

### Step 5 :


Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.

## Program :
```python
Developed By: Ayisha Rinsi K
Register Number: 212223040022

##i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_img=cv2.imread("color image of flower.png")
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_img)
plt.show()
rows, cols, dim = input_img.shape
M = np.float32([[1, 0, 20],
                [0, 1, 50]])
translated_img = cv2.warpAffine(input_img, M, (cols, rows))
plt.axis('off')
plt.imshow(translated_img)
plt.show()


## ii) Image Scaling
scale_matrix = np.float32([[1.5, 0,   0],   # 1.5 × width
                           [0,   1.5, 0],   # 1.5 × height
                           [0,   0,   1]])
scaled_img = cv2.warpPerspective(input_img,
                                 scale_matrix,
                                 (int(cols * 1.5), int(rows * 1.5)))

plt.axis('off')
plt.imshow(scaled_img)
plt.show()

## iii) Image Shearing
M_x = np.float32([[1, 0.2, 0],
                  [0, 1,   0],
                  [0, 0,   1]])

# --- vertical shear (y–axis) ---
M_y = np.float32([[1,   0, 0],
                  [0.2, 1, 0],
                  [0,   0, 1]])

sheared_img_xaxis = cv2.warpPerspective(input_img, M_x, (cols, rows))
sheared_img_yaxis = cv2.warpPerspective(input_img, M_y, (cols, rows))

plt.axis('off')
plt.imshow(sheared_img_xaxis)
plt.show()

plt.axis('off')
plt.imshow(sheared_img_yaxis)
plt.show()

## iv) Image Reflection
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread("color image of flower.png")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)   # add missing conversion code

plt.axis("off")
plt.imshow(input_image)
plt.show()

rows, cols, dim = input_image.shape

M_x = np.float32([[1,  0, 0],                                 
                  [0, -1, rows],
                  [0,  0, 1]])
M_y = np.float32([[-1, 0, cols],                         
                  [0,  1, 0],
                  [0,  0, 1]])

reflected_img_xaxis = cv2.warpPerspective(input_image, M_x, (cols, rows))
reflected_img_yaxis = cv2.warpPerspective(input_image, M_y, (cols, rows))

plt.axis("off")
plt.imshow(reflected_img_xaxis)                             
plt.show()

plt.axis("off")
plt.imshow(reflected_img_yaxis)
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread("color image of flower.png")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

rows, cols, _ = input_image.shape
angle = 45

M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
rotated_img = cv2.warpAffine(input_image, M, (cols, rows))

plt.axis("off")
plt.imshow(rotated_img)
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread("color image of flower.png")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Define cropping coordinates: [y1:y2, x1:x2]
cropped_img = input_image[100:400, 150:450]  # example values

plt.axis("off")
plt.imshow(cropped_img)
plt.show()

```
## Output:
<img width="567" height="747" alt="image" src="https://github.com/user-attachments/assets/f6170d22-7dc9-4bb8-9bf7-197192398563" />

<img width="582" height="750" alt="image" src="https://github.com/user-attachments/assets/46254592-63c2-4503-8ce3-2169cf6241d2" />

<img width="493" height="818" alt="image" src="https://github.com/user-attachments/assets/3b63bf9d-ef31-40a0-8d1a-862fb9e7cdc9" />

<img width="478" height="305" alt="image" src="https://github.com/user-attachments/assets/2ea64290-4c66-4374-a7c0-d101450605cb" />

<img width="466" height="316" alt="image" src="https://github.com/user-attachments/assets/46838526-0944-45ac-add7-ba01a18701a7" />

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
