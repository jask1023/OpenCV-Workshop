#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Install OpenCV
get_ipython().system('pip install opencv-python')


# In[8]:


import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the video feed
    cv2.imshow("Webcam", frame)

    # Break on prqessing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[10]:


import cv2

# Capture video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Display the processed frame
    cv2.imshow("Edge Detection", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[15]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for a color (e.g., black)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 30])

    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours and draw bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 1000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[17]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

sprite_position = [320, 240]  # Starting position of the sprite (center of frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for the target color (e.g., blck)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 30])

    # Create a mask and find contours
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)

        # Calculate the center of the detected object
        object_center = (x + w // 2, y + h // 2)

        # Move sprite towards object center
        sprite_position[0] = object_center[0]
        sprite_position[1] = object_center[1]

    # Draw the "sprite" (e.g., a simple circle)
    cv2.circle(frame, tuple(sprite_position), 20, (255, 0, 0), -1)

    # Display the resulting frame
    cv2.imshow("Game Mechanic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

