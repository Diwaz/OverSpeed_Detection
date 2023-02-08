import cv2

# Load an image
img = cv2.imread("image.jpg")

# Display the image
cv2.imshow("Image", img)

# Wait for a key press
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAll
