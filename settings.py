import cv2
import numpy as np

# Define the filename of the video
with open("videoName.txt", "r") as f:
    video_name = f.read()
filename = video_name

# Load the video
cap = cv2.VideoCapture(filename)

# Read the first frame
ret, frame = cap.read()

# Create a window and display the first frame
cv2.namedWindow("Draw Polygon")
cv2.imshow("Draw Polygon", frame)

# Define the two lists for storing the vertices of the polygons
area1 = []
area2 = []

# Define the function for mouse event


def draw_polygon(event, x, y, flags, param):
    global area1, area2, frame

    # Left mouse button click adds a point to the polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(area1) < 4:
            area1.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        elif len(area2) < 4:
            area2.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # If both polygons are drawn, save the coordinates and exit the window
        if len(area1) == 4 and len(area2) == 4:
            with open('polygon_coordinates.txt', 'w') as f:
                f.write(f"{area1}\n")
                f.write(f"{area2}\n")
            cv2.destroyAllWindows()

    # Right mouse button click removes the last point from the polygon
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(area2) > 0:
            area2.pop()
            frame = frame.copy()
            for pt in area1:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            for pt in area2:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            # cv2.imshow("Draw Polygon", frame)

        elif len(area1) > 0:
            area1.pop()
            frame = frame.copy()
            for pt in area1:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # cv2.imshow("Draw Polygon", frame)


# Set the mouse callback function
cv2.setMouseCallback("Draw Polygon", draw_polygon)

# Loop until the user closes the window
while True:
    cv2.imshow("Draw Polygon", frame)
    frame = cv2.resize(frame, (1280, 720))
    key = cv2.waitKey(1)

    # If the user presses 'q', exit the loop
    if key == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
