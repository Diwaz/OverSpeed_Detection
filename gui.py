import tkinter as tk
import cv2
import numpy as np

# Define the Tkinter window
window = tk.Tk()
window.title("Video Player")

# Define the Canvas widget for displaying the video frames
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Define the OpenCV video capture object
cap = cv2.VideoCapture("test3.mp4")

# Read the first frame of the video
ret, frame = cap.read()

# Define the polygon points variable
poly_pts = []

# Define the function to draw the polygon on the first frame of the video


def draw_polygon(event):
    global poly_pts
    poly_pts.append([event.x, event.y])
    if len(poly_pts) > 1:
        canvas.create_line(poly_pts[-2][0], poly_pts[-2][1],
                           poly_pts[-1][0], poly_pts[-1][1], fill="red", width=2)


canvas.bind("<Button-1>", draw_polygon)

# Define the function to start playing the video with the polygon


def start_video():
    global frame, cap, poly_pts

    # Draw the polygon on the first frame of the video
    poly_frame = np.copy(frame)
    cv2.fillPoly(poly_frame, np.array([poly_pts]), (0, 255, 0))

    # Define the function to update the video frames with the polygon
    def update_frame_with_polygon():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Draw the polygon on the frame
            poly_mask = np.zeros_like(frame[:, :, 0])
            cv2.fillPoly(poly_mask, np.array([poly_pts]), 255)
            frame = cv2.bitwise_and(frame, frame, mask=poly_mask)

            # Update the Canvas widget with the new frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            canvas.img = img
            canvas.create_image(0, 0, image=img, anchor=tk.NW)

        # Schedule the next frame update
        canvas.after(15, update_frame_with_polygon)

    # Start playing the video with the polygon
    update_frame_with_polygon()


# Add a Play button to start playing the video
play_button = tk.Button(window, text="Play", command=start_video)
play_button.pack()

# Start the Tkinter main loop
window.mainloop()


# Release the OpenCV video capture object and destroy the window
cap.release()
cv2.destroyAllWindows()
