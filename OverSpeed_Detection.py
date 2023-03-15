import tkinter as tk
import os


def select_boundary():
    os.system('python settings.py')


def launch_app():
    os.system('python detection.py')


def save_video_name():
    video_name = video_entry.get()
    # You can use this video_name variable to load the video in detections.py or elsewhere
    print("Video name saved as:", video_name)
    with open('videoName.txt', 'w') as f:
        f.write(f"{video_name}")


root = tk.Tk()
root.title("Vehicle Detection App")

# Create video name input label and entry box
video_label = tk.Label(root, text="Enter video name:")
video_label.pack()
video_entry = tk.Entry(root)
video_entry.pack()

# Create button to launch the settings.py file
boundary_button = tk.Button(
    root, text="Select Boundary", command=select_boundary)
boundary_button.pack()

# Create button to launch the detections.py file
app_button = tk.Button(root, text="Launch App", command=launch_app)
app_button.pack()

# Create button to save the video name
save_button = tk.Button(root, text="Save Video Name", command=save_video_name)
save_button.pack()

root.mainloop()
