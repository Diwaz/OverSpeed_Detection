import cv2

# define the path to the video file
video_path = "test11.mov"

# define the desired higher resolution
width, height = 1280, 720

# create a VideoCapture object to read the video file
cap = cv2.VideoCapture(video_path)

# set the capture properties to use a high-quality codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# read the first frame to get the original frame size
ret, frame = cap.read()
if not ret:
    print("Error: could not read frame")
    exit()
orig_height, orig_width = frame.shape[:2]

# resize the frame to the desired higher resolution using cv2.INTER_CUBIC
resized_frame = cv2.resize(frame, (width, height),
                           interpolation=cv2.INTER_CUBIC)

# display the resized frame and wait for a key press
cv2.imshow("Frame", resized_frame)
cv2.waitKey(0)

# release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
