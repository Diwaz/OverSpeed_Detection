import numpy as np
import cv2
import math
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

net = cv2.dnn.readNet("yolov4-tiny.weights", "newCustom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
# print(layer_names)
# output_layer_idx = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# print("here",output_layer_idx);
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
start_time = None
elapsed_time = 0
cap = cv2.VideoCapture("test9.mp4")
object_tracker = DeepSort(
    max_age=5,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,

    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None,)
classess = []
with open('coco.names', 'r') as f:
    classess = [line.strip() for line in f.readlines()]

count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0


def trackM(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')


while True:
    # Read a frame from the video
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    ret, frame = cap.read()

    area1 = [(1273, 3),
             (652, 3),
             (654, 717),
             (1272, 714)]
    area2 = [(512, 274),
             (222, 648),
             (1250, 624),
             (1066, 279)]
    area3 = [
        (1092, 288),
        (667, 416),
        (700, 698),
        (1273, 406)
    ]

    # cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    # cv2.polylines(frame, [np.array(area2, np.int32)], True, (155, 0, 0), 6)
    # cv2.rectangle(frame, (650, 00), (1279, 750), (0, 255, 0), 2)
    # cv2.rectangle(frame, (650, 700), (1250, 350), (0, 255, 0), 2)
    center_points_cur_frame = []
    count += 1
    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Perform forward propagation
    net.setInput(blob)
    outputs = net.forward(output_layers)
   # print(outputs)
    # Extract the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    detections = []
    if start_time is None:
        start_time = time.time()

        # Measure the time spent processing the frame
        # Perform some basic image processing on the frame
        # (in this example, we will just convert the frame to grayscale)

        # Calculate the elapsed time since the timer was started
        elapsed_time = time.time()

        # Add the elapsed time to the display
        timer_text = "Elapsed Time: {:.2f}s".format(elapsed_time)
        print(timer_text)

        cv2.putText(frame, timer_text, (800, 61), font, 1, color, 2)
    for output in outputs:

        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                label = "vehicle"
                confidences.append(float(confidence))
                class_ids.append(class_id)
                print(detections)
                detections.append(
                    ([x, y, x+w, y+h], confidence, 'vehicle'))
                # ([center_x, center_y, int(w-center_x), int(h-center_y)], confidence, label))

    # Perform non-maximum suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    tracks = object_tracker.update_tracks(detections, frame)
    print(tracks)
    # for track in tracks:
    #     if not track.is_confirmed():
    #         continue
    #     track_id = track.track_id
    #     ltrb = track.to_ltrb()

    #     bbox = ltrb
    #     print('here', bbox)
    #     cv2.rectangle(frame, (int(bbox[0]), int(
    #         bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 200, 255), 2)

    # Draw the bounding boxes on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            center_points_cur_frame.append((cx, cy))
            label = f"{classess[class_ids[i]]}"
            # result = cv2.pointPolygonTest(
            #     np.array(area2, np.int32), (int(cx), int(cy)), False)
            # if (result > 0):
            color = (0, 255, 0)
            # cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            print('same as', [x, y, x+w, y+h])
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # if count <= 2:
    #     for pt in center_points_cur_frame:
    #         for pt2 in center_points_prev_frame:
    #             distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

    #             if distance < 60:
    #                 tracking_objects[track_id] = pt
    #                 track_id += 1
    # else:

    #     tracking_objects_copy = tracking_objects.copy()
    #     center_points_cur_frame_copy = center_points_cur_frame.copy()

    #     for object_id, pt2 in tracking_objects_copy.items():
    #         object_exists = False
    #         for pt in center_points_cur_frame_copy:
    #             distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

    #             # Update IDs position
    #             if distance < 60:
    #                 tracking_objects[object_id] = pt
    #                 object_exists = True
    #                 if pt in center_points_cur_frame:
    #                     center_points_cur_frame.remove(pt)
    #                 continue

    #         # Remove IDs lost
    #         if not object_exists:
    #             tracking_objects.pop(object_id)

    #     # Add new IDs found
    #     for pt in center_points_cur_frame:
    #         tracking_objects[track_id] = pt
    #         track_id += 1
    #     for object_id, pt in tracking_objects.items():
    #         result = cv2.pointPolygonTest(
    #             np.array(area2, np.int32), (pt[0], pt[1]), False)
    #         if (result > 0):
    #             cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    #             cv2.putText(frame, str(object_id),
    #                         (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    # # Show the output frame
    cv2.imshow("Video", frame)
    cv2.setMouseCallback('Video', trackM)
    # center_point_prevFrame = center_points_cur_frame.copy()

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
