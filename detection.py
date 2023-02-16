import numpy as np
import cv2

net = cv2.dnn.readNet("yolov4-tiny.weights","newCustom.cfg")
layer_names = net.getLayerNames()
print(layer_names[32])
# output_layer_idx = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# print("here",output_layer_idx);
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("test4.mp4")
classess = []
with open('coco.names', 'r') as f:
    classess = [line.strip() for line in f.readlines()]

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    
    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Perform forward propagation
    net.setInput(blob)
    outputs = net.forward(output_layers)
    print(outputs)
    # Extract the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
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
                confidences.append(float(confidence))
                class_ids.append(class_id)
                print(class_id)

    # Perform non-maximum suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classess[class_ids[i]]}: {confidences[i]:.2f}"
            
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # Show the output frame
    cv2.imshow("Video", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()