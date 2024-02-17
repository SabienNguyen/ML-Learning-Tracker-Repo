import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')

classes = []

with open('coco.names', 'r') as f:
    classes  = [line.strip() for line in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()  
#webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = indices[0]
        box = boxes[i]
        x, y, w, h = box
        label = classes[class_ids[i]]
        confidence = confidences[i]
        percent = "{:.2f}%".format(confidence * 100) # Calculate the confidence percentage
        text = "{}: {}".format(label, percent) # Create the label with the object class and confidence percentage
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    cv2.imshow('frame', frame)

    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
    
        
