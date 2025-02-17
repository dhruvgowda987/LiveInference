import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch

model = YOLO("yolo11n.pt", task='detect')
model.to('cuda')
labels = model.names

cap = cv2.VideoCapture(0)

box_color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
avg_fps = 0
fps_buffer = []
fps_avg_len = 200

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    #preprocessing and normalization
    frame_resized = cv2.resize(frame, (640, 640))
    frame_normalized = frame_resized / 255.0
    input_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float().unsqueeze(0).to('cuda')

    results = model(input_tensor, verbose=False)
    # Extract results
    detections = results[0].boxes

    #extract inference data from  detections
    for i in range(len(detections)):
        #bounding box coordinates returned in tensor format, converted to python array
        xyxy_tensor = detections[i].xyxy.cpu() #tensor format in cpu mem
        xyxy = xyxy_tensor.numpy().squeeze() #tensors to np array
        xmin, ymin, xmax, ymax = xyxy.astype(int) #get indiviual coords and convert to int

        #scale bounding box coordinates back to original image size
        xmin = int(xmin * orig_w / 640)
        ymin = int(ymin * orig_h / 640)
        xmax = int(xmax * orig_w / 640)
        ymax = int(ymax * orig_h / 640)

        #class id/name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        #confidence
        conf = detections[i].conf.item()

        #draw box
        if conf > 0.6:
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, font, 0.5, 1)  #gets label font size
            label_ymin = max(ymin, labelSize[1] + 15)  #prevent top window clipping
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), box_color, cv2.FILLED)  #draw solid box for label
            cv2.putText(frame, label, (xmin, label_ymin-7), font, 0.5, (0, 0, 0), 1)  #draw label
    
    cv2.putText(frame, f'FPS: {avg_fps:0.1f}', (10,20), font, 1, box_color, 2) #display fps
    cv2.imshow('Live Inference', frame) #show img

    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'): #exit
        break

    stop_time = time.perf_counter()
    fps = float(1/(stop_time - start_time))

    #add fps to buffer for avg_fps
    if len(fps_buffer) < fps_avg_len:
        fps_buffer.append(fps)
    else:
        fps_buffer.pop(0)
        fps_buffer.append(fps)

    #avg_fps calculation
    avg_fps = np.mean(fps_buffer)

print(f'Average pipeline FPS: {avg_fps:.2f}')

cap.release()
cv2.destroyAllWindows()
