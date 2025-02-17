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

batch_size = 10
frames = []
original_dimensions = []
processed_frames = []

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    original_dimensions.append((orig_h, orig_w))
    
    frame_resized = cv2.resize(frame, (640, 640))
    frame_normalized = frame_resized / 255.0
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()
    frames.append(frame_tensor)
    
    if len(frames) == batch_size:
        input_batch = torch.stack(frames).to('cuda')
        
        with torch.no_grad():
            results = model(input_batch, verbose=False)
        
        for i, result in enumerate(results):
            detections = result.boxes
            
            frame = frames[i].permute(1, 2, 0).cpu().numpy() * 255.0  # Convert back to original image format
            frame = frame.astype(np.uint8)
            orig_h, orig_w = original_dimensions[i]
            
            # Go through each detection and get bbox coords, confidence, and class
            for j in range(len(detections)):
                
                #bounding box coordinates returned in tensor format, converted to python array
                xyxy_tensor = detections[j].xyxy.cpu() #tensor format in cpu mem
                xyxy = xyxy_tensor.numpy().squeeze() #tensors to np array
                xmin, ymin, xmax, ymax = xyxy.astype(int) #get indiviual coords and convert to int

                # Get bounding box class ID and name
                classidx = int(detections[j].cls.item())
                classname = labels[classidx]

                # Get bounding box confidence
                conf = detections[j].conf.item()

                # Draw box if confidence threshold is high enough
                if conf > 0.5:
                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, font, 0.5, 1)  #gets label font size
                    label_ymin = max(ymin, labelSize[1] + 15)  #prevent top window clipping
            
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), box_color, cv2.FILLED)  #draw solid box for label
                    cv2.putText(frame, label, (xmin, label_ymin-7), font, 0.5, (0, 0, 0), 1)  #draw label
            
            frame = cv2.resize(frame, (orig_w, orig_h))
            processed_frames.append(frame)
        
        frames = []  # Reset frames list for the next batch
        original_dimensions = []  # Reset original dimensions list
    
                
    
    if processed_frames:
        display_frame = processed_frames.pop(0)  #display next available processed frame
        cv2.putText(display_frame, f'FPS: {avg_fps:0.1f}', (10, 20), font, 1, box_color, 2)  #display fps
        cv2.imshow('Live Inference', display_frame)  #show img

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
