import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import threading

model = YOLO("yolo11n.pt", task='detect')
model.to('cuda')
labels = model.names

cap = cv2.VideoCapture(0)

box_color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
avg_fps = 0
fps_buffer = []
fps_avg_len = 200

batch_size = 10  # Adjustable batch size
frames = []
original_dimensions = []
processed_frames = []
frame_lock = threading.Lock()
latest_frame = None

# Load the CUDA kernel
with open('normalize.cu', 'r') as f:
    kernel_code = f.read()

module = SourceModule(kernel_code)
normalize_image = module.get_function('normalize_image')

def capture_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            latest_frame = frame

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

def process_batch(batch_frames, batch_dimensions):
    input_batch = torch.stack(batch_frames).to('cuda')

    # Ensure the tensor is on the GPU
    input_batch_ptr = input_batch.data_ptr()

    # Define block and grid sizes
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(input_batch.shape[3] / block_size[0])), int(np.ceil(input_batch.shape[2] / block_size[1])), batch_size)

    # Launch the CUDA kernel for normalization 
    normalize_image(np.intp(input_batch_ptr), np.int32(input_batch.shape[3]), np.int32(input_batch.shape[2]), np.int32(batch_size), block=block_size, grid=grid_size)
    
    # Synchronize to ensure kernel execution is complete
    cuda.Context.synchronize()

    with torch.no_grad():
        results = model(input_batch, verbose=False)

    processed_batch = []
    for i in range(len(results)):
        detections = results[i].boxes
        frame = input_batch[i].permute(1, 2, 0).cpu().numpy() * 255.0 #puts frame data in non-contiguous memory
        frame = frame.astype(np.uint8)  # Ensure the frame is in uint8 format
        orig_h, orig_w = batch_dimensions[i]

        # Go through each detection and get bbox coords, confidence, and class
        for j in range(len(detections)):
            # Bounding box coordinates
            xyxy_tensor = detections[j].xyxy.cpu()  # Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze()  # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int)  # Extract individual coordinates and convert to int

            # Ensure the frame is compatible with cv::rectangle
            frame = np.ascontiguousarray(frame)

            # Get bounding box class ID and name
            classidx = int(detections[j].cls.item())
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[j].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > 0.5:
                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, font, 0.5, 1)  # Get label font size
                label_ymin = max(ymin, labelSize[1] + 15)  # Prevent top window clipping

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), box_color, cv2.FILLED)  # Draw solid box for label
                cv2.putText(frame, label, (xmin, label_ymin-7), font, 0.5, (0, 0, 0), 1)  # Draw label

        frame = cv2.resize(frame, (orig_w, orig_h))
        processed_batch.append(frame)

    return processed_batch

while True:
    start_time = time.perf_counter()

    with frame_lock:
        frame = latest_frame
        if frame is None:
            continue

    orig_h, orig_w = frame.shape[:2]
    original_dimensions.append((orig_h, orig_w))
    
    frame_resized = cv2.resize(frame, (640, 640))

    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()
    frames.append(frame_tensor)
    
    if len(frames) == batch_size:
        processed_frames.extend(process_batch(frames, original_dimensions))
        frames = []  # Reset frames list for the next batch
        original_dimensions = []  # Reset original dimensions list

    if processed_frames:
        display_frame = processed_frames.pop(0)  # Display next available processed frame
        cv2.putText(display_frame, f'FPS: {avg_fps:0.1f}', (10, 20), font, 1, box_color, 2)  # Display fps
        cv2.imshow('Live Inference', display_frame)  # Show image

    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):  # Exit
        break

    stop_time = time.perf_counter()
    fps = float(1 / (stop_time - start_time))

    # Add fps to buffer for avg_fps
    if len(fps_buffer) < fps_avg_len:
        fps_buffer.append(fps)
    else:
        fps_buffer.pop(0)
        fps_buffer.append(fps)

    # Calculate avg_fps
    avg_fps = np.mean(fps_buffer)

print(f'Average pipeline FPS: {avg_fps:.2f}')

cap.release()
cv2.destroyAllWindows()
capture_thread.join()
