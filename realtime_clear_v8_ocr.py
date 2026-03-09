import cv2
import math 
import time
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
import psutil
from paddleocr import PaddleOCR
from openpyxl import Workbook
from ultralytics import YOLO

# True label of OCR path
label_path = r"room_name.xlsx"
label_dict = pd.read_excel(label_path)

# Room number correction algorithm
def correct(text): 
    match = text
    text = text.replace(" ","")
    for idx, row in label_dict.iterrows():
        unique_label = row['unique']
        if unique_label.replace(" ", "") in text : match = (row['unique'])
    return match

# Setup Argument Parsing
parser = argparse.ArgumentParser(description='Run OCR with optional preprocessing.')
parser.add_argument("--weight", default="weights/yolov8n-gsm.pt", help='Weight PATH for detection is mandatory')
parser.add_argument('--video', action='store', default="videos/black.mp4",help='Video source, 0 for camera')
parser.add_argument('--cpu', action='store_true', default=0, help='Using CPU instead of GPU')

args = parser.parse_args()

# Initialize Excel file
excel_data = []
wb = Workbook()
ws = wb.active
ws.append(['Frame', 'Object_ID','OCR_Result','Computation_Time', 'FPS'])

# OCR Model Load
ocr = PaddleOCR(use_angle_cls=True, lang='en')
total_ocr_time = 0

# start webcam
vid_folder = f"{args.video}"
cap = cv2.VideoCapture(vid_folder)

#set open camera size
if (cap.isOpened() == False):  
    print("Error reading video file") 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4))   
size = (frame_width, frame_height)

# Decide filename
save_name = f"{os.path.basename(args.video)}_{os.path.basename(args.weight)}"
save_name = f"realtime_result/base_{save_name}"

# Video filename writer
writer = cv2.VideoWriter(f'{save_name}.avi',cv2.VideoWriter_fourcc(*'MJPG'),25, size) 

# Read the weight argument
weight = f"{args.weight}"

# oad model
model = YOLO(weight)

# calculate frame
frame_count=1

# for calculate fps
prev_frame_time = 0
new_frame_time = 0
total_fps = 0
min_fps = 999
max_fps = 0
fps = 30 #default for first frame

# for calculate RAM
total_RAM = 0

# For detection label
object_detected = False

while True:
    success, img = cap.read()
    if not success : break
    print("Frame -->", frame_count)
    results = model(img)
    frame_count+=1
    object_index = 0
    
    # process each detected objects
    for r in results:

        boxes = r.boxes
        for box in boxes:
            
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            if confidence >= 0.5 : #based on mAP 0.5

                # set to true for excel
                object_detected = True
                # object index for excel
                object_index += 1

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # Crop image
                final_image = img[int(y1):int(y2),int(x1):int(x2)]

                # Set image for OCR
                image_ocr = final_image

                # OCR process
                # Hitung waktu mulai OCR
                start_time = time.time()
                
                # Paddle text recognition
                ocr_result = ""
                if image_ocr is None or image_ocr.shape[1] == 0 or image_ocr.shape[0] == 0:
                    print("Image not loaded properly or has zero width/height.")
                else :
                    ocr_result = ocr.ocr(image_ocr, cls=True)
                output = ""
                if ocr_result is not None :
                    for idx in range(len(ocr_result)):
                        res = ocr_result[idx]
                        if res is not None :
                            for line in res:
                                output += line[1][0] + " "
                ocr_result = output

                # Calculate OCR time and add to total time
                end_time = time.time()
                ocr_time = end_time - start_time
                total_ocr_time += ocr_time

                # Show time needed to process OCR by this object 
                print(f"OCR time for frame {frame_count}, at object {object_index}: {ocr_time:.4f} sec")

                #correct the recognized text
                recognized_text = correct(ocr_result)

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # show recognized text on bbox
                object_text = recognized_text
                
                # put text on bounding box
                org = [x1,y1]
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, object_text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

                # ['Frame', 'Object_ID','OCR_Result','Computation_Time', 'FPS']
                excel_data.append([frame_count, object_index, object_text, total_ocr_time, fps])

    # FPS calculation
    new_frame_time = time.time()
    frame_time = (new_frame_time+0.001)-prev_frame_time #0.001 to prevent zero division
    fps = 1/frame_time 
    prev_frame_time = new_frame_time

    if fps > max_fps: max_fps=fps #update max fps
    if fps < min_fps: min_fps=fps #updat min fps
    total_fps += fps

    fps = int(fps)
    cv2.putText(img, f"fps: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # RAM calculation
    process = psutil.Process()
    RAM_usage = (process.memory_info().rss)/1000000  # in bytes 
    total_RAM += RAM_usage

    # Record at excel file if there is no detection
    # ['Frame', 'Object_ID','OCR_Result','Computation_Time', 'FPS']
    if object_detected==False : excel_data.append([frame_count, "-", "", frame_time, fps])

    writer.write(img)
    # Show image
    cv2.imshow('Webcam', img) 

    # if q key pressed, then exit the real time camera
    if (cv2.waitKey(1) == ord('q')):
        break

# average fps
print("Average FPS -->", total_fps/frame_count)
print("Min FPS -->", min_fps)
print("Max FPS -->", max_fps)

# average RAM
print("Average RAM -->", total_RAM/frame_count)

# Write data to Excel
for row in excel_data:
    ws.append(row)

wb.save(f"{save_name}_realtime_data_walk.xlsx")

# Release cap
cap.release()
cv2.destroyAllWindows()

#command: python realtime_clear_v8_ocr.py --weight yolov8-ghost-casppf --preprocess inversion --video "F 3.1.mp4"