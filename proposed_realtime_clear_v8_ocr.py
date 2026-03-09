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

def read(image_ocr):
    
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

    #correct the recognized text
    recognized_text = correct(ocr_result)

    return recognized_text

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
save_name = f"realtime_result/proposed_{save_name}"

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

# For tracking
object_list = {}
object_id = 0
distance_threshold = 0.005*frame_width*frame_height ## 0.005 from the screen size
ocr_age_threshold = 18 ## based on wheelchair speed and paddleOCR readability distance

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
                current_object_id = object_id # default object id, will change if similar object detected

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # centroid
                cx, cy, w, h = box.xywh[0]

                # Object tracking method
                if not object_list :
                    # if there was no object detected yet
                    new_object = {"cx": cx, "cy": cy, "obj_age": 0, "ocr_age": -1, "text": ""}
                    object_id+=1
                    object_list[current_object_id] = new_object
                else :
                    candidate_list = {}
                    for obj, info in object_list.items():
                        p2 = (info["cx"],info["cy"]) # Previous centroid point
                        p1 = cx, cy # Current centroid point
                        distance = math.dist(p1,p2)

                        if distance < distance_threshold : # If distance is lower than threshold, add to candidate
                            candidate_list[obj] = {"distance":distance}

                    if candidate_list :                            
                        # Select the nearest object and update the centroids and age
                        lowest_id = min(candidate_list, key=lambda k: candidate_list[k]["distance"])
                        object_list[lowest_id]["cx"] = cx
                        object_list[lowest_id]["cy"] = cy
                        object_list[lowest_id]["obj_age"] = 0
                        current_object_id = lowest_id
                    else :
                        # If there is no nearby object candidates, add new object id
                        new_object = {"cx": cx, "cy": cy, "obj_age": 0, "ocr_age": -1, "text": ""}
                        object_id+=1
                        object_list[current_object_id] = new_object
                
                # Start OCR timer
                start_time = time.time()
                # Proceed to OCR if ocr age older than ocr_age_threshold or -1
                if (object_list[current_object_id]["ocr_age"]) > ocr_age_threshold or (object_list[current_object_id]["ocr_age"] == -1) :
                    # Crop image
                    final_image = img[int(y1):int(y2),int(x1):int(x2)]

                    # Set image for OCR
                    image_ocr = final_image

                    # OCR process
                    object_list[current_object_id]["text"] = read(image_ocr)

                    # Reset ocr age
                    object_list[current_object_id]["ocr_age"] = 0

                # put text on bounding box
                object_text = object_list[current_object_id]["text"]

                org = [x1,y1]
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, object_text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

                ###############################################################################################
                
                # Calculate OCR time and add to total time
                end_time = time.time()
                ocr_time = end_time - start_time
                total_ocr_time += ocr_time

                # put box in cam based on bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # ['Frame', 'Object_ID','OCR_Result','Computation_Time', 'FPS']
                excel_data.append([frame_count, current_object_id, object_text, total_ocr_time, fps])

                # Update age of all OCR and object
                for info in object_list.values():
                    info["obj_age"] += 1
                    info["ocr_age"] += 1
                
                # remove old objects from obj list
                old_objects = [id for id, info in object_list.items() if info["obj_age"] > 30] # assuming 30 fps = 1 second. So, object 
                for id in old_objects:
                    del object_list[id]


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