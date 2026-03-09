import cv2
import math 
import time
import os
import torch
import pandas as pd
import argparse
from ultralytics import YOLO
from paddleocr import PaddleOCR
from openpyxl import Workbook

# Fungsi untuk menghitung Levenshtein Distance
def levenshtein_distance(s1, s2):
    if len(str(s1)) < len(str(s2)):
        return levenshtein_distance(str(s2), str(s1))
    
    if len(s2) == 0:
        return len(str(s1))
    
    previous_row = range(len(str(s2)) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# Character Error Rate (CER) Function
def character_error_rate(s1, s2):
    distance = levenshtein_distance(str(s1), str(s2)) #LD calculate the distance (insert + delete + subtitution)
    return distance / max(len(str(s1)), len(str(s2)))

# Find images function
def find_images_in_subfolders(directory, image_extensions=['.jpg', '.jpeg', '.png']):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

# Setup Argument Parsing
parser = argparse.ArgumentParser(description='Run OCR with optional preprocessing.')
parser.add_argument("--weight", default="weights/yolov8n-gsm.pt", help='Weight PATH for detection is mandatory')
parser.add_argument('--dir', action='store', default="nameplate_campus_test/",help='Define main raw image directory')
parser.add_argument('--save', action='store_true', default=0, help='Enable saving after cropping image')
parser.add_argument('--correct', action='store_true', default=0, help='Enable text correction postprocessing')

args = parser.parse_args()

# Initialize Excel file
excel_data = []
wb = Workbook()
ws = wb.active
ws.append(['Label','Path', 'Object_Index', 'OCR_Result', 'True/False','Computation_Time'])

# Evaluation Variable
total_ld = 0
total_cer = 0
total_wer = 0
total_true = 0
total_img = 0

total_detection_time = 0
total_skew_correction_time = 0
total_ocr_time = 0
total_time = 0
total_text_correction_time = 0

# Initialization EasyOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

# directory
img_dir = args.dir
save_dir = r"realtime_result/"
image_paths = find_images_in_subfolders(img_dir)

# True label of OCR path
label_path = "room_name.xlsx"
label_data = pd.read_excel(label_path)

#detect number
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(args.weight)
classNames = ["plate"]

# Iterasi melalui setiap baris di file Excel
for idx, row in label_data.iterrows():
    subfolder = row['dir']  # Nama subfolder
    unique_label = row['unique']  # Label asli untuk subfolder tersebut
    subfolder_path = os.path.join(img_dir, subfolder)

    # Cek apakah subfolder ada
    if os.path.exists(subfolder_path):
        # Iterasi melalui semua file gambar dalam subfolder
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)

            #default, no object detected
            object_detected = False
            # Load image
            start_time = time.time()
            image = cv2.imread(img_path)
            if image is not None:
                True_False = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #Detect room number object
                results = model(image, stream=True)
                end_time = time.time()
                detection_time = end_time-start_time
                total_detection_time += detection_time
                box_idx = 0
                # coordinates
                for r in results:
                    object_detected = True
                    boxes = r.boxes

                    for box in boxes:
                        # empty the ocr results
                        recognized_text = ""

                        # confidence
                        confidence = math.ceil((box.conf[0]*100))/100
                        if confidence >= 0.7 :
                            
                            box_idx += 1

                            # bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                            # Crop image
                            final_image = image[int(y1):int(y2),int(x1):int(x2)]

                            image_ocr = final_image
                                                       
                            # OCR
                            # Start OCR time
                            start_time = time.time()

                            # Paddle OCR text recognition
                            result = ocr.ocr(image_ocr, cls=True)
                            output = ""
                            if result is not None :
                                for idx in range(len(result)):
                                    res = result[idx]
                                    if res is not None :
                                        for line in res:
                                            output += line[1][0] + " "
                            result = output
                            
                            # Done calculating OCR time and add to total_time
                            end_time = time.time()
                            ocr_time = end_time - start_time
                            total_ocr_time += ocr_time
                            
                            # Show current image process time
                            print(f"Waktu OCR untuk file {img_path}: {ocr_time:.4f} sec")
                            
                            recognized_text = result
                            raw_recognized_text = recognized_text
                   
                            total_img += 1
                            True_False = False
                            if unique_label.replace(" ", "") in recognized_text.replace(" ", "") :
                                total_true += 1
                                True_False = True
                            
                            # If save, then save the results
                            if args.save :
                                # put box in cam
                                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                                # show recognized text on bbox
                                
                                # put text on bounding box
                                org = [x1,y1]
                                fontScale = 1
                                color = (255, 0, 0)
                                thickness = 2
                                cv2.putText(image, recognized_text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

                                #get subfolder name
                                relative_path = os.path.relpath(img_path, img_dir)
                                subfolder_name = os.path.dirname(relative_path) 
                            
                                #create subfolder
                                cropped_dir = os.path.join(save_dir, subfolder_name)
                                os.makedirs(cropped_dir, exist_ok=True)
                                cropped_image_path = os.path.join(cropped_dir, os.path.basename(img_path)) if box_idx==0 else os.path.join(cropped_dir, str(box_idx+1)+"-"+os.path.basename(img_path))
                            
                                # Save crop image
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(cropped_image_path, image)

                                print(f"Cropped image saved to: {cropped_image_path}")

                            
                            # Print result
                            print(f"File: {img_path}")
                            print(f"Recognized Text: {raw_recognized_text}")
                            print(f"Corrected Text: {recognized_text}")
                            print(f"Unique Label: {unique_label}")
                            print(f"Detection Time: {detection_time:.4f} sec")
                            print(f"OCR Time: {ocr_time:.4f} sec")
                            print(f"Process Time: {detection_time+ocr_time:.4f} sec")
                            print('-' * 50)
                            #record excel
                            excel_data.append([unique_label, img_path, box_idx, recognized_text, True_False,detection_time+ocr_time])
                        else:
                            print(f"Confidence to low : {confidence} for {img_path}")
            else:
                print(f"Failed to load image: {img_path}")
                
            if (not object_detected) : excel_data.append([unique_label, img_path, "0", "", True_False, detection_time+ocr_time])

total_time = total_detection_time + total_skew_correction_time + total_ocr_time + total_text_correction_time

# Write data to Excel
for row in excel_data:
    ws.append(row)

wb.save(f"ocr_test_result.xlsx")

print("Process Done.")
print(f"AVG Levenshtein Distance: {total_ld / total_img}")
print(f"AVG Character Error Rate: {total_cer / total_img:.4f}")
print(f"AVG Word Error Rate: {total_wer / total_img:.4f}")
print(f"Room Accuracy: {total_true / total_img:.4f}")
print(f"AVG Detection Process Time : {total_detection_time / total_img:.4f} sec")
print(f"AVG Skew Correction Process Time : {total_skew_correction_time / total_img:.4f} sec")
print(f"AVG OCR Process Time : {total_ocr_time / total_img:.4f} sec")
print(f"AVG Text Correction Process Time : {total_text_correction_time / total_img:.4f} detik")
print(f"AVG Image Process Time : {total_time / total_img:.4f} sec")
print(f"OCR Total Process Time: {total_time:.4f} sec")