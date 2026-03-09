# YOLOv8-GSM
Description
# Install pre-requirements
Install ultralytics library.
```
pip install ultralytics
```
Navigate to your ultralytics library directory that you installed previously, replace it with this ultralytics directory.
(Optional) Install paddle ocr if you want to use the realtime nameplate recognition system. \
Please follow paddleOCR quick start guidelines to install it https://github.com/PaddlePaddle/PaddleOCR.
# Download Dataset
Navigate to https://universe.roboflow.com/ub-sotcz/campus_plate/dataset/2 \
Download the dataset by clicking "Download Dataset" button. \
Choose the YOLOv8 format. \
Select the zip format as download options (You may choose to show download code if you prefer so). \
Inside this dataset folder, there will be `data.yaml` file, please note about this.
# Training YOLO-GSM
For training, you can use this command in the command prompt.
```
yolo detect train batch=16 epochs=200 data={data.yaml location} model="yolov8n-gsm.yaml" pretrained=False optimizer="SGD"
```