import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pyttsx3
import time
import requests
from PIL import Image

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(url)

if response.status_code == 200:
    CLASSES = response.text.strip().split("\n")
else:
    raise Exception("Failed to fetch ImageNet classes. Check your internet connection.")

# Load CLIP model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()

# Define preprocessing transform
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open webcam
cap = cv2.VideoCapture("http://100.76.85.238:8080/video")

last_speak_time = time.time()
speak_delay = 5  # Speak every 5 seconds
frame_skip = 10  # Process every 10th frame for efficiency
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to improve speed

    # Preprocess frame
    input_tensor = preprocess(frame).unsqueeze(0)

    # Perform object recognition
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the highest confidence class
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_class = torch.argmax(probabilities).item()
    detected_label = CLASSES[top_class]

    # Speak detected object if enough time has passed
    current_time = time.time()
    if current_time - last_speak_time > speak_delay:
        detected_text = f"Detected {detected_label}"
        print(detected_text)
        speak(detected_text)
        last_speak_time = current_time

    # Show video feed
    cv2.imshow("Live Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()