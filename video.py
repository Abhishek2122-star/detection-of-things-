import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# Load YOLOv8 model for person detection
yolo_model = YOLO('yolov8n.pt')  # Lightweight YOLOv8 model

# Load EfficientNetV2 model for gender classification
gender_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
gender_model.eval()

# Define transform for image preprocessing for EfficientNetV2
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((224, 224)),  # Resize to EfficientNet input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Load OpenCV pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_gender(person_img):
    """
    Classify the gender of the cropped person image using EfficientNetV2.
    """
    try:
        person_img = transform(person_img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = gender_model(person_img)
            _, predicted = torch.max(outputs, 1)
        return 'female' if predicted.item() == 0 else 'male'
    except Exception as e:
        print(f"Error in gender classification: {e}")
        return "Person"

def detect_and_classify(frame):
    """
    Detect persons using YOLO, refine detections to faces, and classify gender.
    """
    results = yolo_model(frame, conf=0.5)  # Adjust confidence threshold if needed

    for result in results:
        detections = result.boxes.data.cpu()  # Bounding boxes as a tensor
        for box in detections:
            cls = int(box[-1].item())  # Get class ID
            if cls == 0:  # Class 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates

                # Crop the detected person
                person_img = frame[y1:y2, x1:x2]

                # Ensure valid crop
                if person_img.size == 0:
                    continue

                # Detect face within the cropped person image
                gray_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=5)

                for (fx, fy, fw, fh) in faces:
                    face_img = person_img[fy:fy + fh, fx:fx + fw]  # Extract face region
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                    # Classify gender using EfficientNet
                    gender = classify_gender(face_img_rgb)

                    # Draw bounding box and label on the frame
                    label = f'{gender}'
                    cv2.rectangle(frame, (x1 + fx, y1 + fy), (x1 + fx + fw, y1 + fy + fh), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1 + fx, y1 + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def run_camera():
    """
    Open the camera, run YOLO for person detection, and classify gender using face crops.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect persons and classify gender
        detect_and_classify(frame)

        # Show the frame with detections
        cv2.imshow('Person Detection and Gender Classification', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()
