import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# Load YOLOv8 model for person detection
yolo_model = YOLO('yolov8n.pt')  # Use 'n' version for lightweight, adjust as needed

# Load EfficientNetV2 model for gender classification
gender_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
gender_model.eval()

# Define transform for image preprocessing for EfficientNetV2
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to EfficientNet input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

def classify_gender(person_img):
    """
    Classify the gender of the cropped person image using EfficientNetV2.
    """
    try:
        person_img = transform(person_img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = gender_model(person_img)
            _, predicted = torch.max(outputs, 1)
        # 0 for male, 1 for female
        return 'Male' if predicted.item() == 0 else 'Female'
    except Exception as e:
        print(f"Error in gender classification: {e}")
        return "person"

def run_camera():
    """
    Open the camera, run YOLOv8 for person detection, and classify gender using EfficientNetV2.
    """
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform person detection on the frame
        results = yolo_model(frame, conf=0.5)  # Adjust confidence threshold if needed

        # Loop through detection results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if int(box.cls[0]) == 0:  # Class 0 corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Crop the person from the frame for gender classification
                    person_img = frame[y1:y2, x1:x2]

                    # Ensure valid crop
                    if person_img.size == 0:
                        continue

                    # Convert to RGB (EfficientNet expects RGB format)
                    person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

                    # Classify gender
                    gender = classify_gender(person_img_rgb)

                    # Draw bounding box and label
                    label = f'{gender}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the frame with detections
        cv2.imshow('Person Detection and Gender Classification', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()
