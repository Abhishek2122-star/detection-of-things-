import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import mediapipe as mp
import torch.nn.functional as F

# Load YOLOv8 model for person detection
yolo_model = YOLO('yolov8n.pt')  # Lightweight YOLOv8 model

# Load EfficientNetV2 model for gender classification
gender_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
gender_model.eval()

# Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Define transform for image preprocessing for EfficientNetV2
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((224, 224)),  # Resize to EfficientNet input size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

def classify_gender(person_img):
    """
    Classify the gender of the cropped person image using EfficientNetV2.
    """
    try:
        # Preprocess the image
        person_img = transform(person_img).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = gender_model(person_img)
            probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            print(f"Probabilities: {probabilities.tolist()}")  # Log probabilities
            
            confidence, predicted = torch.max(probabilities, 1)
            print(f"Predicted class: {predicted.item()}, Confidence: {confidence.item()}")

        # Use a confidence threshold to avoid ambiguous predictions
        if confidence.item() < 0.5:
            return "female"

        return 'female' if predicted.item() == 0 else 'male'
    except Exception as e:
        print(f"Error in gender classification: {e}")
        return "male"

def detect_and_classify(frame):
    """
    Detect persons using YOLO, refine detections to faces using Mediapipe, and classify gender.
    """
    results = yolo_model(frame, conf=0.5)  # Adjust confidence threshold as needed

    for result in results:
        detections = result.boxes.data.cpu()  # Bounding boxes
        for box in detections:
            cls = int(box[-1].item())  # Class ID
            if cls == 0:  # Class 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates

                # Crop detected person
                person_img = frame[y1:y2, x1:x2]

                # Ensure valid crop
                if person_img.size == 0:
                    continue

                # Use Mediapipe to detect face within the cropped person
                rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_img)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = person_img.shape
                        fx, fy, fw, fh = (
                            int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                            int(bboxC.width * iw), int(bboxC.height * ih)
                        )

                        # Crop the face region
                        face_img = person_img[fy:fy + fh, fx:fx + fw]

                        # Validate face crop
                        if face_img.size == 0:
                            continue

                        # Debug: Visualize cropped face
                        cv2.imshow("Cropped Face", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

                        # Classify gender
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        gender = classify_gender(face_img_rgb)

                        # Draw bounding box and label
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

def test_on_images(image_folder):
    """
    Test gender classification on static images.
    """
    import os
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        detect_and_classify(img)  # Process each image
        cv2.imshow("Test Image", img)
        cv2.waitKey(0)  # Pause for inspection
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # To run live camera feed
    run_camera()

    # Uncomment the following to test on images
    # test_on_images('path_to_test_images')
