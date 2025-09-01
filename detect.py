import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\Beyza\g_teknik\liveness_detect\model\best6.pt")

def detect_fake_real_realtime():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 for default camera, change if you have multiple cameras

    while True:
        # Read frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Process results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                label = "handsup" if cls == 1 else "handsonm"
                color = (0, 255, 0) if label == "handsup" else (0, 0, 255)

                # Draw bounding box and label
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the result
        cv2.imshow("Fake/Real Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time detection
detect_fake_real_realtime()