import mediapipe as mp
import cv2
import numpy as np 
import os 
import time 


mp_holistic=mp.solutions.holistic # mediapipe solution

OUTPUT_PATH ="C://Users//Beyza//g_teknik//liveness_detect//dataset//images"


CLASSES = 0 # change it please 
FRAME_COUNT = 0

def normalized_landmarks(landmarks):
    if not landmarks:
        return []  # Return an empty list if no landmarks are detected
    
    x_coords = [landmark[0] for landmark in landmarks]
    y_coords = [landmark[1] for landmark in landmarks]
    
    x_max = max(x_coords)
    x_min = min(x_coords)
    y_max = max(y_coords)
    y_min = min(y_coords)
    
    normalized_landmarks = []
    for landmark in landmarks:
        x_norm = (landmark[0] - x_min) / (x_max - x_min) if x_max != x_min else 0
        y_norm = (landmark[1] - y_min) / (y_max - y_min) if y_max != y_min else 0
        z_norm = landmark[2]
        visibility = landmark[3]
        normalized_landmarks.append([x_norm, y_norm, z_norm, visibility])
    return normalized_landmarks


def save_img_txt(database_path, landmarks, FRAME, holistic):

    _, image = img_trans(FRAME, holistic)
    SAVE_INTERVAL = 1
    last_save_time = 0
    img_id = 0
    
    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERVAL:
        last_save_time = current_time
        img_id += 1
        timer = str(int(current_time * 1000000))  # Use microseconds for unique filename
        file_path = f"{database_path}//{timer}_{img_id}"
        
        cv2.imwrite(f"{file_path}.jpg", image)
        
        with open(f"{file_path}.txt", 'w') as f:
            for i, landmark in enumerate(landmarks):
                x, y = landmark[0], landmark[1]
                w, h = 0.01, 0.01
                f.write(f'{CLASSES}, {x}, {y}, {w}, {h}\n')
        
        print(f"Saved image and landmarks: {file_path}")


def img_trans(FRAME, holistic):
    # Recolor Feed
    image = cv2.cvtColor(FRAME, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make Detections
    results = holistic.process(image)
    
    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return results, image



def landmark_detect(FRAME, holistic):


    results, _ = img_trans(FRAME, holistic)
    landmarks = []
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return landmarks

CAP = cv2.VideoCapture(0)
camWidth, camHeight = 640, 480
CAP.set(3, camWidth)
CAP.set(4, camHeight)

# Initiate holistic model
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while CAP.isOpened():
        RET, FRAME = CAP.read()
        if not RET:
            break
        
        landmarks = landmark_detect(FRAME, holistic)
        
        print(f"Number of landmarks detected: {len(landmarks)}")
        normalized = normalized_landmarks(landmarks)
        save_img_txt(OUTPUT_PATH, normalized, FRAME, holistic)
        
        _, image = img_trans(FRAME, holistic)

        cv2.imshow('Raw Webcam Feed', image)
        
        if cv2.waitKey(10) == 27:
            break

CAP.release()
cv2.destroyAllWindows()

#?????????? BU KISIMDA NASIL HOLİSTİC KULLANILDIĞINDAN NASIL FONKSİYON TANIMLAYACAĞIMI PEK ANLAYAMADIM ??????????