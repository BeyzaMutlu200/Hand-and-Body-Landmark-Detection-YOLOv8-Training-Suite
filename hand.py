import mediapipe as mp
import cv2
import numpy as np 
import os 
import time 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

OUTPUT_PATH = "C://Users//Beyza//goruntu isleme teknikleri//liveness_detect//dataset//imghand"

CLASSES = 2 # change it please 
FRAME_COUNT = 0

def normalized_landmarks(landmarks):
    if not landmarks:
        return []
    
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    
    x_max, x_min = max(x_coords), min(x_coords)
    y_max, y_min = max(y_coords), min(y_coords)
    
    normalized_landmarks = []
    for landmark in landmarks:
        x_norm = (landmark.x - x_min) / (x_max - x_min) if x_max != x_min else 0
        y_norm = (landmark.y - y_min) / (y_max - y_min) if y_max != y_min else 0
        z_norm = landmark.z
        normalized_landmarks.append([x_norm, y_norm, z_norm])
    return normalized_landmarks

def save_img_txt(database_path, landmarks, frame):
    SAVE_INTERVAL = 1
    last_save_time = 0
    img_id = 0
    
    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERVAL:
        last_save_time = current_time
        img_id += 1
        timer = str(int(current_time * 1000000))
        file_path = f"{database_path}//{timer}_{img_id}"
        
        cv2.imwrite(f"{file_path}.jpg", frame)
        
        with open(f"{file_path}.txt", 'w') as f:
            for landmark in landmarks:
                x, y, z = landmark
                w, h = 0.01, 0.01
                f.write(f'{CLASSES} {x}  {y}  {w} {h}\n')
        
        print(f"Saved image and landmarks: {file_path}")

def process_frame(frame, hands):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return results, image

hands=mp_hands.Hands()
CAP = cv2.VideoCapture(0)
camWidth, camHeight = 640, 480
CAP.set(3, camWidth)
CAP.set(4, camHeight)


while CAP.isOpened():
    RET, FRAME = CAP.read()
    if not RET:
        break
        
    results, image = process_frame(FRAME, hands)
        
    if results.multi_hand_landmarks:
         for hand_landmarks in results.multi_hand_landmarks:
            normalized = normalized_landmarks(hand_landmarks.landmark)
            save_img_txt(OUTPUT_PATH, normalized, image)
        
    cv2.imshow('Hand Tracking', image)
        
    if cv2.waitKey(10) == 27:
        break

CAP.release()
cv2.destroyAllWindows()