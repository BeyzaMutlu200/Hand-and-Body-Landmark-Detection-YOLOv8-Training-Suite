#BUNDAN SONRA YAPMAN GEREKEN İŞLEMLERDEN BİRİ BU RESİMLERİ ÇEKİP TXT DOSYASINA ETİKETLENMİŞ VE KOOODİNATLARIN BİLGİSİNİ GİRMEK. 
import time 
import cv2
import mediapipe as mp

DS_PATH =r"C:\Users\Beyza\g_teknik\liveness_detect\dataset\four"
# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)

# find the center of frame 
def center_frame(x_min,y_min,x_max,y_max):
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return center_x , center_y

# class id 0 = piece 
#

# normalize the frame coords 
def normalize_coords(x_min,y_min,x_max,y_max,frame,list_text):
    CLASS_ID = 2
    FLOATING_POINT = 6
    width , height  = frame.shape[1] , frame.shape[0]
    center_x , center_y = center_frame(x_min,y_min,x_max,y_max) 
    
    x_norm = round((center_x / width) , FLOATING_POINT)
    y_norm = round((center_y / height) , FLOATING_POINT)
 
    # to ensure that the x_norm and y-norm coordinates are between 0-1 
    x_norm = min(x_norm , 1)
    y_norm = min(y_norm , 1)

    w_norm = round((x_max-x_min) / width , FLOATING_POINT)
    h_norm = round((y_max - y_min) / height , FLOATING_POINT)

    #print(x_norm,y_norm,w_norm,h_norm)
    list_text.append(f"{CLASS_ID} {x_norm} {y_norm} {w_norm} {h_norm}")

# save the images and txt files 
def save_img_txt(database_path,frame,list_text):
    SAVE_INTERNAL = 0 
    last_save_time = 0 
    img_id = 0 
    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERNAL and list_text:
        last_save_time = current_time
        img_id += 1
        timer = str(int(current_time * 1000000))
        file_path = f"{database_path}//{timer}_{img_id}"

        cv2.imwrite(f"{file_path}.jpg", frame)

        with open(f"{file_path}.txt",'w') as txt_file:
            txt_file.write("\n".join(list_text))


# detect hands , normalize and find the center of the detected hands
def detect_hands(frame):
       # Flip the image(frame)
    FRAME = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)
    list_text = []

    # If hands are present in the image(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the hand bounding box coordinates
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Draw rectangle around the detected hand
            #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #center_x, center_y = center_frame(x_min, y_min, x_max, y_max)
            #cv2.circle(frame, (center_x, center_y), 7, (255, 255, 255), -1)
            normalize_coords(x_min, y_min, x_max, y_max,frame,list_text)
            save_img_txt(DS_PATH , frame , list_text)

# Initialize MediaPipe Drawing
mpDraw = mp.solutions.drawing_utils

# Start capturing video from webcam
CAP = cv2.VideoCapture(0)
    
while True:
    # Read video frame by frame
    RET, FRAME = CAP.read()

    camWidth, camHeight = 640, 480
    CAP.set(3,camWidth)
    CAP.set(4,camHeight)

    if not RET:
        break

    detect_hands(FRAME)
  
    # Display the video and break the loop if 'q' is pressed
    cv2.imshow('Image', FRAME)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()

#  bu  kodda düzenlemeye gidilmesi kısım şu fkadraj dışına çıkııldığı9nda  o resmi kaydetmemesi kısmı 
#  blurrring fazla olan resimlerin ksydedilmesinin önlenmeside önem arz ediyor 
# w ve h korridnatlarında normnalize edilmesi gerek 
