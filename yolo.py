from ultralytics import YOLO

def main():
    model = YOLO(r"yolov8n.pt")
    model.train(data=r"yaml\data.yaml", epochs=10)

if __name__ == '__main__':
    main()
