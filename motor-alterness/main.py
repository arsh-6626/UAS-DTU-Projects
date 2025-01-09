from ultralytics import YOLO
import os
import cv2
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

model = YOLO('/home/cha0s/ViTPose/demo/best_body.pt')
cap = cv2.VideoCapture('/home/cha0s/vid1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, save=False, conf=0.25)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
