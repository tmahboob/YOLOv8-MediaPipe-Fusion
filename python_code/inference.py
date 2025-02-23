from ultralytics import YOLO
import cv2

# Define paths
model_path = 'D:/ML project/python/model/runs/pose/train/weights/last.pt'
image_path = './2.jpeg'
img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]

for result in results:
    keypoints = result.keypoints.xy
    for keypoint_indx, keypoint in enumerate(keypoints[0]):
        x, y = int(keypoint[0].item()), int(keypoint[1].item())
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Change color to red (BGR format)
        cv2.putText(img, str(keypoint_indx), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Change text color to red

cv2.imshow('img', img)
cv2.waitKey(0)
