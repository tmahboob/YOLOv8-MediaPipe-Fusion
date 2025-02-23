import os
from ultralytics import YOLO
import cv2
import glob
import json
import numpy as np

# Define paths
model_path = 'D:/ML project/python/model/runs/pose/train/weights/last.pt'
test_images_path = './test/images/'
test_labels_path = './test/labels/'

# Load the YOLO model
model = YOLO(model_path)


# Function to read annotations
def read_annotations(label_path):
    with open(label_path, 'r') as file:
        annotations = json.load(file)
    return annotations


# Function to calculate metrics
def calculate_metrics(true_keypoints, predicted_keypoints):
    true_positives = np.sum((true_keypoints == 1) & (predicted_keypoints == 1))
    false_positives = np.sum((true_keypoints == 0) & (predicted_keypoints == 1))
    false_negatives = np.sum((true_keypoints == 1) & (predicted_keypoints == 0))

    accuracy = np.mean(true_keypoints == predicted_keypoints)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision & recall) if (precision & recall) > 0 else 0

    return accuracy, precision, recall, f1


# Lists to store true and predicted keypoints
true_keypoints = []
predicted_keypoints = []

# Process each image in the test set
for image_file in glob.glob(test_images_path + '*.jpeg'):
    img = cv2.imread(image_file)

    # Run model on the image
    results = model(image_file)[0]

    # Get the corresponding label file
    label_file = os.path.join(test_labels_path, os.path.basename(image_file).replace('.jpeg', '.json'))
    annotations = read_annotations(label_file)

    for result, annotation in zip(results, annotations):
        pred_keypoints = result.keypoints.xy.tolist()
        true_keypoints.extend(annotation['keypoints'])  # Assuming 'keypoints' is a list in annotations
        predicted_keypoints.extend(pred_keypoints)

        # Draw keypoints on the image for visualization (optional)
        for keypoint in pred_keypoints:
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    # Display the image with keypoints (optional)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Convert keypoints lists to arrays
true_keypoints = np.array(true_keypoints)
predicted_keypoints = np.array(predicted_keypoints)

# Calculate accuracy metrics
accuracy, precision, recall, f1 = calculate_metrics(true_keypoints, predicted_keypoints)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
