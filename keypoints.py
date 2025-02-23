import cv2

# Load the pre-trained OpenPose model
net = cv2.dnn.readNetFromTensorflow("path_to_openpose_model.pb", "path_to_openpose_model.pbtxt")

# Specify the number of expected points in the output
n_points = 33

# Video file path
video_path = 'input_video.mp4'
output_file = 'pose_landmarks.txt'

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_index = 0

with open(output_file, 'w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Prepare the input image for the model
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)

        # Set the input to the model
        net.setInput(blob)

        # Run forward pass
        output = net.forward()

        # Extract the detected keypoints
        points = []
        for i in range(n_points):
            # Get the confidence map for the keypoint
            prob_map = output[0, i, :, :]

            # Find global maxima of the probability map
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            # Scale the point to the size of the original image
            x = int((width * point[0]) / output.shape[3])
            y = int((height * point[1]) / output.shape[2])

            points.append((x, y))

        # Write the landmarks to the file
        f.write(f"Frame {frame_index}:\n")
        for idx, (x, y) in enumerate(points):
            f.write(f"Landmark {idx}: ({x}, {y})\n")
        f.write("\n")

        frame_index += 1

# Release the video capture object
cap.release()

print(f"Pose landmarks have been written to {output_file}")
