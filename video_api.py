import cv2
import mmpose
import torch
import os
import json
import sys
import time
from mmpose.apis import MMPoseInferencer
import matplotlib.pyplot as plt

args = sys.argv[1:]
video_path = args[0]

frames_json = {}

inferencer = MMPoseInferencer(
    pose2d='face',
    device='cuda:0')


def draw_predictions(frame, results, frame_num):
    # Iterate through the predictions and draw them on the frame
    frames_json[frame_num] = results['predictions'][0]

    for person in results['predictions'][0]: #predictions[0] is the list of dictionaries
        # Extract the necessary information from the prediction
        # (e.g., keypoints coordinates, bounding boxes)
        keypoints = person['keypoints']
        bbox = person['bbox'][0]

        # Draw keypoints on the frame
        for keypoint in keypoints:
            x, y = keypoint[:2]
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Draw bounding box on the frame
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        try:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        except:
            # print the error
            print("Error in drawing bounding box")


    return frame

def vid_inference(video_path, vid_name):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    height = width = None

    directory = f"pred/{vid_name}"
    os.makedirs(directory, exist_ok=True)


    # Read the video frame by frame
    while True:
        # Read a single frame from the video
        ret, frame = video.read()

        if height is None:
            height, width, _ = frame.shape

        # If the frame was not successfully read, break the loop
        if not ret:
            break

        # Process the frame (e.g., apply some operations)
        result_generator = inferencer(frame, show=True, radius = 1)
        result = dict(next(result_generator))

        # Draw predictions on the frame
        frame_with_predictions = draw_predictions(frame, result)

        # Save the frame with predictions as an image
        cv2.imwrite(f'pred/{vid_name}/frame_{frame_count}.png', frame_with_predictions)

        frame_count += 1

    # Release the video object
    video.release()
    return (frame_count, height, width)

def recombine_frames(vid_name, frame_count, height, width):
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Make 
    directory = f"pred/"
    os.makedirs(directory, exist_ok=True)

    output_path = 'pred/'+ vid_name+ '.mp4'
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Iterate through the frames
    for i in range(frame_count):
        # Load the frame image
        frame_image = cv2.imread('/pred/' + vid_name + '/frame_' + str(i) + '.png')

        # Write the frame image to the video
        out.write(frame_image)

    # Release the VideoWriter object
    out.release()

    # Specify the path to the file
    file_path = 'pred/' + vid_name +'.mp4'

    time.sleep(2)

    # Check if the file exists
    if os.path.exists(file_path):
        print("File was created successfully")
    else:
        print("File was unable to be created")


def video_api(vid_path, vid_name):
    """Vid must be """
    frame_count, height, width = vid_inference(vid_path, vid_name)
    recombine_frames(vid_name, frame_count, height, width)

vid_name = os.path.basename(video_path)
video_api(video_path, vid_name)

with open(f'/frames/pred/{vid_name}/frames.json', 'w') as json_file:
    json.dump(frames_json, json_file)

