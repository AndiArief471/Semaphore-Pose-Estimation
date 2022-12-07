import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
import pyautogui

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils


def detectPose(image, pose, display=True):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:

            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    if display:

        plt.figure(figsize=[22, 22])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    else:

        return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:

        angle += 360

    return angle

def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'

    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    if left_elbow_angle > 165 and left_elbow_angle < 189:

        if right_shoulder_angle > 0 and right_shoulder_angle < 15:

            if left_shoulder_angle > 15 and left_shoulder_angle < 45:

                label = "A"

                #pyautogui.press("a")

                #time.sleep(1)

            if left_shoulder_angle > 80 and left_shoulder_angle < 110:

                label = "B"

                #pyautogui.press("b")

                #time.sleep(1)

            if left_shoulder_angle > 130 and left_shoulder_angle < 160:

                label = "C"

                #pyautogui.press("c")

                #time.sleep(1)

            if left_shoulder_angle > 165 and left_shoulder_angle < 195:

                label = "D"

                #pyautogui.press("d")

                #time.sleep(1)

        if right_shoulder_angle > 330 and right_shoulder_angle < 360:

            if left_shoulder_angle > 80 and left_shoulder_angle < 110:

                label = "H"

                #pyautogui.press("h")

                #time.sleep(1)

            if left_shoulder_angle > 120 and left_shoulder_angle < 150 and right_elbow_angle > 120 and right_elbow_angle < 135:

                label = "I"

                #pyautogui.press("i")

                #time.sleep(1)

    if right_shoulder_angle > 120 and right_shoulder_angle < 150:

        if left_shoulder_angle > 300 and left_shoulder_angle < 330:

            label = "W"

            #pyautogui.press("w")

            #time.sleep(1)

        if left_shoulder_angle > 340 and left_shoulder_angle < 360:

            label = "X"

            #pyautogui.press("x")

            #time.sleep(1)

    if right_elbow_angle > 165 and right_elbow_angle < 195:

        if left_shoulder_angle > 0 and left_shoulder_angle < 15:

            if right_shoulder_angle > 130 and right_shoulder_angle < 160:

                label = "E"

                #pyautogui.press("e")

                #time.sleep(1)

            if right_shoulder_angle > 80 and right_shoulder_angle < 110:

                label = "F"

                #pyautogui.press("f")

                #time.sleep(1)

            if right_shoulder_angle > 15 and right_shoulder_angle < 45:

                label = "G"

                #pyautogui.press("g")

                #time.sleep(1)

        if left_shoulder_angle > 165 and left_shoulder_angle < 195:

            if right_shoulder_angle > 80 and right_shoulder_angle < 110:

                label = "J"

                #pyautogui.press("j")

                #time.sleep(1)

            if right_shoulder_angle > 15 and right_shoulder_angle < 45:

                label = "V"

                #pyautogui.press("v")

                #time.sleep(1)

        if left_shoulder_angle > 25 and left_shoulder_angle < 45:

            if right_shoulder_angle > 165 and right_shoulder_angle < 195:

                label = "K"

                #pyautogui.press("k")

                #time.sleep(1)

            if right_shoulder_angle > 130 and right_shoulder_angle < 160:

                label = "L"

                #pyautogui.press("l")

                #time.sleep(1)

            if right_shoulder_angle > 80 and right_shoulder_angle < 110:

                label = "M"

                #pyautogui.press("m")

                #time.sleep(1)

            if right_shoulder_angle > 15 and right_shoulder_angle < 45:

                label = "N"

                #pyautogui.press("n")

                #time.sleep(1)

        if left_shoulder_angle > 130 and left_shoulder_angle < 150:

            if right_shoulder_angle > 165 and right_shoulder_angle < 195:

                label = "T"

                #pyautogui.press("t")

                #time.sleep(1)

            if right_shoulder_angle > 130 and right_shoulder_angle < 160:

                label = "U"

                #pyautogui.press("u")

                #time.sleep(1)

            if right_shoulder_angle > 80 and right_shoulder_angle < 110:

                label = "Y"

                #pyautogui.press("y")

                #time.sleep(1)

        if left_shoulder_angle > 80 and left_shoulder_angle < 110:

            if right_shoulder_angle > 165 and right_shoulder_angle < 195:

                label = "P"

                #pyautogui.press("p")

                #time.sleep(1)

            if right_shoulder_angle > 130 and right_shoulder_angle < 160:

                label = "Q"

                #pyautogui.press("q")

                #time.sleep(1)

            if right_shoulder_angle > 80 and right_shoulder_angle < 110:

                label = "R"

                #pyautogui.press("r")

                #time.sleep(1)

            if right_shoulder_angle > 15 and right_shoulder_angle < 45:

                label = "S"

                #pyautogui.press("s")

                #time.sleep(1)



        if left_elbow_angle > 190 and left_elbow_angle < 240:


            if left_shoulder_angle > 340 and left_shoulder_angle < 360 or left_shoulder_angle > 0 and left_shoulder_angle < 20:

                if right_shoulder_angle > 70 and right_shoulder_angle < 105:

                    label = "Z"

                    #pyautogui.press("z")

                    #time.sleep(1)

    if right_elbow_angle > 115 and right_elbow_angle < 135:

        if right_shoulder_angle > 320 and right_shoulder_angle < 340 and left_shoulder_angle > 120 and left_shoulder_angle < 140:

            label = "O"

            #pyautogui.press("o")

            #time.sleep(1)

    if right_elbow_angle > 140 and right_elbow_angle < 160 and left_elbow_angle > 200 and left_elbow_angle < 250:

        label = "Space"

        #pyautogui.press("space")

        #time.sleep(1)

    if label != 'Unknown Pose':

        color = (0, 255, 0)

    cv2.putText(output_image, label, (30, 60), cv2.FONT_HERSHEY_PLAIN, 5, color, 5)

    if display:

        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    else:

        return output_image, label

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

while camera_video.isOpened():

    ok, frame = camera_video.read()

    if not ok:

        continue

    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape

    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    frame, landmarks = detectPose(frame, pose_video, display=False)

    if landmarks:

        frame, _ = classifyPose(landmarks, frame, display=False)

    cv2.imshow('Pose Classification', frame)

    k = cv2.waitKey(1) & 0xFF

    if (k == 27):

        break

camera_video.release()
cv2.destroyAllWindows()