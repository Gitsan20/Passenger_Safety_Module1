import cv2
import requests
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import threading
import time

# URL of the ESP32-CAM stream
esp32_cam_url = 'http://172.20.10.2/cam-mid.jpg'  

detector = FaceMeshDetector(maxFaces=1)

# Define eye landmarks
idList = [130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 28, 27, 29, 30, 247, 56]

# Define mouth landmarks for the inner boundary of the lips
top_mouth_landmarks = [184, 183, 74, 42, 73, 41, 72, 38, 11, 12, 302, 268, 303, 271, 364, 272, 407, 408]
bottom_mouth_landmarks = [96, 77, 89, 90, 179, 180, 86, 85, 15, 16, 316, 315, 403, 404, 319, 320, 325, 307]

blinkCounter = 0
yawn_duration = 0
color = (255, 0, 255)
yawn_start_time = None
yawn_end_time = None
total_yawn_duration = 0
counter = 0

# Adjust waitKey value for better performance
wait_key_delay = 1

# Check if OpenCL is available
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL is enabled.")
else:
    print("OpenCL is not available on this system.")

def is_yawning(face_landmarks):
    # Calculate the distance between top and bottom landmarks of the mouth
    top_mouth_points = [face_landmarks[idx] for idx in top_mouth_landmarks]
    bottom_mouth_points = [face_landmarks[idx] for idx in bottom_mouth_landmarks]
    top_mouth_center = np.mean(top_mouth_points, axis=0)
    bottom_mouth_center = np.mean(bottom_mouth_points, axis=0)
    mouth_height = np.linalg.norm(np.array(bottom_mouth_center) - np.array(top_mouth_center))

    # Check if the mouth is open wide enough (adjust threshold as needed)
    return mouth_height > 15

def process_frame():
    global blinkCounter, counter, color, yawn_duration, yawn_start_time, yawn_end_time, total_yawn_duration
    while True:
        # Fetch frame from ESP32-CAM
        response = requests.get(esp32_cam_url)
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]

            # Blink detection
            for id in idList:
                cv2.circle(img, face[id], 3, color, cv2.FILLED)

            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lengthVer, _ = detector.findDistance(leftUp, leftDown)
            lengthHor, _ = detector.findDistance(leftLeft, leftRight)

            cv2.line(img, leftUp, leftDown, (0, 200, 0), 1)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 1)

            ratio = int((lengthVer / lengthHor) * 100)

            if ratio < 30 and counter == 0:
                blinkCounter += 1
                color = (0, 200, 0)
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (255, 0, 255)

            # Yawn detection
            mouth_open = is_yawning(face)
            if mouth_open:
                if yawn_start_time is None:
                    yawn_start_time = time.time()  # Start the timer only when the mouth opens
            else:
                if yawn_start_time is not None:  # Mouth closed, stop the timer
                    yawn_end_time = time.time()
                    yawn_duration = yawn_end_time - yawn_start_time
                    if yawn_duration > 1:  # Adjust threshold as needed
                        total_yawn_duration += yawn_duration
                    yawn_start_time = None

            # Draw circles over the mouth landmarks
            for id in top_mouth_landmarks:
                cv2.circle(img, face[id], 3, color, cv2.FILLED)
            for id in bottom_mouth_landmarks:
                cv2.circle(img, face[id], 3, color, cv2.FILLED)

            # Check for drowsiness
            if total_yawn_duration > 10 and blinkCounter < 7:
                cv2.putText(img, "Driver is Sleepy", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif blinkCounter > 20:
                cv2.putText(img, "Alert! Stressful Driving!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            elif counter > 10 and total_yawn_duration == 0:
                cv2.putText(img, "Alert! Driver is Sleeping !", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Adjust font size and style for the window
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, f'Blink Count: {blinkCounter}', (10, 60),
                        font, font_scale, color, 1, cv2.LINE_AA)
            cv2.putText(img, f'Yawn Duration: {total_yawn_duration:.2f} seconds', (10, 90),
                        font, font_scale, color, 1, cv2.LINE_AA)

        cv2.imshow("Image", img)
        # Adjust waitKey value
        if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'):
            break

# Create and start the processing thread
thread = threading.Thread(target=process_frame)
thread.start()

# Main thread continues to wait for 'q' to quit
thread.join()

cv2.destroyAllWindows()
