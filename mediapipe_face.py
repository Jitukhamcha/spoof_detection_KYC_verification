import time
import cv2
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def detect_face_landmark(image,min_detection_confidence=0.7):
    with mp_face_detection.FaceDetection(min_detection_confidence) as face_detection:
        h,w = image.shape[:2]
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        keypoints = []
        if results.detections:
            
            for detection in results.detections:

                # print('Nose tip:')
                # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
                # for m in mp_face_detection.FaceKeyPoint:
                #     print(m)
                
                face_box = detection.location_data.relative_bounding_box
                kp = list(detection.location_data.relative_keypoints)
                faces.append([  face_box.xmin * w, 
                                face_box.ymin * h, 
                                face_box.xmin * w + face_box.width * w, 
                                face_box.ymin * h + face_box.height * h])
                for i in range(6):
                    keypoints.append([int(kp[i].x * w), int(kp[i].y * h)])

        faces = np.array(faces).astype("int")

        # else:
        #     faces = None
        #     keypoints = None

        return faces, keypoints