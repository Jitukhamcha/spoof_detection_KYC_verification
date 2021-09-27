import numpy as np 
import mediapipe_face
import predict_liveness

from scipy.spatial import distance

def detect_pose(keypoints):
    hpose = None
    rEye =  tuple(keypoints[0])
    lEye = tuple(keypoints[1])

    nose = np.array(keypoints[2])
    mouth = np.array(keypoints[3])
    mid = (nose + mouth)/2

    if (lEye[0]!= rEye[0]):
        slope = (lEye[1]-rEye[1])/(lEye[0]-rEye[0])
        y_incpt= rEye[1]-(slope*rEye[0])

        y = slope*mid[0] + y_incpt

        if rEye[0] < mid[0] < lEye[0]:
            k1 = distance.euclidean(rEye, (mid[0],int(y)))
            k2 = distance.euclidean((mid[0],int(y)), lEye)

            k3 = distance.euclidean((mid[0], nose[1]), (mid[0], mouth[1]))
            k4 = distance.euclidean((mid[0],nose[1]), (mid[0],int(y)))
            print(k2/k1, k1/k2)
            if k2 / k1 <= 0.5:
                hpose = "right"
            elif k1 / k2 <= 0.5:
                hpose = "left"

    return hpose


def frame_grabber_keypoints(frame):
    faces, keypoints = mediapipe_face.detect_face_landmark(frame)
    return faces, keypoints

def passive_liveness(frame, face_box):
    prediction = predict_liveness.predict(frame,face_box)
    return prediction