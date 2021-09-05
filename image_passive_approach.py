import numpy as np 
import cv2
import mediapipe_face
import predict_liveness

def liveness_on_image(image):
    faces,_ = mediapipe_face.detect_face_landmark(image)
    
    if len(faces) != 0:
        faces = faces.astype("int")
        for face in faces:
            face_box = face.tolist()
            prediction = predict_liveness.predict(image,face_box)

            # draw result of prediction
            print(prediction)
            label = np.argmax(prediction)
            value = round(prediction[0][label]/2, 2)

            if label == 1 and value >=0.8:
                print(f"Real Face. Score = {value}")
                result_text = "Real {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                print(f"Fake Face. Score: {value}")
                result_text = "Fake {:.2f}".format(value)
                color = (0, 0, 255)

            cv2.rectangle(
                image,
                (face_box[0], face_box[1]),
                (face_box[2], face_box[3]),
                color, 2)
            cv2.putText(
                image,
                result_text,
                (face_box[0], face_box[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 1, color,1)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey()

if __name__ == '__main__':
    image_path = "test/2021-09-05-121453.jpg"
    image = cv2.imread(image_path)
    liveness_on_image(image)