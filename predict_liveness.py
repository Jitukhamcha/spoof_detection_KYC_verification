import os
import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

model_dir = "./resources/anti_spoof_models"

model_test = AntiSpoofPredict(device_id=0)

models = ['4_0_0_80x80_MiniFASNetV1SE.pth', '2.7_80x80_MiniFASNetV2.pth']
modelv1 = model_test._load_model(os.path.join(model_dir, models[0]))
modelv2 = model_test._load_model(os.path.join(model_dir, models[1]))

def predict(image, face_box, model_dir = model_dir):
    image_cropper = CropImage()

    prediction = np.zeros((1, 3))
    test_speed = 0

    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": face_box,
            "scale": True,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()

        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    return prediction