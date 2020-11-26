import cv2
import numpy as np
import keras.models

import region_proposal as rp
import show as show
import detect as detector
import file_io as file_io
import preprocess as preproc
import classify as cls

detect_model = "detector_model.hdf5"
recognize_model = "recognize_model.hdf5"

mean_value_for_detector = 107.524
mean_value_for_recognizer = 112.833

model_input_shape = (32,32,1)

def recognizer(img_files):
    preproc_for_detector = preproc.GrayImgPreprocessor(mean_value_for_detector)
    preproc_for_recognizer = preproc.GrayImgPreprocessor(mean_value_for_recognizer)

    char_detector = cls.CnnClassifier(detect_model, preproc_for_detector, model_input_shape)
    char_recognizer = cls.CnnClassifier(recognize_model, preproc_for_recognizer, model_input_shape)
    
    digit_spotter = detector.DigitSpotter(char_detector, char_recognizer, rp.MserRegionProposer())
    
    predictions = []
    for img_file in img_files[0:]:
        # 2. image
        img = cv2.imread(img_file)
        
        predictions.append(digit_spotter.run(img, threshold=0.5, do_nms=True, nms_threshold=0.1,show_result=False))

    speeds = []
    for i in predictions:
        speed = 0
        for j,l in enumerate(i):
            if(l%5!=0):
                speed += l*10
            else:
                speed += l
        speeds.append(speed)
    speeds = np.array(speeds)
    predicted_speeds = np.unique(speeds,return_counts=True)
    predicted_speed_improvised = predicted_speeds[0][np.argmax(predicted_speeds[1])]
    return predicted_speed_improvised







