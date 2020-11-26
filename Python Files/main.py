import capture as cap
import sign_detector as signdet
import predict_speed as preds

sign_detector_model = 'models/saved_model_sign_detector.h5'

image_patch = cap.get_frames(10)
sorted_image_patch = signdet.run(image_patch,sign_detector_model)

#for i,j in enumerate(sorted_image_patch):
    #cv2.imwrite('testcase/sign'+str(i)+'.jpg',j)

#import predict_speed as ps

#from os import listdir
#import numpy as np
#import cv2

#DIR = 'testcase'
#list_of_images = listdir(DIR)

#image_patch_traffic_signs = []
#for i in list_of_images:
    #image_patch_traffic_signs.append(cv2.imread(DIR+'/'+str(i)))

#image_patch_traffic_signs = np.array(image_patch_traffic_signs[0:10])
speed = preds.recognizer(sorted_image_patch)
print(speed)