import keras
import numpy as np
import cv2

def remove_none_images(images):
    true_image_indices = []
    for i in images:
        h,b,c = i.shape
        if(b==0 or h==0 or c==0):
            true_image_indices.append(False)
        else:
            true_image_indices.append(True)
    return true_image_indices

def model_ready(images):
    imgs = []
    for image in images:
        imge = cv2.resize(image,(100,100),interpolation=cv2.INTER_AREA)
        imge = cv2.cvtColor(imge,cv2.COLOR_RGB2GRAY)
        imgs.append(imge)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0],100,100,1)
    return imgs

def run(img_files,model_file):
    img_files_array = np.array(img_files)
    true_image_indices = remove_none_images(img_files_array)
    non_None_images = img_files_array[true_image_indices]
    model_ready_images = model_ready(non_None_images)
    model = keras.models.load_model(model_file)
    predictions = model.predict(model_ready_images)
    predictions_improvised = []
    for i in predictions:
        predictions_improvised.append(i[1]>0.7)
    return img_files_array[predictions_improvised]

if __name__ == "__main__":
    pass