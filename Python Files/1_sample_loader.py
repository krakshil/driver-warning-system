import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import extractor as extractor_
import file_io as file_io
import annotation as ann
import show as show
import region_proposal as rp

N_IMAGES = None
DIR = 'annotation/train/train'
ANNOTATION_FILE = "annotation/train/digitStruct.json"
NEG_OVERLAP_THD = 0.05
POS_OVERLAP_THD = 0.6
PATCH_SIZE = (32,32)

if __name__ == "__main__":

    # 1. file load
    files = file_io.list_files(directory=DIR, pattern="*.png", recursive_option=False, n_files_to_sample=N_IMAGES, random_order=False)
    n_files = len(files)
    n_train_files = int(n_files * 0.8)
    print(n_train_files)
    
    extractor = extractor_.Extractor(rp.MserRegionProposer(), ann.SvhnAnnotation(ANNOTATION_FILE), rp.OverlapCalculator())
    train_samples, train_labels = extractor.extract_patch(files[:n_train_files], PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)

    extractor = extractor_.Extractor(rp.MserRegionProposer(), ann.SvhnAnnotation(ANNOTATION_FILE), rp.OverlapCalculator())
    validation_samples, validation_labels = extractor.extract_patch(files[n_train_files:], PATCH_SIZE, POS_OVERLAP_THD, NEG_OVERLAP_THD)

    print(train_samples.shape, train_labels.shape)
    print(validation_samples.shape, validation_labels.shape)
    
    images_train, labels_train = file_io.dataset_refactor(train_samples,train_labels)
    images_valid, labels_valid = file_io.dataset_refactor(validation_samples,validation_labels)

    print(images_train.shape, labels_train.shape)
    print(images_valid.shape, labels_valid.shape)

    #data1 = np.random.randn(100000,32,32,3)
    #hf = h5py.File('train.h5','w')
    #hf.create_dataset('images',data=data1)
    #hf.close()

    file_io.FileHDF5().write(images_train, "train.h5", "images", "w", dtype="uint8")
    file_io.FileHDF5().write(labels_train, "train.h5", "labels", "a", dtype="int")
 
    file_io.FileHDF5().write(images_valid, "val.h5", "images", "w", dtype="uint8")
    file_io.FileHDF5().write(labels_valid, "val.h5", "labels", "a", dtype="int")
    # (457723, 32, 32, 3) (457723, 1)
    # (113430, 32, 32, 3) (113430, 1)
