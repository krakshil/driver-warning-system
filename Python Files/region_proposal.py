import numpy as np
from shapely.geometry import Polygon
import cv2
from matplotlib import pyplot as plt
import crop
import show

class Regions:
    
    def __init__(self, image, boxes, cropper=crop.CropperWithoutPad()):
        self._image = image
        self._boxes = boxes
        self._cropper = cropper
    
    def get_boxes(self):
        return self._boxes
    
    def get_patches(self, dst_size=None):
        patches = []    
        for bb in self._boxes:
            patch = self._crop(bb)
            if dst_size:
                desired_ysize = dst_size[0]
                desired_xsize = dst_size[1]
                patch = cv2.resize(patch, (desired_xsize, desired_ysize), interpolation=cv2.INTER_AREA)
            patches.append(patch)
            
        if dst_size:
            return np.array(patches)
        else:
            return patches
    
    def _crop(self, box):
        patch = self._cropper.crop(self._image, box)
        return patch

class _RegionProposer:
    
    def __init__(self):
        pass
    
    def detect(self, img):
        pass

    def _to_gray(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray


class MserRegionProposer(_RegionProposer):
    
    def detect(self, img):
        gray = self._to_gray(img)
        mser = cv2.MSER_create(_delta = 1)
        msers, _ = mser.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in msers]
        bounding_boxes = self._get_boxes(hulls)
        regions = Regions(img, bounding_boxes)
        return regions
    
    def _get_boxes(self, regions):
        bbs = []
        for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
            bbs.append((y, y+h, x, x+w))
        return np.array(bbs)



class OverlapCalculator:
    
    def __init__(self):
        pass
    
    def calc_ious_per_truth(self, boxes, true_boxes):
        return self._calc(boxes, true_boxes)
    
    def calc_maximun_ious(self, boxes, true_boxes):
        ious_for_each_gt = self._calc(boxes, true_boxes)
        ious = np.max(ious_for_each_gt, axis=0)
        return ious
    
    def _calc(self, boxes, true_boxes):
        ious_for_each_gt = []
        for truth_box in true_boxes:
            ious=[]
            ty1=truth_box[0]
            ty2=truth_box[1]
            tx1=truth_box[2]
            tx2=truth_box[3]
            truebox=[[tx1,ty1],[tx2,ty1],[tx2,ty2],[tx1,ty2]]
            tpoly = Polygon(truebox)
            for box in boxes:
                by1=box[0]
                by2=box[1]
                bx1=box[2]
                bx2=box[3]
                bbox=[[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]]
                boxpoly = Polygon(bbox)
                iou = tpoly.intersection(boxpoly).area / tpoly.union(boxpoly).area
                ious.append(iou)
            ious_for_each_gt.append(ious)
        
        # (n_truth, n_boxes)
        ious_for_each_gt = np.array(ious_for_each_gt)
        return ious_for_each_gt