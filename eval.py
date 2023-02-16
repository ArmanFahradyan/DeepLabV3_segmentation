import torch
import torchvision
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm
from DeepLabV3 import deeplabv3_segment


parser = argparse.ArgumentParser()
parser.add_argument('-s', "--source_path", type=str, required=True, help="path of the data directory where are located \
                                                                         'images'(required) and 'gt'(optional) directories")
parser.add_argument('-d', "--destination_path", type=str, required=True, help="path of the destination directory where should be the output images")
parser.add_argument('-m', "--model_path", type=str, default='', help="path of the model. By default it is DeepLabV3 with resnet101 backbone")
args = parser.parse_args()


def eval_by_deeplabv3(source_path, destination_path, model_path):

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    if not model_path:

        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    else:
        model = torch.load(model_path)


    img_names = [file_name for file_name in os.listdir(source_path+"images") if not file_name.startswith(".")]
    img_names.sort()

    for name in tqdm(img_names):
        pred = deeplabv3_segment(model=model, img_path=source_path+"images/"+name, destination_path='', individual_obj_masks=False)
        if os.path.exists(source_path+"gt"):
            if len(pred.shape) == 2:
                gt_mask = cv2.imread(source_path+"gt/"+name, cv2.IMREAD_GRAYSCALE)
            else:
                gt_mask = cv2.imread(source_path+"gt/"+name)
            
            assert gt_mask.shape == pred.shape, "shape mismatch between groundtruth mask and prediction"
            seperator_shape = (pred.shape[0], 7, pred.shape[2]) if len(pred.shape) == 3 else (pred.shape[0], 7)
            seperator = np.ones(seperator_shape)*123
            pred = np.concatenate([pred, seperator, gt_mask], axis=1)
        cv2.imwrite(destination_path+name[:name.rfind(".")]+".png", pred)


if __name__ == "__main__":
    eval_by_deeplabv3(args.source_path, args.destination_path, args.model_path)
