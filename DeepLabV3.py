import torch
from PIL import Image
import numpy as np
import os
import cv2
import torchvision
from torchvision import transforms

import argparse


def get_filename_and_extension_from_path(path):
    slash_pos = path.rfind("/")
    dot_pos = path.rfind(".")
    file_name = path[slash_pos+1:dot_pos]
    ext_name = path[dot_pos:]
    return file_name, ext_name


def deeplabv3_segment(model, img, img_path=None, destination_path='', individual_class_masks=False, normalize=False):

    model.eval()

    if img is not None:
        input_image = img
    else:
        input_image = Image.open(img_path)
        input_image = input_image.convert("RGB")
    preprocess_list = [transforms.ToTensor()]
    if normalize:
        preprocess_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    preprocess = transforms.Compose(preprocess_list)

    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    if output.shape[0] == 1:
        output_predictions = output[0]
        output_predictions[output_predictions >= 0.1] = 1
        output_predictions[output_predictions < 0.1] = 0
        np_out = output_predictions.data.cpu().numpy() * 255
        if destination_path:
            if not os.path.exists(destination_path):
                os.mkdir(destination_path)
            img_name, ext_name = get_filename_and_extension_from_path(img_path)
            cv2.imwrite(destination_path+"by_deeplabv3_"+img_name+ext_name, np_out)
        return np_out
    else:
        output_predictions = output.argmax(0)

        np_r = output_predictions.data.cpu().numpy() 

        if individual_class_masks:
            ans = []
            for unique_value in np.unique(np_r)[np.unique(np_r) != 0]:
                mask = np_r == unique_value
                ans.append(mask.astype("uint8") * 255)
        else:
            ans = np_r

        if destination_path:
            if not os.path.exists(destination_path):
                os.mkdir(destination_path)
            img_name, ext_name = get_filename_and_extension_from_path(img_path)
            if individual_class_masks:
                if not os.path.exists(destination_path+"by_deeplabv3_"+img_name):
                    os.mkdir(destination_path+"by_deeplabv3_"+img_name)
                for i, mask in enumerate(ans):
                    cv2.imwrite(destination_path+"by_deeplabv3_"+img_name+f"/{i}"+ext_name, mask)
            else:
                cv2.imwrite(destination_path+"by_deeplabv3_"+img_name+ext_name, ans)
            
        return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--source", type=str, required=True, help="path of the image")
    parser.add_argument('-i', "--individual_class_masks", action="store_true", help="flag indicating that output should be \
                                                                            directory containing individual class masks")
    parser.add_argument('-n', "--normalize", action="store_true",
                        help="flag indicating if we want to normalize the input")
    parser.add_argument('-d', "--destination", type=str, default='', help="path of the destination directory. \
                                                                Not storing outputs if destination path is not provided")
    parser.add_argument('-m', "--model_path", type=str, default='', help="path of the model. \
                                                                        By default it is DeepLabV3 with resnet101 backbone")
    args = parser.parse_args()

    if not args.model_path:
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    else:
        model = torch.load(args.model_path)
    deeplabv3_segment(model, None, args.source, args.destination, args.individual_class_masks, args.normalize)
