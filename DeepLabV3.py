import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torchvision
from torchvision import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', "--source", type=str, required=True, help="path of the image")
parser.add_argument('-i', "--individual_obj_masks", action="store_true", help="flag indicating that output should be directory containing individual object masks")
parser.add_argument('-d', "--destination", type=str, default='', help="path of the destination directory. Not storing outputs if destination path is not provided")
parser.add_argument('-m', "--model_path", type=str, default='', help="path of the model. By default it is DeepLabV3 with resnet101 backbone")
args = parser.parse_args()

if not args.model_path:

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
else:
    model = torch.load(args.model_path)


def get_filename_and_extension_from_path(path):
    slash_pos = path.rfind("/")
    dot_pos = path.rfind(".")
    file_name = path[slash_pos+1:dot_pos]
    ext_name = path[dot_pos:]
    return file_name, ext_name

def deeplabv3_segment(model, img_path, destination_path, individual_obj_masks):

    model.eval()

    input_image = Image.open(img_path)
    # input_image = input_image.resize((256, 256))
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # print("<---->", output.min(), output.max())
    # output[output >= 0.1] = 1
    # output[output < 0.1] = 0


    # print(output.shape)

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
        # print(output[:, 0, 0])
        output_predictions = output.argmax(0)

        # print(torch.unique(output_predictions))
    
        # -----------------------------

        # # create a color pallette, selecting a color for each class
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** num_classes - 1])
        # colors = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")

        # # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        # r.putpalette(colors)

        np_r = output_predictions.data.cpu().numpy() # np.array(r)

        # -----------------------------

        if individual_obj_masks:
            ans = []
            for unique_value in np.unique(np_r)[np.unique(np_r) != 0]:
                mask = np_r == unique_value
                ans.append(mask.astype("uint8") * 255)
                # plt.imshow(mask)
                # plt.show()
        else:
            # ans = np.zeros(np_r.shape + (3,))    
            # for unique_value in np.unique(np_r)[np.unique(np_r) != 0]:
            #     color = np.random.random(3) * 255
            #     mask = np_r == unique_value
            #     # plt.imshow(mask)
            #     # plt.show()
            #     mask = np.repeat(mask[:, :, np.newaxis], repeats=3, axis=2)
            #     ans = np.where(mask, color, ans)
            # ans = ans.astype("uint8")
            ans = np_r
            # plt.imshow(ans)
            # plt.show()

        if destination_path:
            if not os.path.exists(destination_path):
                os.mkdir(destination_path)
            img_name, ext_name = get_filename_and_extension_from_path(img_path)
            if individual_obj_masks:
                if not os.path.exists(destination_path+"by_deeplabv3_"+img_name):
                    os.mkdir(destination_path+"by_deeplabv3_"+img_name)
                for i, mask in enumerate(ans):
                    cv2.imwrite(destination_path+"by_deeplabv3_"+img_name+f"/{i}"+ext_name, mask)
            else:
                cv2.imwrite(destination_path+"by_deeplabv3_"+img_name+ext_name, ans)
            
        return ans

    # transparency_mask = Image.new("L", input_image.size, 150)

    # # color_mask = Image.new("RGB", input_image.size, (255, 0, 0))

    # input_image.paste(r, (0, 0), transparency_mask)

    # plt.imshow(input_image)
    # plt.show()

if __name__ == "__main__":
    deeplabv3_segment(model, args.source, args.destination, args.individual_obj_masks)