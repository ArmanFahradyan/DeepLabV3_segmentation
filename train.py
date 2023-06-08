import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import copy
import csv
from tqdm import tqdm

from matplotlib import pyplot as plt
from PIL import Image


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs, testing=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # criterion.to(device)
    # Initialize the log file for training and validating loss and metrics
    fieldnames = ['epoch', 'train_loss', 'val_loss'] + \
        [f'train_{m}' for m in metrics.keys()] + \
        [f'val_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in (["val"] if testing else ['train', "val"]):
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device) # .type(torch.cuda.IntTensor64) # .to(torch.int64)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    logits = outputs['out']
                    num_classes = logits.shape[1]
                    loss = criterion(logits, masks) #  + 0.1 * (num_classes == 1)
                    
                    if num_classes == 1:
                        y_true = masks.data.cpu().numpy().ravel()
                        y_pred = logits.data.cpu().numpy().ravel()
                    else:
                        y_true = F.one_hot(masks, num_classes).permute(0, 3, 1, 2).data.cpu().numpy().ravel()
                        assert y_true.sum() == len(y_true) // num_classes
                        y_pred = F.one_hot(logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).data.cpu().numpy().ravel() # y_pred = F.softmax(logits, dim=1) # .data.cpu().numpy().ravel()


                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            if num_classes == 1:
                                # Use a classification threshold of 0.1
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true > 0, y_pred > 0.1))
                            else:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(masks.data.cpu().numpy().ravel(), logits.argmax(dim=1).data.cpu().numpy().ravel(), average='weighted')
                                )
                        elif name == 'iou':
                            if num_classes == 1:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(torch.sigmoid(logits), masks, num_classes).mean())
                            else:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(logits.argmax(dim=1), masks, num_classes).mean())
                        else:
                            if num_classes == 1:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true.astype('uint8'), y_pred))
                            else:
                                pass

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == "val" and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model