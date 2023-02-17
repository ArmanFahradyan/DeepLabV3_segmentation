from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from train import train_model
from metrics import hybrid_loss
# from torchmetrics import Dice
# from torchgeometry.losses import DiceLoss


@click.command()
@click.option("--data_directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option("--model_path",
              default='',
              help="Specify the model(weights) path. By default model is DeepLabV3 with resnet101 backbone")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--num_classes",
              default=1,
              type=int,
              help="Specify the number of classes in the dataset.")
def main(data_directory, exp_directory, model_path, epochs, batch_size, num_classes):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    if model_path:
        model = torch.load(model_path)
    else:
        model = createDeepLabv3(num_classes)
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = hybrid_loss # dice_loss # DiceLoss()  #  Dice(average='micro', threshold=0.1)  # torch.nn.MSELoss(reduction='mean') #    dice_loss  #  hybrid_loss # 
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_sep_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs,
                    testing=False)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()