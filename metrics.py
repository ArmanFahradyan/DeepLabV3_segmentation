import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss
from torchgeometry.losses import DiceLoss

from sklearn.metrics import f1_score


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        # target = target.view(-1, 1)
        # N,C,H,W => N,C,H*W
        target = target.view(target.size(0), target.size(1), -1)

        # N,C,H*W => N,H*W,C
        target = target.transpose(1, 2)

        # N,H*W,C => N*H*W,C
        target = target.contiguous().view(-1, target.size(2))
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.type(torch.int64)) # added type
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        
        
def focal_loss(logits, true):
    num_classes = logits.shape[1]

    if num_classes == 1:
        return sigmoid_focal_loss(inputs=logits, targets=true, reduction='mean')
    else:
        one_hot_true = F.one_hot(true.to(torch.int64).squeeze(1), num_classes).permute(0, 3, 1, 2) # ???

        return FocalLoss(gamma=2)(logits, one_hot_true.to(torch.float32))

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        # true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        # true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # true_1_hot_f = true_1_hot[:, 0:1, :, :]
        # true_1_hot_s = true_1_hot[:, 1:2, :, :]
        # true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        # pos_prob = torch.sigmoid(logits)
        # neg_prob = 1 - pos_prob
        # probas = torch.cat([pos_prob, neg_prob], dim=1)
        pred = torch.sigmoid(logits)
        intersection = torch.sum(pred * true)
        cardinality = torch.sum(pred + true)
        dice = 2 * intersection / (cardinality + eps)
        return (1 - dice)
    else:
        return DiceLoss()(logits, true) 
    #     true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    #     true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    #     probas = F.softmax(logits, dim=1)
    # true_1_hot = true_1_hot.type(logits.type())
    # dims = (0,) + tuple(range(2, true.ndimension()))
    # intersection = torch.sum(probas * true_1_hot, dims)
    # cardinality = torch.sum(probas + true_1_hot, dims)
    # dice_loss = (2. * intersection / (cardinality + eps)).mean()
    # return (1 - dice_loss)


# def hybrid_loss(prediction, target):
#     """Calculating the loss"""

#     # gamma=0, alpha=None --> CE
#     focal = FocalLoss(gamma=0, alpha=None)

#     bce = focal(prediction, target)
#     dice = dice_loss(prediction, target)
#     loss = bce + dice

#     return loss

class HybridLoss(nn.Module):

    def __init__(self, *losses):
        super(HybridLoss, self).__init__()
        self.losses = []
        for loss in losses:
            self.losses.append(loss)

    def forward(self, input, target):
        overall_loss = 0
        for loss in self.losses:
            overall_loss += loss(input, target)
        return overall_loss


# def hybrid_loss(logits, target):

#     # bce = nn.BCEWithLogitsLoss()
#     loss = focal_loss(logits, target) + dice_loss(logits, target) # bce(logits + 0.1, target)  # sigmoid_focal_loss(inputs=logits+0.1, targets=target, reduction='mean')

#     return loss


def iou(pred, target, n_classes = 3):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union > 0:
        ious.append(float(intersection) / float(max(union, 1)))

  return np.array(ious)


def f1(logits, masks):
    assert len(logits.shape) == 4 and len(masks.shape) == 3, "invalid shape of input"
    num_classes = logits.shape[1]
    y_true = F.one_hot(masks.squeeze(1).to(torch.int64), num_classes).permute(0, 3, 1, 2).data.cpu().numpy().ravel()
    y_pred = F.one_hot(F.softmax(logits, dim=1).argmax(dim=1), num_classes).permute(0, 3, 1, 2).data.cpu().numpy().ravel()
    return f1_score(y_true > 0, y_pred > 0)
