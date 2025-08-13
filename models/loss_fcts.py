import torch
import torch.nn as nn
import torchvision


def get_loss_function(loss_name, **args):
    if loss_name == "SigmoidFocalLoss":
        return FocalLossModule(**args)
    else:
        loss_function_class = getattr(nn, loss_name, None)
        return loss_function_class(**args)


class FocalLossModule(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLossModule, self).__init__()
        self.alpha = alpha  # optional alpha parameter for class weights
        self.gamma = gamma  # gamma parameter for focal loss
        self.reduction = reduction  # specifies reduction to apply to the output

    def forward(self, logits, targets):
        # Calculate the sigmoid focal loss using torchvision's function
        loss = torchvision.ops.sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss
