import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from losses import ContentLoss, StyleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_layers = ["conv_4"]
style_layers   = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    content_layers=content_layers,
    style_layers=style_layers,
):
    cnn = copy.deepcopy(cnn)

    # normalization
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # losses container
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # conv tracker
    for layer in cnn.children():

        # resnet accomdation
        nonseq_layer = 1
        if isinstance(layer, nn.Sequential):
            nonseq_layer = 0
            if isinstance(layer[0], torchvision.models.resnet.BasicBlock):
                for l in layer[0].children():
                    if isinstance(layer, nn.Conv2d):
                        i += 1
                        name = "conv_{}".format(i)
                    elif isinstance(layer, nn.ReLU):
                        name = "relu_{}".format(i)
                        layer = nn.ReLU(inplace=False)
                    elif isinstance(layer, nn.MaxPool2d):
                        name = "pool_{}".format(i)
                    elif isinstance(layer, nn.BatchNorm2d):
                        name = "bn_{}".format(i)
                    model.add_module(name, layer)
        elif isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)


        if nonseq_layer:
            model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)


    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
