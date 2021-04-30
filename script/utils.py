import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iterate_images(dir: str) -> list:
    image_paths = []
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(dir, filename)
            image_paths.append(path)
    return image_paths

# reference: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def image_loader(image_name):
    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128

    loader = transforms.Compose(
        [
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)


def savefig(tensor, path=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
