import json
import os
import warnings

import torch
import torchvision.models as models
from tqdm import tqdm

from style_transfer import run_style_transfer
from utils import image_loader, iterate_images, savefig

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir  = "../output"
    content_dir = "../data/content_images"
    style_dir   = "../data/style_images"

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # desired depth layers to compute style/content losses :
    num_steps = 500
    style_weight, content_weight = 1000000, 1

    content_images = iterate_images(content_dir)
    style_images   = iterate_images(style_dir)
    
    content_tensors = [image_loader(content_img) for content_img in content_images]
    style_tensors  = [image_loader(style_img) for style_img in style_images]

    cnns = {
        "vgg11":     models.vgg11(pretrained=True).features.to(device).eval(),
        "vgg19":    models.vgg19(pretrained=True).features.to(device).eval(),
        "vgg11_bn":  models.vgg11_bn(pretrained=True).features.to(device).eval(),
        "vgg19_bn": models.vgg19_bn(pretrained=True).features.to(device).eval(),
        "resnet18":  models.resnet18(pretrained=True).to(device).eval(),
        "resnet34":  models.resnet34(pretrained=True).to(device).eval()
    }

    histories = {key: {} for key in cnns}
    for model_name, cnn in tqdm(cnns.items(), total=len(cnns)):
        use_resnet = "resnet" in model_name
        model_output_dir = os.path.join(output_dir, model_name.strip())
        os.makedirs(model_output_dir, exist_ok=True)
        for content_path, content_img in tqdm(
            zip(content_images, content_tensors), total=len(content_images), leave=False
        ):
            content_title = os.path.splitext(content_path)[0].split("/")[-1]
            for style_path, style_img in tqdm(
                zip(style_images, style_tensors), total=len(style_images), leave=False
            ):
                style_title = os.path.splitext(style_path)[0].split("/")[-1]
                filename = "+".join([content_title, style_title])
                if style_img.size() == content_img.size():
                    input_img = content_img.clone().detach().requires_grad_(True)
                    output, history = run_style_transfer(
                        cnn,
                        cnn_normalization_mean,
                        cnn_normalization_std,
                        content_img,
                        style_img,
                        input_img,
                        style_weight=style_weight,
                        content_weight=content_weight,
                        num_steps=num_steps,
                        use_resnet=use_resnet,
                    )
                    histories[model_name][filename] = history
                    savefig(output, os.path.join(model_output_dir, filename))
                else:
                    print('Invalid combination found!')
                    print(filename)

    with open(os.path.join(output_dir, "histories.json"), "w") as f:
        json.dump(histories_copy, f, indent=4, sort_keys=True)
