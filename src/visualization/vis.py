import math
import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

from engine.logger import Logger


def normalize(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    max_val = tensor.max() if dim is None else tensor.max(dim=dim, keepdim=True)
    min_val = tensor.min() if dim is None else tensor.min(dim=dim, keepdim=True)
    return (tensor - min_val) / (max_val - min_val)


def register_hook(model: nn.Module, layers: List[str]) -> Dict[str, List[torch.Tensor]]:
    result = defaultdict(list)

    def hook(
        self: nn.Module,
        args: Tuple[torch.Tensor],
        output: torch.Tensor,
        layer: str,
    ):
        result[layer].append(output.detach())

    for layer in layers:
        try:
            module = model.get_submodule(layer)
            module.register_forward_hook(partial(hook, layer=layer))
        except AttributeError:
            pass

    return result


def remove_hook(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()


def retrieve_weights(model: nn.Module, layers: List[str]) -> Dict[str, torch.Tensor]:
    result = dict()

    for layer in layers:
        try:
            module = model.get_submodule(layer)
            if hasattr(module, "weight"):
                result[layer] = module.weight.detach()
        except AttributeError:
            pass

    return result


def heatmap(
    tensor: torch.Tensor,
    img_labels: Optional[List[Image.Image]] = None,
    text_labels: Optional[List[str]] = None,
) -> Figure:
    assert tensor.dim() == 2

    batch, dim = tensor.shape
    figure = plt.figure(num=1, clear=True)
    ax = figure.add_subplot()

    heatmap = sns.heatmap(
        tensor.numpy(),
        cmap="viridis",
        cbar=True,
        yticklabels=False,
        ax=ax,
    )
    heatmap.set_ylabel("1st Dimension")
    heatmap.set_xlabel("2nd Dimension")

    if img_labels and text_labels:
        heatmap.set_ylabel("")
        heatmap.set_yticks(torch.arange(batch).numpy() + 0.5)
        heatmap.set_xticks(torch.arange(dim).numpy() + 0.5)

        for idx, (img, text) in enumerate(zip(img_labels, text_labels)):
            zoom = 16 / max(*img.size)
            box = OffsetImage(img, zoom=zoom)
            box = AnnotationBbox(
                offsetbox=box,
                xy=(0, idx + 0.5),
                xybox=(-20, 0),
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
            )

            heatmap.add_artist(box)
            heatmap.text(
                x=-0.12,
                y=idx + 0.5,
                s=text,
                va="center",
                ha="right",
                transform=heatmap.get_yaxis_transform(),
                fontsize=6,
            )

    figure.tight_layout()
    return figure


def create_grid(
    num: int,
    tensor: torch.Tensor,
    img_label: Optional[Image.Image] = None,
    text_label: Optional[str] = None,
) -> Figure:
    assert tensor.dim() == 3

    figure = plt.figure(num=num, clear=True)
    nchannel = tensor.size(0) + 1 if img_label else tensor.size(0)
    nrow = math.floor(math.sqrt(nchannel))
    ncol = math.ceil(nchannel / nrow)

    start_idx = 1
    if img_label:
        start_idx = 2
        ax = figure.add_subplot(nrow, ncol, 1)
        ax.axis("off")
        ax.imshow(img_label)

    for idx, feat in enumerate(tensor, start=start_idx):
        ax = figure.add_subplot(nrow, ncol, idx)
        ax.axis("off")
        ax.imshow(feat, cmap="viridis")

    if text_label:
        figure.suptitle(text_label, fontsize=14, y=0.95)

    figure.tight_layout()
    return figure


def draw_weights(weights: torch.Tensor, title: Optional[str] = None) -> Figure:
    if weights.dim() == 4:
        return create_grid(1, weights.mean(dim=1), text_label=title)
    elif weights.dim() == 2:
        return heatmap(weights)
    else:
        raise NotImplementedError


def draw_feature_map(
    feature_map: torch.Tensor,
    img_labels: List[Image.Image],
    text_labels: List[str],
) -> Figure | List[Figure]:
    if feature_map.dim() == 4:
        return [
            create_grid(num, feats, img, txt)
            for num, (feats, img, txt) in enumerate(
                zip(feature_map, img_labels, text_labels), start=1
            )
        ]
    elif feature_map.dim() == 2:
        return [heatmap(feature_map, img_labels, text_labels)]
    else:
        raise NotImplementedError


def draw_grad_cam(
    model: nn.Module,
    target_layer: List[nn.Module],
    targets: List[ClassifierOutputTarget],
    input_tensor: torch.Tensor,
    input_img: Image.Image,
    method: Callable[..., BaseCAM] = GradCAM,
    reshape_transform: Optional[Callable[..., torch.Tensor]] = None,
) -> Figure:
    figure = plt.figure(num=1, clear=True)
    size = len(targets) + 1
    nrow = math.floor(math.sqrt(size))
    ncol = math.ceil(size / nrow)

    ax = figure.add_subplot(nrow, ncol, 1)
    ax.set_title("Original")
    ax.axis("off")
    ax.imshow(input_img)

    img_array = np.array(input_img, dtype=np.float32) / 255
    with method(model, target_layer, reshape_transform=reshape_transform) as cam:
        repeated_tensor = input_tensor.repeat(len(targets), 1, 1, 1)
        cam_output = cam(repeated_tensor, targets)

        for idx, mask in enumerate(cam_output, start=2):
            heatmap = show_cam_on_image(img_array, mask, use_rgb=True)
            heatmap = Image.fromarray(heatmap)
            title = (
                f"Predicted : {targets[idx - 2].category}"
                if idx > 2
                else f"Actual : {targets[idx - 2].category}"
            )

            ax = figure.add_subplot(nrow, ncol, idx)
            ax.set_title(title)
            ax.axis("off")
            ax.imshow(heatmap)

    figure.tight_layout()
    return figure


def generate_vis(
    epoch: int,
    img_dir: Path,
    model: nn.Module,
    layers: List[str],
    tensor_transform: Callable[..., torch.Tensor],
    raw_transform: Callable[..., Image.Image],
    logger: Logger,
) -> None:
    model.eval()
    model.cpu()
    hooks = register_hook(model, layers)
    weights = retrieve_weights(model, layers)

    raw_img = [raw_transform(Image.open(img)) for img in img_dir.glob("*.png")][:5]
    raw_name = [str(img).split("/")[-1][:-4] for img in img_dir.glob("*.png")][:5]
    tensors = [tensor_transform(img) for img in raw_img]

    batch = 16
    with warnings.catch_warnings(action="ignore"):
        for layer, weight in tqdm(weights.items(), desc="Weight Figures", leave=True):
            figure = draw_weights(normalize(weight), layer)
            logger.write_figure(figure, f"weights_{layer}", epoch)
            figure.clear()

        for ind in range(0, len(raw_img), batch):
            input_tensor = torch.stack(tensors[ind : ind + batch])
            orig_img = raw_img[ind : ind + batch]
            img_name = raw_name[ind : ind + batch]
            img_label = list(map(lambda x: int(x.split("_")[-1]), img_name))

            preds = model(input_tensor).argmax(dim=1)
            for layer, item in tqdm(
                hooks.items(),
                desc=f"({ind} / {len(raw_img)}) Feature Map",
                leave=True,
            ):
                feature = item[-1]
                figure = draw_feature_map(normalize(feature), orig_img, img_name)
                logger.write_figure(figure, f"feature_{layer}", epoch)

                if isinstance(figure, list):
                    list(map(lambda x: x.clear(), figure))
                else:
                    figure.clear()

            for img, tensor, label, pred, img_name in tqdm(
                zip(orig_img, input_tensor, img_label, preds, img_name),
                desc=f"({ind} / {len(raw_img)}) GradCAM",
                leave=True,
                total=len(orig_img),
            ):
                figure = draw_grad_cam(
                    model,
                    [model.layers[-1]],
                    [ClassifierOutputTarget(label), ClassifierOutputTarget(pred)],
                    tensor,
                    img,
                )
                logger.write_figure(
                    figure,
                    f"GradCAM_{img_name.split('_')[0]}",
                    epoch,
                )
                figure.clear()

            hooks.clear()

    plt.cla()
    plt.clf()
    plt.close("all")
    remove_hook(model)


if __name__ == "__main__":
    pass
