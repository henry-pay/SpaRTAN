import random as random
from collections.abc import Callable
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageNet
from torchvision.transforms import v2


def transform(train: bool, config: DictConfig) -> v2.Compose:
    if config.augment and train:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(size=config.crop_size),
                v2.RandomHorizontalFlip(p=config.h_prob),
                v2.RandAugment(
                    magnitude=config.magnitude,
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.ToDtype(
                    torch.float32 if config.data_dtype == "32-bit" else torch.float16,
                    scale=True,
                ),
                v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                v2.RandomErasing(p=config.re_prob),
            ]
        )
    else:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=config.img_size),
                v2.CenterCrop(size=config.crop_size),
                v2.ToDtype(
                    torch.float32 if config.data_dtype == "32-bit" else torch.float16,
                    scale=True,
                ),
                v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )


def worker_seed(_) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_transform(config: DictConfig) -> Tuple[v2.Compose, Callable]:
    return (
        transform(False, config),
        lambda x: x.resize((config.crop_size, config.crop_size)),
    )


def build(
    config: DictConfig,
    distributed: bool = False,
) -> Generator[Tuple[List[DataLoader], List[DataLoader], DataLoader], None, None]:
    data_dir = Path(config.data_dir)

    if config.name == "IMAGENET":
        data_class = ImageNet
    else:
        raise NotImplementedError

    train_dataset = data_class(
        root=data_dir,
        split="train",
        transform=transform(True, config.transform),
    )
    test_dataset = data_class(
        root=data_dir,
        split="val",
        transform=transform(False, config.transform),
    )
    cutmix_mixup = v2.RandomChoice(
        [
            v2.CutMix(
                alpha=config.transform.cutmix_alpha,
                num_classes=config.num_classes,
            ),
            v2.MixUp(
                alpha=config.transform.mixup_alpha,
                num_classes=config.num_classes,
            ),
        ]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.data_seed)
    train_dataloader = DataLoader(
        train_dataset,
        config.batch_size,
        shuffle=None if distributed else True,
        sampler=DistributedSampler(
            train_dataset, shuffle=True, seed=config.data_seed, drop_last=True
        )
        if distributed
        else None,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_seed,
        generator=train_generator,
        collate_fn=(
            (lambda x: cutmix_mixup(*default_collate(x)))
            if config.transform.augment
            else default_collate
        ),
    )

    test_generator = torch.Generator()
    test_generator.manual_seed(config.data_seed)
    test_dataloader = DataLoader(
        test_dataset,
        config.batch_size,
        shuffle=None if distributed else True,
        sampler=DistributedSampler(
            test_dataset, shuffle=True, seed=config.data_seed, drop_last=False
        )
        if distributed
        else None,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_seed,
        generator=test_generator,
    )

    return train_dataloader, test_dataloader
