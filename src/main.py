import gc
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
from omegaconf import DictConfig
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import distribute_bn
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dataset.imagenet import build as build_imagenet
from dataset.imagenet import build_transform
from engine.logger import Logger
from model.spartan import build as build_model
from trainer.normal import evaluate, train
from visualization.vis import generate_vis


def create_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    filepath: Path,
    config: DictConfig,
    epoch: int,
    model: nn.Module,
    optim: Optimizer,
    sched: LRScheduler,
    scaler: Optional[torch.GradScaler] = None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": config,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
        },
        filepath,
    )


def load_checkpoint(
    filepath: Optional[Path],
    model: nn.Module,
    optim: Optimizer,
    sched: LRScheduler,
    scaler: Optional[torch.GradScaler] = None,
    resume: bool = False,
) -> int:
    epoch = 1

    if filepath is not None:
        checkpoint = torch.load(filepath)
        missing_key, unexpected_key = model.load_state_dict(
            checkpoint["model"], strict=False
        )

        if resume:
            epoch = checkpoint["epoch"] + 1
            optim.load_state_dict(checkpoint["optim"])
            sched.load_state_dict(checkpoint["sched"])

            if scaler and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])

        print(f"Missing Keys in Model Dict\n {','.join(missing_key)}\n")
        print(f"Unexpected Key in Model Dict\n {','.join(unexpected_key)}\n")

    flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224).cuda().contiguous())
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} - {param.numel()} - {param.stride()}")

    print(
        f"Total params : {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(flop_count_table(flops))

    return epoch


def set_weight_decay(
    model: nn.Module, skip_list: Tuple[str] = ()
) -> List[Dict[str, Any]]:
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            len(param.shape) == 1
            or name.endswith("bias")
            or any([n in name for n in skip_list])
        ):
            no_decay_names.append(name)
            no_decay.append(param)
        else:
            has_decay.append(param)

    for name in no_decay_names:
        print(f"NO WEIGHT DECAY : {name}")
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def objective(config: DictConfig) -> None:
    if config.distributed:
        dist.init_process_group("nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        torch.cuda.set_device(torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    current_date, current_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S").split("_")
    save_dir = create_dir(Path(config.checkpoint_dir) / current_date / current_time)
    logger_dir = create_dir(Path(config.logger_dir) / current_date / current_time)
    images_dir = create_dir(Path(config.data.data_dir) / "images")

    #### Reproducibility ####
    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Use deterministic or randomized algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)
    #########################

    train_loader, test_loader = build_imagenet(config.data, config.distributed)
    transform, raw_transform = build_transform(config.data.transform)
    logger = Logger(logger_dir, config.window)

    test_loss = []
    test_acc = []

    model = build_model(config.data.num_classes, config.model.build)
    optim = AdamW(
        set_weight_decay(model, skip_list=tuple(config.model.skip_list)),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )
    sched = CosineLRScheduler(
        optim,
        t_initial=config.t_initial,
        lr_min=config.min_lr,
        warmup_t=config.warmup_t,
        warmup_lr_init=config.warmup_init_lr,
        t_in_epochs=True,
    )
    train_crit = (
        SoftTargetCrossEntropy()
        if config.data.transform.augment
        else CrossEntropyLoss(label_smoothing=config.label_smoothing)
    )
    eval_crit = CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler() if config.use_amp else None

    model_without_ddp = model
    if torch.cuda.is_available():
        model = model.cuda(rank)

    init_epoch = load_checkpoint(
        config.load_file, model, optim, sched, scaler, config.resume
    )

    if config.distributed:
        model = DDP(model, device_ids=[rank])

    for epoch in range(init_epoch, config.epoch + 1):
        train(
            epoch,
            train_loader,
            model,
            train_crit,
            optim,
            logger,
            scaler,
            config.max_norm,
        )

        if config.distributed:
            distribute_bn(model, world_size, reduce=True)

        evaluate(
            epoch,
            test_loader,
            model,
            eval_crit,
            logger,
        )
        sched.step(epoch - 1)

        if rank == 0:
            logger.print()
            logger.write_csv("metric.csv", epoch)
            logger.write_scalar(epoch, "Loss", "Acc")
        logger.clear()

        if save_dir.exists():
            save_checkpoint(
                save_dir / f"{config.model.name}_latest.pth",
                config,
                epoch,
                model_without_ddp,
                optim,
                sched,
                scaler,
            )

            if (epoch % config.save_freq == 0 or epoch == config.epoch) and rank == 0:
                save_checkpoint(
                    save_dir / f"{config.model.name}_{epoch}.pth",
                    config,
                    epoch,
                    model_without_ddp,
                    optim,
                    sched,
                    scaler,
                )

        if epoch % config.log_freq == 0:
            logger.write_histogram(model_without_ddp, epoch)

        gc.collect()
        torch.cuda.empty_cache()

    evaluate(epoch + 1, test_loader, model, eval_crit, logger)
    test_loss.append(logger["eval_loss_val"].g_average)
    test_acc.append(logger["eval_acc_1"].g_average)
    if rank == 0:
        logger.print()
        logger.write_csv("metric.csv", epoch + 1)
        logger.write_scalar(epoch + 1, "Loss", "Acc")
    logger.clear()

    if rank == 0:
        generate_vis(
            epoch + 1,
            images_dir,
            model.module if config.distributed else model,
            config.model.vis,
            transform,
            raw_transform,
            logger,
        )

    # Free memory
    del model, optim, sched
    del train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

    logger.close()


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    objective(config)
    if config.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
