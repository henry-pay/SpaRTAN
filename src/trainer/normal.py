import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from timm.utils import accuracy, reduce_tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.logger import Logger

device_type = "cuda" if torch.cuda.is_available() else "cpu"
device = (
    torch.device(device_type)
    if torch.cuda.is_available()
    else torch.device(device_type)
)


def train(
    epoch: int,
    dataloder: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optim: Optimizer,
    logger: Logger,
    scaler: Optional[torch.GradScaler] = None,
    max_norm: float = 1.0,
):
    model.train()
    optim.zero_grad()

    if not dist.is_initialized():
        global device
        model.to(device)
    else:
        device = dist.get_rank()

    data = tqdm(dataloder, total=len(dataloder), leave=True)
    for iter_idx, (X, y) in enumerate(data):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        skip = False

        if scaler:
            with torch.amp.autocast(device_type, torch.float16):
                out = model(X.contiguous())
                loss = criterion(out, y)
        else:
            out = model(X)
            loss = criterion(out, y)

        if torch.isnan(loss) or not math.isfinite(loss.item()):
            print(f"NaN Loss Encountered at {epoch}_{iter_idx}")
            skip = True
            module = model
            if dist.is_initialized():
                module = model.module
            logger.write_failure(epoch, iter_idx, device, module, X, out, y)

        if not skip:
            if scaler:
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()

                if max_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                optim.step()

        optim.zero_grad()
        torch.cuda.synchronize()

        if not skip:
            if dist.is_initialized():
                loss_val = reduce_tensor(loss.data, dist.get_world_size()).item()
            else:
                loss_val = loss.item()

            batch = y.size(0)
            logger(train_loss_val=(loss_val, batch))

            data.set_description(f"Train Epoch {epoch}")
            data.set_postfix(loss=f"{logger['train_loss_val'].average:.4f}")


@torch.no_grad
def evaluate(
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    logger: Logger,
):
    model.eval()

    if not dist.is_initialized():
        global device
        model.to(device)
    else:
        device = dist.get_rank()

    data = tqdm(dataloader, total=len(dataloader), leave=True)
    for X, y in data:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(X.contiguous())
        loss = criterion(out, y)

        torch.cuda.synchronize()

        acc_1, acc_5 = accuracy(out, y, topk=(1, 5))
        if dist.is_initialized():
            loss_val = reduce_tensor(loss, dist.get_world_size()).item()
            acc_1 = reduce_tensor(acc_1, dist.get_world_size())
            acc_5 = reduce_tensor(acc_5, dist.get_world_size())
        else:
            loss_val = loss.item()

        batch = y.size(0)
        logger(
            eval_loss_val=(loss_val, batch),
            eval_acc_1=(acc_1, batch),
            eval_acc_5=(acc_5, batch),
        )

        data.set_description(f"Eval  Epoch {epoch}")
        data.set_postfix(
            loss=f"{logger['eval_loss_val'].average:.4f}",
            top_1=f"{logger['eval_acc_1'].average:.3f}",
            top_5=f"{logger['eval_acc_5'].average:.3f}",
        )


if __name__ == "__main__":
    pass
