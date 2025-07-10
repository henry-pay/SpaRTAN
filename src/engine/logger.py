import math
from collections import deque
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid


class Metric:
    def __init__(self, window: int):
        self.queue = deque(maxlen=window)
        self.total = self.count = 0

    def __call__(self, value: float, count: int) -> None:
        assert count > 0, "Count cannot be 0"
        self.queue.append(value)
        self.total += value * count
        self.count += count

    def __len__(self) -> int:
        return self.count

    def __str__(self) -> str:
        return f"{self.g_average:.2f}" if self.count != 0 else ""

    @property
    def average(self) -> float:
        if len(self.queue) == 0:
            return 0.0
        return sum(self.queue) / len(self.queue)

    @property
    def max(self) -> float:
        if len(self.queue) == 0:
            return None
        return max(self.queue)

    @property
    def g_average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def clear(self) -> None:
        self.queue.clear()
        self.total = self.count = 0


class Logger:
    def __init__(self, root_dir: str | Path, metric_window: int):
        self.root_dir = Path(root_dir)
        self.writer = SummaryWriter(root_dir / "tensorboard")
        self.metric_window = metric_window
        self.metrics: Dict[str, Metric] = dict()

    def __len__(self) -> int:
        return len(self.metrics)

    def __getitem__(self, key: str) -> Metric:
        if isinstance(key, str):
            return self.metrics[key]
        raise TypeError("Key is invalid type")

    def __setitem__(self, key: str, value: Metric) -> None:
        if isinstance(key, str):
            self.metrics[key] = value
            return
        raise TypeError("Key is invalid type")

    def __call__(self, **kwargs) -> None:
        for key, item in kwargs.items():
            if key in self.metrics:
                self[key](*item)
            else:
                self[key] = Metric(self.metric_window)
                self[key](*item)

    def write_csv(self, filename: str, epoch: int) -> None:
        file_path = self.root_dir / filename
        assert file_path.suffix == ".csv", "File ext is incorrect"

        with file_path.open(mode="a+", encoding="utf8") as file:
            header = ",".join(["epoch", *self.metrics.keys()])
            values = ",".join(map(str, [epoch, *self.metrics.values()]))

            file.seek(0)
            if header not in file.readline().strip():
                file.write(header)
                file.write("\n")

            file.write(values)
            file.write("\n")

    def write_scalar(self, epoch: int, *args: str) -> None:
        for tag in args:
            self.writer.add_scalars(
                f"{tag}",
                {
                    k: v.g_average
                    for k, v in self.metrics.items()
                    if tag.lower() in k.lower()
                },
                epoch,
            )

    def write_network(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        limit: int = 10_000,
    ) -> None:
        img, _ = next(iter(dataloader))
        self.writer.add_graph(model, img)

        imgs = []
        lbls = []
        batch = img.size(0)
        for img, lbl in dataloader:
            imgs.append(img)
            lbls.append(lbl)

            if len(imgs) * batch > limit:
                break

        imgs = torch.cat(imgs, dim=0)
        lbls = torch.cat(lbls, dim=0).view(-1).tolist()
        total_count = imgs.size(0)

        self.writer.add_image(
            "input images",
            make_grid(imgs, math.floor(math.sqrt(len(imgs)))),
        )
        self.writer.add_embedding(
            mat=imgs.view(total_count, -1),
            metadata=lbls,
            label_img=imgs,
            tag="input images",
        )

    def write_histogram(self, model: nn.Module, epoch: int) -> None:
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"{name}", param, epoch)

    def write_embedding(
        self,
        epoch: int,
        tag: str,
        embed: torch.Tensor,
        imgs: torch.Tensor,
        lbls: torch.Tensor,
    ) -> None:
        assert embed.dim() == 2
        assert imgs.dim() == 4
        assert lbls.dim() == 1

        lbls = lbls.view(-1).tolist()
        self.writer.add_embedding(
            mat=embed, metadata=lbls, label_img=imgs, global_step=epoch, tag=tag
        )

    def write_pr_curve(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> None:
        for class_idx in range(max(labels) + 1):
            t_label = labels == class_idx
            t_prob = probs[:, class_idx]

            self.writer.add_pr_curve(f"{class_idx}", t_label, t_prob, epoch)

    def write_figure(
        self,
        figure: Figure | List[Figure],
        tag: str,
        epoch: int,
    ) -> None:
        self.writer.add_figure(tag, figure, epoch)

    def print(self) -> None:
        train_metric = []
        eval_metric = []

        for key, item in self.metrics.items():
            if "train" in key.lower():
                train_metric.append((key, item))
            elif "eval" in key.lower():
                eval_metric.append((key, item))

        print(
            "Train :",
            ", ".join(
                map(
                    lambda x: f"{x[0]} - {x[1]}",
                    sorted(train_metric, key=lambda x: x[0]),
                )
            ),
        )

        print(
            "Eval  :",
            ", ".join(
                map(
                    lambda x: f"{x[0]} - {x[1]}",
                    sorted(eval_metric, key=lambda x: x[0]),
                )
            ),
        )

    def write_failure(
        self,
        epoch: int,
        iter_idx: int,
        rank: int,
        model: nn.Module,
        inp_tensor: torch.Tensor,
        out_tensor: torch.Tensor,
        label: torch.Tensor,
    ) -> None:
        directory = self.root_dir / "debug" / f"{rank}_{epoch}_{iter_idx}"
        if not directory.exists():
            directory.mkdir(parents=True)

        torch.save(model, directory / "model.pth")
        torch.save(inp_tensor.cpu().detach(), directory / "inp.pt")
        torch.save(out_tensor.cpu().detach(), directory / "out.pt")
        torch.save(label.cpu().detach(), directory / "label.pt")

    def clear(self) -> None:
        for metric in self.metrics.values():
            metric.clear()

    def close(self) -> None:
        self.writer.close()


if __name__ == "__main__":
    pass
