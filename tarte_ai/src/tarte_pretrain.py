"""
TARTE Pretrainer that pretrains a transformer with data from knowledge graphs.
"""

import torch
import numpy as np
from datetime import datetime
from tarte_ai.src.tarte_model import TARTE_Pretrain_NN


def _calculate_similarity(x):
    """
    Function to calculate the similarity using Gaussian Similarity.
    """
    out = torch.nn.functional.normalize(x, dim=1)
    dist = torch.cdist(out, out)
    sig = torch.median(dist) + 1e-9  # For stability
    out = torch.exp(-(dist / (2 * sig))).squeeze(dim=1)
    return out


def _calculate_infonce(similarity, y_target):
    """Function to calculate the infonce loss."""
    data_size = similarity.size(0)
    pos_mask = (y_target - torch.eye(data_size, device=y_target.device)).to(torch.bool)
    loss = -similarity[pos_mask].reshape(data_size, pos_mask.sum(0)[0]).sum(dim=1)
    loss += torch.logsumexp(
        similarity - (np.exp(1) + 1) * torch.eye(data_size, device=y_target.device),
        dim=1,
    )
    return loss.mean()


class TARTE_Pretrainer:
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        model_configs: dict,
        save_dir: str = "./ckpt",
        loss_terms: str = "nce",
        learning_rate: float = 1e-6,
        warmup_steps: int = 10000,
        save_every: int = 1000,
        multi_gpu: bool = False,
        device: str = "0",
    ):
        """TARTE pretrainer with knowledge graphs.

        This pretrainer is based on the transformers with data preprocessed from knowledge graphs.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader,
            The custom dataloader that preprocesses data from knowledge graph.
        model_configs : dict,
            The dictionary of model (transformer) configurations.
        save_dir : str, default='./ckpt',
            The directory to save configurations and model weights.
        loss_terms: str, default="nce",
            The loss to use for the contrastive pretraining.
        learning_rate : float, default=1e-6,
            The learning rate of the model. The model uses AdamW as the optimizer.
        warmup_steps : int, default=10000,
            The warm up steps for the scheduler.
        save_every :  int, default=1000,
            The step to save the checkpoints and logs.
        multi_gpu: bool, default=False,
            Indicates whether to use multi-gpu. The running script must set initial processes for multi-gpu usage.
        device: 'cpu' or {'int'}, default= '0',
            The device number used for the estimator.
        """

        self.train_loader = train_loader
        self.model_configs = model_configs
        self.save_dir = save_dir
        self.loss_terms = loss_terms
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_every = save_every
        self.device = device
        self.multi_gpu = multi_gpu

    def fit(self):
        """Fit function to pretrain the model."""

        self._set_pretrainer()
        self.model.train()
        step = 0
        self._save_checkpoint(step)

        # Iterate with number of steps defined by the train_loader
        for (
            x,
            edge_attr,
            mask,
            original_mask,
            y_target,
        ) in self.train_loader:
            step += 1

            # Send to device
            x = x.to(self.device, non_blocking=True)
            edge_attr = edge_attr.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            original_mask = original_mask.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)

            # Run step
            self._run_step(
                x,
                edge_attr,
                mask,
                y_target,
                step,
            )

        return None

    def _run_step(
        self,
        x,
        edge_attr,
        mask,
        y_target,
        step,
    ):
        """Standard step function of feed-foward and back-prop."""

        self.optimizer.zero_grad()  # Clear gradients.
        out_con = self.model(x, edge_attr, mask)  # Feed-Forward
        sim = [_calculate_similarity(out_) for out_ in out_con]

        if self.loss_terms == "nce":
            mask_self = torch.eye(out_con_.size(0)).to(torch.bool)
            out_con_, y_target = out_con_[~mask_self], y_target[~mask_self]
            loss = self.criterion_con_(out_con_, y_target)  # Calculate loss
        elif self.loss_terms == "infonce":
            loss = torch.tensor(0, dtype=torch.float32, device=x.device)
            for sim_ in sim:
                loss += _calculate_infonce(sim_, y_target)
            loss = loss / len(sim)

        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Step for the optimizer.
        self.scheduler.step()  # Step for the scheduler.

        if self.multi_gpu:
            if self.device == 0 and step % self.save_every == 0:
                now = datetime.now()
                self.loss_.append(
                    f"{round(loss.detach().item(), 4)}|{now.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self._save_checkpoint(step)
        else:
            if step % self.save_every == 0:
                now = datetime.now()
                self.loss_.append(
                    f"{round(loss.detach().item(), 4)}|{now.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self._save_checkpoint(step)

        return None

    def _save_checkpoint(self, step):
        """Function to save the checkpoint and logs."""

        if self.multi_gpu:
            ckp = self.model.module.state_dict()
        else:
            ckp = self.model.state_dict()
        save_path = self.save_dir + f"/ckpt_step{step}.pt"
        torch.save(ckp, save_path)

        log_path = self.save_dir + f"/log_train.txt"
        with open(log_path, "w") as output:
            for row in self.loss_:
                output.write(str(row) + "\n")

    def _set_pretrainer(self):
        """Set pretrainer."""

        now = datetime.now()
        self.loss_ = [f"0|{now.strftime('%Y-%m-%d %H:%M:%S')}"]

        # Model
        torch.manual_seed(20122024)
        torch.cuda.manual_seed(20122024)
        np.random.seed(20122024)

        model = TARTE_Pretrain_NN(**self.model_configs)

        self.model = model.to(self.device)
        if self.multi_gpu:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(model, device_ids=[self.device])
        else:
            self.model = model.to(self.device)
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        # Scheduler - warmup / decay
        self.scheduler_warm = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1,
            total_iters=self.warmup_steps,
        )
        self.scheduler_decay = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=int(len(self.train_loader) - self.warmup_steps),
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.scheduler_warm, self.scheduler_decay],
            milestones=[self.warmup_steps],
        )
        # torch loss calcuation
        if self.loss_terms == "nce":
            self.criterion_con_ = torch.nn.BCELoss()
        else:
            self.criterion_con_ = None

        return None
