# flake8: noqa: E203
import os
from typing import Any, Dict, Optional, Tuple

from loguru import logger
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.models.unet import UNetModel  # type: ignore[import-untyped]
from torchcfm import ExactOptimalTransportConditionalFlowMatcher  # type: ignore[import-untyped]
import copy
from torchdyn.core import NeuralODE  # type: ignore[import-untyped]

from minecraft_copilot_ml.data_loader import MinecraftSchematicsDatasetItemType


def ema(source: nn.Module, target: nn.Module, decay: float) -> None:
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay + source_dict[key].data * (1 - decay))


class MinecraftCopilotTrainer(pl.LightningModule):
    def __init__(  # type: ignore[no-any-unimported]
        self,
        unet_model: UNetModel,
        unique_blocks_dict: Dict[str, int],
        save_dir: str = "output",
    ):
        super(MinecraftCopilotTrainer, self).__init__()
        self.unique_blocks_dict = unique_blocks_dict
        self.reverse_unique_blocks_dict = {v: k for k, v in unique_blocks_dict.items()}
        self.unet_model = unet_model
        self.ema_model = copy.deepcopy(unet_model)
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        self.step_number = 0
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def forward(self, t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        return self.unet_model(t, xt)  # type: ignore[no-any-return]

    def step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int, mode: str) -> None:
        optimizer = self.trainer.optimizers[0]
        optimizer.zero_grad()

        block_maps, block_map_masks = batch
        pre_processed_block_maps = self.pre_process(block_maps)
        tensor_block_map_masks = (
            torch.from_numpy(block_map_masks).float().to("cuda" if torch.cuda.is_available() else "cpu").long()
        )
        x1 = pre_processed_block_maps
        x0 = torch.randn_like(x1)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = self(t, xt)
        loss = (ut - vt) ** 2
        loss = loss * tensor_block_map_masks  # Mask out the loss for the blocks outside the schematic
        loss = loss.mean()
        self.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        ema(self.unet_model, self.ema_model, 0.9999)

        # Total loss
        loss_dict = {
            "loss": loss,
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        for name, value in loss_dict.items():
            self.log(
                f"{mode}_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=block_maps.shape[0],
            )

    def pre_process(self, x: np.ndarray) -> torch.Tensor:
        vectorized_x = np.vectorize(lambda x: self.unique_blocks_dict.get(x, self.unique_blocks_dict["minecraft:air"]))(
            x
        )
        vectorized_x = vectorized_x.astype(np.int64)
        x_tensor = torch.from_numpy(vectorized_x)
        x_tensor = x_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = F.one_hot(x_tensor, num_classes=len(self.unique_blocks_dict)).permute(0, 4, 1, 2, 3).float()
        return x_tensor

    def post_process(self, x: torch.Tensor) -> np.ndarray:
        predicted_block_maps: np.ndarray = np.vectorize(self.reverse_unique_blocks_dict.get)(x.argmax(dim=1).numpy())
        return predicted_block_maps

    def generate_samples(self, model: UNetModel, model_name: str, timestep: int) -> None:  # type: ignore[no-any-unimported]
        memory_train = self.training

        self.eval()
        model_ = copy.deepcopy(model)
        node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
        with torch.no_grad():
            traj = node_.trajectory(
                torch.randn((1, len(self.unique_blocks_dict), 16, 16, 16), device=self.device),
                t_span=torch.linspace(0, 1, 100, device=self.device),
            )
        for time_step in range(traj.shape[0]):
            post_processed = self.post_process(traj[-1])
            np.save(f"sample_{time_step}.npy", post_processed, allow_pickle=True)

        self.train(memory_train)

    def training_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> None:
        self.step(batch, batch_idx, "train")
        self.step_number += 1
        if self.step_number % 1 == 0:
            logger.info(f"Generating samples at step {self.step_number}...")
            self.generate_samples(self.unet_model, "unet", self.step_number)
            self.generate_samples(self.ema_model, "ema", self.step_number)
            logger.info(f"Saving model at step {self.step_number}...")
            torch.save(
                {
                    "net_model": self.unet_model,
                    "ema_model": self.ema_model,
                    "step": self.step_number,
                },
                self.save_dir + f"/model_{self.step_number}.pth",
            )

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=2e-4)

    def on_train_start(self) -> None:
        print(self)
