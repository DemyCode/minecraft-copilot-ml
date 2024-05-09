# flake8: noqa: E203
import os
from typing import Any, Dict, Optional, Tuple

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from improved_diffusion.unet import UNetModel  # type: ignore[import-untyped]
from loguru import logger
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
from torchcfm import ExactOptimalTransportConditionalFlowMatcher  # type: ignore[import-untyped]

from minecraft_copilot_ml.data_loader import MinecraftSchematicsDatasetItemType


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
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        self.step_number = 0
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.automatic_optimization = False

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet_model(xt, t)  # type: ignore[no-any-return]

    def step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int, mode: str) -> None:
        optimizer = self.trainer.optimizers[0]
        optimizer.zero_grad()
        for param in self.parameters():
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        for param in self.unet_model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        block_maps, block_map_masks = batch
        pre_processed_block_maps = self.pre_process(block_maps)
        tensor_block_map_masks = torch.from_numpy(block_map_masks).float().to(self.device).long()
        tensor_block_map_masks_for_one_hot = torch.zeros(
            (len(self.unique_blocks_dict), block_maps.shape[0], 16, 16, 16)
        ).to(self.device)
        tensor_block_map_masks_for_one_hot[:, tensor_block_map_masks == 1] = 1
        tensor_block_map_masks_for_one_hot = tensor_block_map_masks_for_one_hot.permute(1, 0, 2, 3, 4)
        x1 = pre_processed_block_maps
        x0 = torch.randn_like(x1)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = self(xt, t)
        loss = (ut - vt) ** 2
        loss = loss * tensor_block_map_masks_for_one_hot  # Mask out the loss for the blocks outside the schematic
        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

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
        torch.cuda.empty_cache()
        for tensor in [
            pre_processed_block_maps,
            tensor_block_map_masks,
            tensor_block_map_masks_for_one_hot,
            x1,
            x0,
            t,
            xt,
            ut,
            vt,
            loss,
        ]:
            tensor.detach()
            del tensor

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
        predicted_block_maps: np.ndarray = np.vectorize(self.reverse_unique_blocks_dict.get)(
            x.argmax(dim=1).cpu().numpy()
        )
        return predicted_block_maps

    def generate_samples(self, model: UNetModel, model_name: str) -> None:  # type: ignore[no-any-unimported]
        memory_train = model.training

        model.eval()

        def vector_field(t: float, x: np.ndarray) -> np.ndarray:
            reshaped_x = x.reshape(1, len(self.unique_blocks_dict), 16, 16, 16)
            x_tensor = torch.from_numpy(reshaped_x).float().to(self.device)
            res = model(x_tensor, torch.tensor([t], device=self.device).float())
            return res.detach().cpu().numpy().reshape(-1)  # type: ignore[no-any-return]

        traj = solve_ivp(
            fun=vector_field,
            t_span=(0, 1),
            y0=np.random.standard_normal((1, len(self.unique_blocks_dict), 16, 16, 16)).reshape(-1),
            t_eval=np.linspace(0, 1, 10),
        )
        sol = traj["y"].transpose(1, 0)
        for time_step in range(sol.shape[0]):
            post_processed = self.post_process(
                torch.from_numpy(sol[time_step].reshape(1, len(self.unique_blocks_dict), 16, 16, 16))
            )
            np.save(f"{self.save_dir}/sample_{model_name}_{time_step}.npy", post_processed, allow_pickle=True)
        self.train(memory_train)

    def training_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> None:
        self.step(batch, batch_idx, "train")
        self.log("step", self.step_number, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if self.step_number % 20_000 == 0:
            logger.info(f"Generating samples at step {self.step_number}...")
            self.generate_samples(self.unet_model, f"unet_{self.step_number}")
            logger.info(f"Saving model at step {self.step_number}...")
            torch.save(
                {
                    "net_model": self.unet_model,
                    "step": self.step_number,
                },
                self.save_dir + f"/model_{self.step_number}.pth",
            )
        self.step_number += 1

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
