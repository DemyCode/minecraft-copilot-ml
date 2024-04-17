# flake8: noqa: E203
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from minecraft_copilot_ml.data_loader import MinecraftSchematicsDatasetItemType

from torchcfm.models.unet import UNetModel  # type: ignore[import-untyped]
from torchcfm import ExactOptimalTransportConditionalFlowMatcher  # type: ignore[import-untyped]


class LightningUNetModel(pl.LightningModule):
    def __init__(self, model: UNetModel, unique_blocks_dict: Dict[str, int]) -> None:  # type: ignore[no-any-unimported]
        super(LightningUNetModel, self).__init__()
        self.model = model
        self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        self.unique_blocks_dict = unique_blocks_dict

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return self.model(t, x, y=y)  # type: ignore[no-any-return]

    def step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int, mode: str) -> torch.Tensor:
        block_maps, _, block_map_masks, _ = batch
        tensor_block_map_masks = (
            torch.from_numpy(block_map_masks).float().to("cuda" if torch.cuda.is_available() else "cpu")
        )
        tensor_block_map_masks_for_one_hot = torch.zeros(
            (len(self.unique_blocks_dict), block_maps.shape[0], 16, 16, 16)
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        tensor_block_map_masks_for_one_hot[:, tensor_block_map_masks == 1] = 1
        tensor_block_map_masks_for_one_hot = tensor_block_map_masks_for_one_hot.permute(1, 0, 2, 3, 4)
        # print(tensor_block_map_masks_for_one_hot.shape)
        pre_processed_block_maps = self.pre_process(block_maps)
        x1 = pre_processed_block_maps
        x0 = torch.randn_like(x1)
        sample_location_and_conditional_flow: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            (self.flow_matcher.sample_location_and_conditional_flow(x0, x1)),
        )
        t, xt, ut = sample_location_and_conditional_flow
        vt = self(t, xt)
        loss = (vt - ut) ** 2
        loss = loss * tensor_block_map_masks_for_one_hot
        loss = loss.sum()
        loss_dict = {
            "loss": loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
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
        return loss  # type: ignore[no-any-return]

    def pre_process(self, x: np.ndarray) -> torch.Tensor:
        vectorized_x = np.vectorize(lambda x: self.unique_blocks_dict.get(x, self.unique_blocks_dict["minecraft:air"]))(
            x
        )
        vectorized_x = vectorized_x.astype(np.int64)
        x_tensor = torch.from_numpy(vectorized_x)
        x_tensor = x_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.nn.functional.one_hot(x_tensor, num_classes=len(self.unique_blocks_dict))
        x_tensor = x_tensor.permute(0, 4, 1, 2, 3).float()
        return x_tensor

    def training_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }

    def on_train_start(self) -> None:
        print(self)
