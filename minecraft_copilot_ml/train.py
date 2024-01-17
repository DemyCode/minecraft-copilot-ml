from typing import Any, Set, Tuple
from loguru import logger
import mlflow  # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import torch
from sklearn.preprocessing import OneHotEncoder  # type: ignore
import pytorch_lightning as pl
import json

from minecraft_copilot_ml.environment import settings
from minecraft_copilot_ml.model import UNet3D


class MinecraftSchematicsDataset(Dataset):
    def __init__(  # type: ignore
        self,
        X_files: list[str],
        y_files: list[str],
        one_hot_encoder: OneHotEncoder,
    ) -> None:
        self.x_files = X_files
        self.y_files = y_files
        self.one_hot_encoder = one_hot_encoder

    def __len__(self) -> int:
        return len(self.x_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_file = self.x_files[idx]
        y_file = self.y_files[idx]
        x_array: np.ndarray = np.load(x_file, allow_pickle=True)
        y_array: np.ndarray = np.load(y_file, allow_pickle=True)
        x_array = self.one_hot_encoder.transform(x_array.reshape(-1, 1)).reshape(16, 16, 16, -1).transpose(3, 0, 1, 2)
        y_array = self.one_hot_encoder.transform(y_array.reshape(-1, 1)).reshape(16, 16, 16, -1).transpose(3, 0, 1, 2)
        return torch.from_numpy(x_array).float(), torch.from_numpy(y_array).float()


def main() -> None:
    if os.environ.get("MLFLOW_TRACKING_URI") is None:
        logger.error("MLFLOW_TRACKING_URI is not set")
        raise Exception()
    X_HOME = settings.X_HOME
    Y_HOME = settings.Y_HOME
    # Download the right data files

    full_list_of_files_X = []
    for root, _, files in os.walk(X_HOME):
        for file in files:
            full_list_of_files_X.append(os.path.join(root, file))
    full_list_of_files_Y = []
    for root, _, files in os.walk(Y_HOME):
        for file in files:
            full_list_of_files_Y.append(os.path.join(root, file))
    full_list_of_files = full_list_of_files_X + full_list_of_files_Y

    # Generating one hot encoding
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    unique_values_set: Set[Any] = set()
    # only list on X files because y files are the same thing but with destroyed blocks
    for file in tqdm(full_list_of_files, desc="Generating one hot encoding", smoothing=0):
        array_16_16_16: np.ndarray = np.load(file, allow_pickle=True)
        unique_values_set = unique_values_set.union(set(array_16_16_16.flatten()))
    if "None" in unique_values_set:
        unique_values_set.remove("None")
    if None in unique_values_set:
        unique_values_set.remove(None)
    unique_values_list = list(unique_values_set)
    one_hot_encoder.fit(np.array(unique_values_list).reshape(-1, 1))

    BATCH_SIZE = 32
    EPOCHS = 20

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(
        full_list_of_files_X, full_list_of_files_Y, test_size=0.2, random_state=42
    )
    logger.info(f"Initializing model with {len(unique_values_set)} classes")
    model = UNet3D(len(unique_values_set))
    used_cpus = 1
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        used_cpus = cpu_count // 2
        logger.info(f"Using {used_cpus} cpus")

    train_dataset = MinecraftSchematicsDataset(X_train, y_train, one_hot_encoder)
    test_dataset = MinecraftSchematicsDataset(X_test, y_test, one_hot_encoder)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=used_cpus,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE // 2,
        shuffle=False,
        num_workers=used_cpus,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
    )
    mlflow.pytorch.autolog()
    experiment_id = mlflow.create_experiment("minecraft_copilot_ml")
    with mlflow.start_run(experiment_id=experiment_id):
        with open("categories.json", "w") as f:
            json.dump(one_hot_encoder.categories_[0].tolist(), f)
        mlflow.log_artifact("categories.json")
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":  # pragma: no cover
    main()
