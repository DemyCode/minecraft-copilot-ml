from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import mlflow
import boto3
import re
from minecraft_copilot_ml.data_loader import MinecraftSchematicsDataset

from minecraft_copilot_ml.model import UNet3D

if __name__ == "__main__":
    with mlflow.start_run():
        N_UNIQUE_MINECRAFT_BLOCKS = 877
        EPOCHS = 20
        model = UNet3D(N_UNIQUE_MINECRAFT_BLOCKS)
        if torch.cuda.is_available():
            model = model.cuda()

        mlflow.log_param("n_unique_minecraft_blocks", N_UNIQUE_MINECRAFT_BLOCKS)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("n_unique_minecraft_blocks", N_UNIQUE_MINECRAFT_BLOCKS)
        mlflow.log_param("model", str(model))

        # Download the right data files
        s3_client = boto3.client("s3")
        list_regex = [r".*"]
        list_all_data_files = s3_client.list_objects(Bucket="minecraft-copilot-ml")
        list_all_data_files_names = [x["Key"] for x in list_all_data_files["Contents"]]
        list_all_filtered_data_files_names = [
            file for file in list_all_data_files_names if any(re.match(regex, file) for regex in list_regex)
        ]
        for file in list_all_filtered_data_files_names:
            s3_client.download_file("minecraft-copilot-ml", file, f"./minecraft_copilot_ml/data/{file}")

        file_list = []

        train_files, test_files = train_test_split(file_list, test_size=0.2)
        train_dataset = MinecraftSchematicsDataset(train_files)
        test_dataset = MinecraftSchematicsDataset(test_files)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in tqdm(range(EPOCHS)):
            mlflow.log_metric("epoch", epoch)
            model.train()
            training_loss = 0
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                training_loss += loss.item()
                optimizer.step()
            mlflow.log_metric("training_loss", training_loss)
            model.eval()
            total_validation_loss = 0
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                total_validation_loss += loss.item()
            print(f"End of epoch {epoch}: loss = {total_validation_loss:.2f}")
        print("Done training")
        torch.save(model.state_dict(), "minecraft_copilot_ml/model.pth")
