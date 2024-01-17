from unittest.mock import patch, MagicMock

# import numpy as np

# from minecraft_copilot_ml.train import MinecraftSchematicsDataset


class TestMinecraftSchematicsDataset:
    @patch("minecraft_copilot_ml.train.requests.get")
    @patch("minecraft_copilot_ml.train.open")
    @patch("minecraft_copilot_ml.train.os.listdir")
    def test___init__(self, mock_get: MagicMock, mock_open: MagicMock, mock_listdir: MagicMock) -> None:
        # Given
        # list_files = [
        #     "minecraft_copilot_ml/data/destroyed_25/destroyed_25_0.npy",
        # ]
        # one_hot_dict = {"air": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), "stone": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])}
        # When
        # dataset = MinecraftSchematicsDataset(list_files, one_hot_dict)
        # Then
        # assert dataset.list_16_16_16 == ["destroyed_25_0.npy"]
        # assert dataset.list_16_16_16_destroyed_25 == ["destroyed_25_0.npy"]
        # assert dataset.list_16_16_16_destroyed_50 == []
        # assert dataset.list_16_16_16_destroyed_75 == []
        # assert dataset.one_hot_dict == one_hot_dict
        pass

    def test_get_unique_values(self) -> None:
        pass

    def test___len__(self) -> None:
        pass

    def test___getitem__(self) -> None:
        pass
