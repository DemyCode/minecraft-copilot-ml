from unittest.mock import MagicMock, patch
import numpy as np

from minecraft_copilot_ml.data_loader import get_random_block_map, list_files_of_s3


@patch("minecraft_copilot_ml.data_loader.requests.get")
def test_list_files_of_s3(mock_get: MagicMock) -> None:
    # Given
    s3_public_link = "https://minecraft-schematics-raw.s3.amazonaws.com"
    params = {"list-type": "2"}

    # When
    list_of_files = list_files_of_s3(
        s3_public_link=s3_public_link,
        params=params,
    )

    # Then
    assert len(list_of_files) > 0


def test_get_random_block_map() -> None:
    # Given
    block_map = np.arange(0, 3**3).reshape(3, 3, 3)

    # When
    random_block_map = get_random_block_map(block_map)

    # Then
    assert random_block_map.shape == (16, 16, 16)
