from minecraft_copilot_ml.processing_16 import (
    convert_all_nbts_to_numpy_array_with_minecraft_ids,
    convert_nbt_to_numpy_array_with_minecraft_ids,
    list_of_forbidden_files,
)


def test_list_of_forbidden_files() -> None:
    assert list_of_forbidden_files == [
        "14281.schematic",
        "12188.schematic",
        "8197.schematic",
        "576.schematic",
        "3322.schematic",
        "243.schematic",
        "13197.schematic",
        "15716.schem",
        "11351.schematic",
        "11314.schematic",
        "14846.schem",
        "9171.schematic",
        "13441.schematic",
        "15111.schem",
        "452.schematic",
        "1924.schematic",
    ]


def test_convert_nbt_to_numpy_array_with_minecraft_ids() -> None:
    # When
    convert_nbt_to_numpy_array_with_minecraft_ids("minecraft-schematics-16/1.schematic")
    # Then
    # assert result.shape == (16, 16, 16)
    # assert result[0, 0, 0] == "minecraft:air"
    # assert result[0, 0, 1] == "minecraft:air"


# @patch()
def test_convert_all_nbts_to_numpy_array_with_minecraft_ids() -> None:
    convert_all_nbts_to_numpy_array_with_minecraft_ids()
