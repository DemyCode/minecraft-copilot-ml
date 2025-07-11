import nbtlib
import numpy as np
import os

# Classic Minecraft block ID to name mapping (for older schematic formats)
# This is a partial mapping of common blocks - you may need to extend this
BLOCK_ID_TO_NAME = {
    0: "minecraft:air",
    1: "minecraft:stone",
    2: "minecraft:grass_block",
    3: "minecraft:dirt",
    4: "minecraft:cobblestone",
    5: "minecraft:oak_planks",
    6: "minecraft:sapling",
    7: "minecraft:bedrock",
    8: "minecraft:water",
    9: "minecraft:water",
    10: "minecraft:lava",
    11: "minecraft:lava",
    12: "minecraft:sand",
    13: "minecraft:gravel",
    14: "minecraft:gold_ore",
    15: "minecraft:iron_ore",
    16: "minecraft:coal_ore",
    17: "minecraft:oak_log",
    18: "minecraft:oak_leaves",
    19: "minecraft:sponge",
    20: "minecraft:glass",
    21: "minecraft:lapis_ore",
    22: "minecraft:lapis_block",
    23: "minecraft:dispenser",
    24: "minecraft:sandstone",
    25: "minecraft:note_block",
    26: "minecraft:bed",
    27: "minecraft:powered_rail",
    28: "minecraft:detector_rail",
    29: "minecraft:sticky_piston",
    30: "minecraft:cobweb",
    31: "minecraft:grass",
    32: "minecraft:dead_bush",
    33: "minecraft:piston",
    34: "minecraft:piston_head",
    35: "minecraft:white_wool",
    36: "minecraft:piston_extension",
    37: "minecraft:dandelion",
    38: "minecraft:poppy",
    39: "minecraft:brown_mushroom",
    40: "minecraft:red_mushroom",
    41: "minecraft:gold_block",
    42: "minecraft:iron_block",
    43: "minecraft:stone_slab",
    44: "minecraft:stone_slab",
    45: "minecraft:bricks",
    46: "minecraft:tnt",
    47: "minecraft:bookshelf",
    48: "minecraft:mossy_cobblestone",
    49: "minecraft:obsidian",
    50: "minecraft:torch",
    51: "minecraft:fire",
    52: "minecraft:spawner",
    53: "minecraft:oak_stairs",
    54: "minecraft:chest",
    55: "minecraft:redstone_wire",
    56: "minecraft:diamond_ore",
    57: "minecraft:diamond_block",
    58: "minecraft:crafting_table",
    59: "minecraft:wheat",
    60: "minecraft:farmland",
    61: "minecraft:furnace",
    62: "minecraft:furnace",
    63: "minecraft:sign",
    64: "minecraft:oak_door",
    65: "minecraft:ladder",
    66: "minecraft:rail",
    67: "minecraft:cobblestone_stairs",
    68: "minecraft:wall_sign",
    69: "minecraft:lever",
    70: "minecraft:stone_pressure_plate",
    71: "minecraft:iron_door",
    72: "minecraft:wooden_pressure_plate",
    73: "minecraft:redstone_ore",
    74: "minecraft:redstone_ore",
    75: "minecraft:redstone_torch",
    76: "minecraft:redstone_torch",
    77: "minecraft:stone_button",
    78: "minecraft:snow",
    79: "minecraft:ice",
    80: "minecraft:snow_block",
    81: "minecraft:cactus",
    82: "minecraft:clay",
    83: "minecraft:sugar_cane",
    84: "minecraft:jukebox",
    85: "minecraft:oak_fence",
    86: "minecraft:pumpkin",
    87: "minecraft:netherrack",
    88: "minecraft:soul_sand",
    89: "minecraft:glowstone",
    90: "minecraft:nether_portal",
    91: "minecraft:jack_o_lantern",
    92: "minecraft:cake",
    93: "minecraft:repeater",
    94: "minecraft:repeater",
    95: "minecraft:white_stained_glass",
    96: "minecraft:trapdoor",
    97: "minecraft:infested_stone",
    98: "minecraft:stone_bricks",
    99: "minecraft:brown_mushroom_block",
    100: "minecraft:red_mushroom_block",
    101: "minecraft:iron_bars",
    102: "minecraft:glass_pane",
    103: "minecraft:melon",
    104: "minecraft:pumpkin_stem",
    105: "minecraft:melon_stem",
    106: "minecraft:vine",
    107: "minecraft:oak_fence_gate",
    108: "minecraft:brick_stairs",
    109: "minecraft:stone_brick_stairs",
    110: "minecraft:mycelium",
    111: "minecraft:lily_pad",
    112: "minecraft:nether_bricks",
    113: "minecraft:nether_brick_fence",
    114: "minecraft:nether_brick_stairs",
    115: "minecraft:nether_wart",
    116: "minecraft:enchanting_table",
    117: "minecraft:brewing_stand",
    118: "minecraft:cauldron",
    119: "minecraft:end_portal",
    120: "minecraft:end_portal_frame",
    121: "minecraft:end_stone",
    122: "minecraft:dragon_egg",
    123: "minecraft:redstone_lamp",
    124: "minecraft:redstone_lamp",
    125: "minecraft:oak_slab",
    126: "minecraft:oak_slab",
    127: "minecraft:cocoa",
    128: "minecraft:sandstone_stairs",
    129: "minecraft:emerald_ore",
    130: "minecraft:ender_chest",
    131: "minecraft:tripwire_hook",
    132: "minecraft:tripwire",
    133: "minecraft:emerald_block",
    134: "minecraft:spruce_stairs",
    135: "minecraft:birch_stairs",
    136: "minecraft:jungle_stairs",
    137: "minecraft:command_block",
    138: "minecraft:beacon",
    139: "minecraft:cobblestone_wall",
    140: "minecraft:flower_pot",
    141: "minecraft:carrots",
    142: "minecraft:potatoes",
    143: "minecraft:wooden_button",
    144: "minecraft:skeleton_skull",
    145: "minecraft:anvil",
    146: "minecraft:trapped_chest",
    147: "minecraft:light_weighted_pressure_plate",
    148: "minecraft:heavy_weighted_pressure_plate",
    149: "minecraft:comparator",
    150: "minecraft:comparator",
    151: "minecraft:daylight_detector",
    152: "minecraft:redstone_block",
    153: "minecraft:nether_quartz_ore",
    154: "minecraft:hopper",
    155: "minecraft:quartz_block",
    156: "minecraft:quartz_stairs",
    157: "minecraft:activator_rail",
    158: "minecraft:dropper",
    159: "minecraft:white_terracotta",
    160: "minecraft:white_stained_glass_pane",
    161: "minecraft:acacia_leaves",
    162: "minecraft:acacia_log",
    163: "minecraft:acacia_stairs",
    164: "minecraft:dark_oak_stairs",
    165: "minecraft:slime_block",
    166: "minecraft:barrier",
    167: "minecraft:iron_trapdoor",
    168: "minecraft:prismarine",
    169: "minecraft:sea_lantern",
    170: "minecraft:hay_block",
    171: "minecraft:white_carpet",
    172: "minecraft:terracotta",
    173: "minecraft:coal_block",
    174: "minecraft:packed_ice",
    175: "minecraft:sunflower",
    176: "minecraft:white_banner",
    177: "minecraft:wall_banner",
    178: "minecraft:daylight_detector",
    179: "minecraft:red_sandstone",
    180: "minecraft:red_sandstone_stairs",
    181: "minecraft:red_sandstone_slab",
    182: "minecraft:red_sandstone_slab",
    183: "minecraft:spruce_fence_gate",
    184: "minecraft:birch_fence_gate",
    185: "minecraft:jungle_fence_gate",
    186: "minecraft:dark_oak_fence_gate",
    187: "minecraft:acacia_fence_gate",
    188: "minecraft:spruce_fence",
    189: "minecraft:birch_fence",
    190: "minecraft:jungle_fence",
    191: "minecraft:dark_oak_fence",
    192: "minecraft:acacia_fence",
    193: "minecraft:spruce_door",
    194: "minecraft:birch_door",
    195: "minecraft:jungle_door",
    196: "minecraft:acacia_door",
    197: "minecraft:dark_oak_door",
    198: "minecraft:end_rod",
    199: "minecraft:chorus_plant",
    200: "minecraft:chorus_flower",
    201: "minecraft:purpur_block",
    202: "minecraft:purpur_pillar",
    203: "minecraft:purpur_stairs",
    204: "minecraft:purpur_slab",
    205: "minecraft:purpur_slab",
    206: "minecraft:end_stone_bricks",
    207: "minecraft:beetroots",
    208: "minecraft:grass_path",
    209: "minecraft:end_gateway",
    210: "minecraft:repeating_command_block",
    211: "minecraft:chain_command_block",
    212: "minecraft:frosted_ice",
    213: "minecraft:magma_block",
    214: "minecraft:nether_wart_block",
    215: "minecraft:red_nether_bricks",
    216: "minecraft:bone_block",
    217: "minecraft:structure_void",
    218: "minecraft:observer",
    219: "minecraft:white_shulker_box",
    220: "minecraft:orange_shulker_box",
    221: "minecraft:magenta_shulker_box",
    222: "minecraft:light_blue_shulker_box",
    223: "minecraft:yellow_shulker_box",
    224: "minecraft:lime_shulker_box",
    225: "minecraft:pink_shulker_box",
    226: "minecraft:gray_shulker_box",
    227: "minecraft:light_gray_shulker_box",
    228: "minecraft:cyan_shulker_box",
    229: "minecraft:purple_shulker_box",
    230: "minecraft:blue_shulker_box",
    231: "minecraft:brown_shulker_box",
    232: "minecraft:green_shulker_box",
    233: "minecraft:red_shulker_box",
    234: "minecraft:black_shulker_box",
    235: "minecraft:white_glazed_terracotta",
    236: "minecraft:orange_glazed_terracotta",
    237: "minecraft:magenta_glazed_terracotta",
    238: "minecraft:light_blue_glazed_terracotta",
    239: "minecraft:yellow_glazed_terracotta",
    240: "minecraft:lime_glazed_terracotta",
    241: "minecraft:pink_glazed_terracotta",
    242: "minecraft:gray_glazed_terracotta",
    243: "minecraft:light_gray_glazed_terracotta",
    244: "minecraft:cyan_glazed_terracotta",
    245: "minecraft:purple_glazed_terracotta",
    246: "minecraft:blue_glazed_terracotta",
    247: "minecraft:brown_glazed_terracotta",
    248: "minecraft:green_glazed_terracotta",
    249: "minecraft:red_glazed_terracotta",
    250: "minecraft:black_glazed_terracotta",
    251: "minecraft:white_concrete",
    252: "minecraft:white_concrete_powder",
    255: "minecraft:structure_block",
}

def load_schematic_to_numpy(file_path):
    """
    Load a Minecraft schematic file and convert it to a NumPy array with block names.
    
    Args:
        file_path (str): Path to the schematic file
        
    Returns:
        tuple: (blocks_array, dimensions)
            - blocks_array: NumPy array of shape (height, length, width) with block names
            - dimensions: Dictionary with 'height', 'length', 'width' values
    """
    # Load the schematic file
    nbt_file = nbtlib.load(file_path)
    
    # Get dimensions
    height = nbt_file.get('Height', 0)
    length = nbt_file.get('Length', 0)
    width = nbt_file.get('Width', 0)
    
    dimensions = {
        'height': height,
        'length': length,
        'width': width
    }
    
    # Check if this is a newer format with a Palette
    if 'Palette' in nbt_file:
        # For newer format (1.13+)
        palette = {v: k for k, v in nbt_file['Palette'].items()}
        blocks_data = np.array(nbt_file['BlockData'])
        
        # Reshape the blocks data
        blocks_array = np.zeros((height, length, width), dtype=object)
        for y in range(height):
            for z in range(length):
                for x in range(width):
                    index = y * length * width + z * width + x
                    if index < len(blocks_data):
                        block_id = blocks_data[index]
                        blocks_array[y, z, x] = palette.get(block_id, "minecraft:air")
    else:
        # For older format (pre-1.13)
        blocks_data = np.array(nbt_file['Blocks'])
        
        # Reshape the blocks data
        blocks_array = np.zeros((height, length, width), dtype=object)
        for y in range(height):
            for z in range(length):
                for x in range(width):
                    index = y * length * width + z * width + x
                    if index < len(blocks_data):
                        block_id = blocks_data[index]
                        blocks_array[y, z, x] = BLOCK_ID_TO_NAME.get(block_id, f"unknown:{block_id}")
    
    return blocks_array, dimensions

# Example usage
if __name__ == "__main__":
    # Load a sample schematic file
    sample_file = os.path.join('minecraft-schematics-raw', '1.schematic')
    blocks, dims = load_schematic_to_numpy(sample_file)
    
    print(f"Loaded schematic with dimensions: {dims}")
    print(f"Blocks array shape: {blocks.shape}")
    
    # Print a slice of the blocks array to see some block names
    print("\nSample of blocks (height=0, first 5x5):")
    print(blocks[0, :5, :5])
    
    # Count unique blocks
    unique_blocks, counts = np.unique(blocks, return_counts=True)
    print(f"\nFound {len(unique_blocks)} unique block types")
    
    # Print the most common blocks
    block_counts = sorted(zip(unique_blocks, counts), key=lambda x: x[1], reverse=True)
    print("\nMost common blocks:")
    for block, count in block_counts[:10]:
        print(f"{block}: {count}")