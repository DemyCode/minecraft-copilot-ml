import numpy as np
import nbtlib

BLOCK_ID_TO_NAME = {
    0: "minecraft:air", 1: "minecraft:stone", 2: "minecraft:grass_block",
    3: "minecraft:dirt", 4: "minecraft:cobblestone", 5: "minecraft:oak_planks",
    6: "minecraft:oak_sapling", 7: "minecraft:bedrock", 8: "minecraft:water",
    9: "minecraft:water", 10: "minecraft:lava", 11: "minecraft:lava",
    12: "minecraft:sand", 13: "minecraft:gravel", 14: "minecraft:gold_ore",
    15: "minecraft:iron_ore", 16: "minecraft:coal_ore", 17: "minecraft:oak_log",
    18: "minecraft:oak_leaves", 19: "minecraft:sponge", 20: "minecraft:glass",
    21: "minecraft:lapis_ore", 22: "minecraft:lapis_block", 23: "minecraft:dispenser",
    24: "minecraft:sandstone", 25: "minecraft:note_block", 26: "minecraft:bed",
    27: "minecraft:powered_rail", 28: "minecraft:detector_rail", 29: "minecraft:sticky_piston",
    30: "minecraft:cobweb", 31: "minecraft:grass", 32: "minecraft:dead_bush",
    33: "minecraft:piston", 34: "minecraft:piston_head", 35: "minecraft:white_wool",
    37: "minecraft:dandelion", 38: "minecraft:poppy", 39: "minecraft:brown_mushroom",
    40: "minecraft:red_mushroom", 41: "minecraft:gold_block", 42: "minecraft:iron_block",
    43: "minecraft:smooth_stone", 44: "minecraft:stone_slab", 45: "minecraft:bricks",
    46: "minecraft:tnt", 47: "minecraft:bookshelf", 48: "minecraft:mossy_cobblestone",
    49: "minecraft:obsidian", 50: "minecraft:torch", 51: "minecraft:fire",
    52: "minecraft:spawner", 53: "minecraft:oak_stairs", 54: "minecraft:chest",
    55: "minecraft:redstone_wire", 56: "minecraft:diamond_ore", 57: "minecraft:diamond_block",
    58: "minecraft:crafting_table", 59: "minecraft:wheat", 60: "minecraft:farmland",
    61: "minecraft:furnace", 62: "minecraft:furnace", 63: "minecraft:sign",
    64: "minecraft:oak_door", 65: "minecraft:ladder", 66: "minecraft:rail",
    67: "minecraft:cobblestone_stairs", 68: "minecraft:wall_sign", 69: "minecraft:lever",
    70: "minecraft:stone_pressure_plate", 71: "minecraft:iron_door",
    72: "minecraft:wooden_pressure_plate", 73: "minecraft:redstone_ore",
    74: "minecraft:redstone_ore", 75: "minecraft:redstone_torch",
    76: "minecraft:redstone_torch", 77: "minecraft:stone_button", 78: "minecraft:snow",
    79: "minecraft:ice", 80: "minecraft:snow_block", 81: "minecraft:cactus",
    82: "minecraft:clay", 83: "minecraft:sugar_cane", 84: "minecraft:jukebox",
    85: "minecraft:oak_fence", 86: "minecraft:pumpkin", 87: "minecraft:netherrack",
    88: "minecraft:soul_sand", 89: "minecraft:glowstone", 90: "minecraft:nether_portal",
    91: "minecraft:jack_o_lantern", 92: "minecraft:cake", 93: "minecraft:repeater",
    94: "minecraft:repeater", 95: "minecraft:white_stained_glass",
    96: "minecraft:trapdoor", 97: "minecraft:infested_stone", 98: "minecraft:stone_bricks",
    99: "minecraft:brown_mushroom_block", 100: "minecraft:red_mushroom_block",
    101: "minecraft:iron_bars", 102: "minecraft:glass_pane", 103: "minecraft:melon",
    104: "minecraft:pumpkin_stem", 105: "minecraft:melon_stem", 106: "minecraft:vine",
    107: "minecraft:oak_fence_gate", 108: "minecraft:brick_stairs",
    109: "minecraft:stone_brick_stairs", 110: "minecraft:mycelium",
    111: "minecraft:lily_pad", 112: "minecraft:nether_bricks",
    113: "minecraft:nether_brick_fence", 114: "minecraft:nether_brick_stairs",
    115: "minecraft:nether_wart", 116: "minecraft:enchanting_table",
    117: "minecraft:brewing_stand", 118: "minecraft:cauldron",
    119: "minecraft:end_portal", 120: "minecraft:end_portal_frame",
    121: "minecraft:end_stone", 122: "minecraft:dragon_egg",
    123: "minecraft:redstone_lamp", 124: "minecraft:redstone_lamp",
    125: "minecraft:oak_slab", 126: "minecraft:oak_slab", 127: "minecraft:cocoa",
    128: "minecraft:sandstone_stairs", 129: "minecraft:emerald_ore",
    130: "minecraft:ender_chest", 131: "minecraft:tripwire_hook",
    132: "minecraft:tripwire", 133: "minecraft:emerald_block",
    134: "minecraft:spruce_stairs", 135: "minecraft:birch_stairs",
    136: "minecraft:jungle_stairs", 137: "minecraft:command_block",
    138: "minecraft:beacon", 139: "minecraft:cobblestone_wall",
    140: "minecraft:flower_pot", 141: "minecraft:carrots", 142: "minecraft:potatoes",
    143: "minecraft:wooden_button", 144: "minecraft:skeleton_skull",
    145: "minecraft:anvil", 146: "minecraft:trapped_chest",
    147: "minecraft:light_weighted_pressure_plate",
    148: "minecraft:heavy_weighted_pressure_plate", 149: "minecraft:comparator",
    150: "minecraft:comparator", 151: "minecraft:daylight_detector",
    152: "minecraft:redstone_block", 153: "minecraft:nether_quartz_ore",
    154: "minecraft:hopper", 155: "minecraft:quartz_block",
    156: "minecraft:quartz_stairs", 157: "minecraft:activator_rail",
    158: "minecraft:dropper", 159: "minecraft:white_terracotta",
    160: "minecraft:white_stained_glass_pane", 161: "minecraft:acacia_leaves",
    162: "minecraft:acacia_log", 163: "minecraft:acacia_stairs",
    164: "minecraft:dark_oak_stairs", 165: "minecraft:slime_block",
    166: "minecraft:barrier", 167: "minecraft:iron_trapdoor",
    168: "minecraft:prismarine", 169: "minecraft:sea_lantern",
    170: "minecraft:hay_block", 171: "minecraft:white_carpet",
    172: "minecraft:terracotta", 173: "minecraft:coal_block",
    174: "minecraft:packed_ice", 175: "minecraft:sunflower",
    176: "minecraft:white_banner", 177: "minecraft:wall_banner",
    178: "minecraft:daylight_detector", 179: "minecraft:red_sandstone",
    180: "minecraft:red_sandstone_stairs", 181: "minecraft:red_sandstone_slab",
    182: "minecraft:red_sandstone_slab", 183: "minecraft:spruce_fence_gate",
    184: "minecraft:birch_fence_gate", 185: "minecraft:jungle_fence_gate",
    186: "minecraft:dark_oak_fence_gate", 187: "minecraft:acacia_fence_gate",
    188: "minecraft:spruce_fence", 189: "minecraft:birch_fence",
    190: "minecraft:jungle_fence", 191: "minecraft:dark_oak_fence",
    192: "minecraft:acacia_fence", 193: "minecraft:spruce_door",
    194: "minecraft:birch_door", 195: "minecraft:jungle_door",
    196: "minecraft:acacia_door", 197: "minecraft:dark_oak_door",
    198: "minecraft:end_rod", 199: "minecraft:chorus_plant",
    200: "minecraft:chorus_flower", 201: "minecraft:purpur_block",
    202: "minecraft:purpur_pillar", 203: "minecraft:purpur_stairs",
    204: "minecraft:purpur_slab", 205: "minecraft:purpur_slab",
    206: "minecraft:end_stone_bricks", 207: "minecraft:beetroots",
    208: "minecraft:grass_path", 209: "minecraft:end_gateway",
    210: "minecraft:repeating_command_block", 211: "minecraft:chain_command_block",
    212: "minecraft:frosted_ice", 213: "minecraft:magma_block",
    214: "minecraft:nether_wart_block", 215: "minecraft:red_nether_bricks",
    216: "minecraft:bone_block", 217: "minecraft:structure_void",
    218: "minecraft:observer", 251: "minecraft:white_concrete",
    252: "minecraft:white_concrete_powder", 255: "minecraft:structure_block",
}

_COLORS = [
    "white", "orange", "magenta", "light_blue", "yellow", "lime",
    "pink", "gray", "light_gray", "cyan", "purple", "blue",
    "brown", "green", "red", "black",
]

def _colored(base, data):
    return {(data_val, f"minecraft:{color}_{base}") for data_val, color in enumerate(_COLORS)}

BLOCK_ID_DATA_TO_NAME = {
    # Stone variants
    (1, 1): "minecraft:granite", (1, 2): "minecraft:polished_granite",
    (1, 3): "minecraft:diorite", (1, 4): "minecraft:polished_diorite",
    (1, 5): "minecraft:andesite", (1, 6): "minecraft:polished_andesite",
    # Dirt
    (3, 1): "minecraft:coarse_dirt", (3, 2): "minecraft:podzol",
    # Planks
    (5, 1): "minecraft:spruce_planks", (5, 2): "minecraft:birch_planks",
    (5, 3): "minecraft:jungle_planks", (5, 4): "minecraft:acacia_planks",
    (5, 5): "minecraft:dark_oak_planks",
    # Saplings
    (6, 1): "minecraft:spruce_sapling", (6, 2): "minecraft:birch_sapling",
    (6, 3): "minecraft:jungle_sapling", (6, 4): "minecraft:acacia_sapling",
    (6, 5): "minecraft:dark_oak_sapling",
    # Sand
    (12, 1): "minecraft:red_sand",
    # Logs (data & 3 = type, regardless of orientation bits)
    **{(17, d): n for d, n in {
        0: "minecraft:oak_log", 1: "minecraft:spruce_log",
        2: "minecraft:birch_log", 3: "minecraft:jungle_log",
        4: "minecraft:oak_log", 5: "minecraft:spruce_log",
        6: "minecraft:birch_log", 7: "minecraft:jungle_log",
        8: "minecraft:oak_log", 9: "minecraft:spruce_log",
        10: "minecraft:birch_log", 11: "minecraft:jungle_log",
    }.items()},
    # Leaves (data & 3 = type)
    **{(18, d): n for d, n in {
        0: "minecraft:oak_leaves", 1: "minecraft:spruce_leaves",
        2: "minecraft:birch_leaves", 3: "minecraft:jungle_leaves",
        4: "minecraft:oak_leaves", 5: "minecraft:spruce_leaves",
        6: "minecraft:birch_leaves", 7: "minecraft:jungle_leaves",
        8: "minecraft:oak_leaves", 9: "minecraft:spruce_leaves",
        10: "minecraft:birch_leaves", 11: "minecraft:jungle_leaves",
        12: "minecraft:oak_leaves", 13: "minecraft:spruce_leaves",
        14: "minecraft:birch_leaves", 15: "minecraft:jungle_leaves",
    }.items()},
    # Sponge
    (19, 1): "minecraft:wet_sponge",
    # Sandstone
    (24, 1): "minecraft:chiseled_sandstone", (24, 2): "minecraft:smooth_sandstone",
    # Wool (16 colors)
    **{(35, d): f"minecraft:{c}_wool" for d, c in enumerate(_COLORS)},
    # Double slabs
    (43, 0): "minecraft:smooth_stone", (43, 1): "minecraft:sandstone",
    (43, 2): "minecraft:oak_planks", (43, 3): "minecraft:cobblestone",
    (43, 4): "minecraft:bricks", (43, 5): "minecraft:stone_bricks",
    (43, 6): "minecraft:nether_bricks", (43, 7): "minecraft:quartz_block",
    # Stone slabs
    **{(44, d % 8): n for d, n in enumerate([
        "minecraft:stone_slab", "minecraft:sandstone_slab", "minecraft:oak_slab",
        "minecraft:cobblestone_slab", "minecraft:brick_slab", "minecraft:stone_brick_slab",
        "minecraft:nether_brick_slab", "minecraft:quartz_slab",
    ])},
    **{(44, d + 8): n for d, n in enumerate([
        "minecraft:stone_slab", "minecraft:sandstone_slab", "minecraft:oak_slab",
        "minecraft:cobblestone_slab", "minecraft:brick_slab", "minecraft:stone_brick_slab",
        "minecraft:nether_brick_slab", "minecraft:quartz_slab",
    ])},
    # Stained glass (16 colors)
    **{(95, d): f"minecraft:{c}_stained_glass" for d, c in enumerate(_COLORS)},
    # Stone bricks
    (98, 1): "minecraft:mossy_stone_bricks", (98, 2): "minecraft:cracked_stone_bricks",
    (98, 3): "minecraft:chiseled_stone_bricks",
    # Cobblestone wall
    (139, 1): "minecraft:mossy_cobblestone_wall",
    # Hardened clay / terracotta (16 colors)
    **{(159, d): f"minecraft:{c}_terracotta" for d, c in enumerate(_COLORS)},
    # Stained glass panes (16 colors)
    **{(160, d): f"minecraft:{c}_stained_glass_pane" for d, c in enumerate(_COLORS)},
    # Acacia / dark oak logs
    **{(162, d): n for d, n in {
        0: "minecraft:acacia_log", 1: "minecraft:dark_oak_log",
        4: "minecraft:acacia_log", 5: "minecraft:dark_oak_log",
        8: "minecraft:acacia_log", 9: "minecraft:dark_oak_log",
    }.items()},
    # Carpet (16 colors)
    **{(171, d): f"minecraft:{c}_carpet" for d, c in enumerate(_COLORS)},
    # Red sandstone
    (179, 1): "minecraft:chiseled_red_sandstone", (179, 2): "minecraft:smooth_red_sandstone",
    # Quartz
    (155, 1): "minecraft:chiseled_quartz_block", (155, 2): "minecraft:quartz_pillar",
    (155, 3): "minecraft:quartz_pillar", (155, 4): "minecraft:quartz_pillar",
    # Prismarine
    (168, 1): "minecraft:prismarine_bricks", (168, 2): "minecraft:dark_prismarine",
    # Concrete (16 colors)
    **{(251, d): f"minecraft:{c}_concrete" for d, c in enumerate(_COLORS)},
    # Concrete powder (16 colors)
    **{(252, d): f"minecraft:{c}_concrete_powder" for d, c in enumerate(_COLORS)},
}

_LOOKUP = None

def _build_lookup():
    global _LOOKUP
    lookup = np.full(256 * 16, "minecraft:air", dtype=object)
    for bid, name in BLOCK_ID_TO_NAME.items():
        for dat in range(16):
            lookup[bid * 16 + dat] = name
    for (bid, dat), name in BLOCK_ID_DATA_TO_NAME.items():
        lookup[bid * 16 + dat] = name
    _LOOKUP = lookup


def _norm(name: str) -> str:
    """Strip block state properties: 'minecraft:oak_log[axis=y]' -> 'minecraft:oak_log'."""
    return name.split("[")[0]


def load_schematic(file_path: str) -> np.ndarray:
    global _LOOKUP
    if _LOOKUP is None:
        _build_lookup()

    nbt = nbtlib.load(file_path)
    root = nbt.get("Schematic", nbt)

    height = int(root.get("Height", 0))
    length = int(root.get("Length", 0))
    width = int(root.get("Width", 0))

    if height == 0 or length == 0 or width == 0:
        raise ValueError(f"Zero dimension in {file_path}: {height}x{length}x{width}")

    total = height * length * width

    if "Palette" in root and "BlockData" in root:
        palette = {int(v): _norm(str(k)) for k, v in root["Palette"].items()}
        raw = np.array(root["BlockData"], dtype=np.int32).ravel()
        if len(raw) < total:
            raw = np.pad(raw, (0, total - len(raw)))
        raw = raw[:total].reshape(height, length, width)
        max_id = max(palette.keys()) + 1 if palette else 1
        lookup = np.full(max_id, "minecraft:air", dtype=object)
        for k, v in palette.items():
            if k < max_id:
                lookup[k] = v
        return lookup[np.clip(raw, 0, max_id - 1)]

    blocks = np.array(nbt.get("Blocks", []), dtype=np.int32).ravel()
    data = np.array(nbt.get("Data", []), dtype=np.int32).ravel()

    if len(blocks) < total:
        blocks = np.pad(blocks, (0, total - len(blocks)))
    if len(data) < total:
        data = np.pad(data, (0, total - len(data)))

    blocks = (blocks[:total] & 0xFF).reshape(height, length, width)
    data = (data[:total] & 0x0F).reshape(height, length, width)

    return _LOOKUP[blocks * 16 + data]


def _unpack_litematic(longs: np.ndarray, n: int, bits: int) -> np.ndarray:
    """Unpack n values from packed longs using litematica's straddling bit format.

    Litematica stores block entries that CAN straddle 64-bit long boundaries,
    unlike Minecraft's chunk format which aligns entries within each long.
    Formula: start = i*bits, sa = start//64, ea = (start+bits-1)//64, sb = start%64
    """
    longs_u = longs.view(np.uint64)
    mask = np.uint64((1 << bits) - 1)
    i = np.arange(n, dtype=np.int64)
    start = i * bits
    sa = (start >> 6).astype(np.intp)
    ea = ((start + bits - 1) >> 6).astype(np.intp)
    sb = (start & 63).astype(np.uint64)
    v = longs_u[sa] >> sb
    straddle = sa != ea
    if straddle.any():
        ea_safe = np.minimum(ea, len(longs_u) - 1)
        v = np.where(straddle, v | (longs_u[ea_safe] << (np.uint64(64) - sb)), v)
    return (v & mask).astype(np.int32)


def load_litematic(file_path: str) -> np.ndarray:
    """Load a .litematic file (Litematica mod format). Returns the largest region."""
    nbt = nbtlib.load(file_path)
    regions = nbt["Regions"]

    best: np.ndarray | None = None
    best_size = 0

    for region in regions.values():
        palette = [_norm(str(b["Name"])) for b in region["BlockStatePalette"]]
        sx = abs(int(region["Size"]["x"]))
        sy = abs(int(region["Size"]["y"]))
        sz = abs(int(region["Size"]["z"]))
        total = sx * sy * sz
        if total == 0 or total <= best_size:
            continue

        if len(palette) <= 1:
            name = palette[0] if palette else "minecraft:air"
            arr = np.full((sy, sz, sx), name, dtype=object)
        else:
            bits = max(int(np.ceil(np.log2(len(palette)))), 2)
            longs = np.array(region["BlockStates"], dtype=np.int64)
            indices = _unpack_litematic(longs, total, bits)
            palette_arr = np.array(palette, dtype=object)
            arr = palette_arr[indices.clip(0, len(palette) - 1)].reshape(sy, sz, sx)

        best_size = total
        best = arr

    if best is None:
        raise ValueError(f"No valid regions in {file_path}")
    return best


def load_any(file_path: str) -> np.ndarray:
    """Load any supported Minecraft structure format (.schematic, .schem, .litematic)."""
    if file_path.lower().endswith(".litematic"):
        return load_litematic(file_path)
    return load_schematic(file_path)
