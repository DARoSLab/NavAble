#!/usr/bin/env python3
"""
Example: Loading NavAble assets in Isaac Sim

This script demonstrates how to load accessibility object assets
into an Isaac Sim environment for simulation-based training.

Usage:
    # Run from Isaac Sim Python environment
    python load_assets.py --asset data/usd_assets/tactile_paving.usd
"""

import argparse
from pathlib import Path


def load_single_asset(usd_path: str, position: tuple = (0, 0, 0)):
    """Load a single USD asset into the scene."""
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from pxr import Gf

    world = World()

    asset_name = Path(usd_path).stem
    prim_path = f"/World/{asset_name}"

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    # Set position
    from omni.isaac.core.utils.prims import get_prim_at_path
    prim = get_prim_at_path(prim_path)
    if prim:
        from pxr import UsdGeom
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position))

    print(f"Loaded asset: {asset_name} at position {position}")
    return prim_path


def load_assets_from_directory(usd_dir: str, spacing: float = 1.0):
    """Load all USD assets from a directory in a grid layout."""
    from omni.isaac.core import World

    world = World()
    usd_dir = Path(usd_dir)
    usd_files = list(usd_dir.glob("*.usd"))

    print(f"Found {len(usd_files)} USD assets")

    loaded = []
    for i, usd_file in enumerate(usd_files):
        row = i // 5
        col = i % 5
        position = (col * spacing, row * spacing, 0)
        prim_path = load_single_asset(str(usd_file), position)
        loaded.append(prim_path)

    return loaded


def create_navigation_scene(assets_config: dict):
    """
    Create a navigation scene with accessibility objects.

    Example config:
    {
        "tactile_paving": {"path": "data/usd_assets/tactile.usd", "positions": [(0, 0, 0), (1, 0, 0)]},
        "aps_signal": {"path": "data/usd_assets/aps.usd", "positions": [(5, 0, 0)]}
    }
    """
    from omni.isaac.core import World

    world = World()
    scene_prims = {}

    for obj_type, config in assets_config.items():
        usd_path = config["path"]
        positions = config.get("positions", [(0, 0, 0)])

        scene_prims[obj_type] = []
        for i, pos in enumerate(positions):
            prim_path = load_single_asset(usd_path, pos)
            scene_prims[obj_type].append(prim_path)

    return scene_prims


def main():
    parser = argparse.ArgumentParser(description="Load NavAble assets in Isaac Sim")
    parser.add_argument("--asset", "-a", help="Path to single USD asset")
    parser.add_argument("--directory", "-d", help="Path to directory of USD assets")
    parser.add_argument("--spacing", type=float, default=1.0, help="Spacing between assets in grid")
    args = parser.parse_args()

    if args.asset:
        load_single_asset(args.asset)
    elif args.directory:
        load_assets_from_directory(args.directory, args.spacing)
    else:
        print("Provide --asset or --directory")
        print("\nExample usage:")
        print("  python load_assets.py --asset data/usd_assets/tactile_paving.usd")
        print("  python load_assets.py --directory data/usd_assets/ --spacing 1.5")


if __name__ == "__main__":
    main()
