"""
NavAble Synthetic Data Extension — Omniverse Kit Extension for Isaac Sim 5.1.0
==============================================================================

Synthetic-data collection toolkit for accessibility object detection research.
Provides:

* **GamepadCameraController** — FPS-style camera control via Logitech F710 /
  any XInput-compatible gamepad.
* **TrajectoryRecorder / TrajectoryPlayer** — Record per-frame camera poses and
  replay them deterministically.
* **DataRecorder** — Replicator BasicWriter-based capture of RGB, semantic
  segmentation, colorized semantic segmentation, and 2-D tight bounding boxes.
* **AssetBrowser** — Browse a folder of USD assets, swap them via USD
  references, and apply semantic labels automatically.
* **DataCollectorWindow** — Unified ``omni.ui`` window with single-root project
  settings, trajectory list, annotator info panel, and simplified UX.

Usage
-----
Enable the extension ``navable_synth_data_collector`` in the Extension Manager.
A menu item will appear under *Window → NavAble Synthetic Data Extension*.
"""

__version__ = "1.0.0"

try:
    from .extension import NavAbleSynthDataExtension  # noqa: F401
except ImportError:
    # Imported outside Isaac Sim (unit tests, CLI bootstrap) — extension
    # entry point requires carb/omni which aren't always available.
    pass
