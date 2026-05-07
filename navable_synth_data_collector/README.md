# NavAble Synthetic Data Extension

An Omniverse Kit / Isaac Sim extension for synthetic data collection in
accessibility object detection research.

|                |                                                              |
| -------------- | ------------------------------------------------------------ |
| **Platform**   | Isaac Sim 5.1.0 / Kit 107.3.3                                |
| **Python**     | 3.11 (Isaac Sim's bundled interpreter)                       |
| **GPU**        | NVIDIA RTX (Replicator requires RTX for ray-traced captures) |
| **Controller** | Logitech F710 (XInput) or any Xbox-compatible gamepad        |

## Features

- **Gamepad-driven FPS camera control** — fly a first-person camera through any
  USD stage with an XInput-compatible controller.
- **Trajectory recording & playback** — record per-frame camera poses to JSON
  and replay them deterministically.
- **Replicator-based multi-modal capture** — RGB, semantic segmentation,
  colorized segmentation, and tight 2-D bounding boxes via
  `omni.replicator.core`.
- **USD asset browser with auto-labeling** — cycle through a folder of USD
  assets; each is loaded as a reference and tagged with a semantic class.
- **Location management** — per-environment named spawn points. Saved
  transforms follow the asset across capture runs.
- **Collect All Data** — one-click matrix capture across every environment ×
  location × trajectory × asset in your project.
- **Headless CLI** — run any of the above from the shell without opening
  Isaac Sim's GUI.

---

## Contents

- [Installation](#installation)
- [Quick Start (GUI)](#quick-start-gui)
- [UI Reference](#ui-reference)
- [Gamepad Map](#gamepad-map)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Output Layout](#output-layout)
- [Trajectory JSON Format](#trajectory-json-format)
- [Developer Guide](#developer-guide)
- [Running the Tests](#running-the-tests)
- [Dependencies](#dependencies)

---

## Installation

### Prerequisites

1. **Isaac Sim 5.1.0** installed and runnable on your machine.
2. **NVIDIA RTX GPU** (Replicator's ray-traced annotators require RT cores).
3. **An XInput-compatible gamepad** (Logitech F710, Xbox controller, etc.).
   The extension can run without a controller, but the gamepad camera and
   record-toggle bindings will be disabled.

### Step 1 — Install the extension into Isaac Sim

Drop the entire `navable_synth_data_collector/` folder (this repository) into
one of Isaac Sim's extension search paths. Two locations work out of the box:

```bash
# Option A — Isaac Sim's user extensions (recommended)
mkdir -p ~/Documents/Kit/shared/exts
cp -r /path/to/navable_synth_data_collector ~/Documents/Kit/shared/exts/

# Option B — bundled into your Isaac Sim install
cp -r /path/to/navable_synth_data_collector \
      <ISAAC_SIM_ROOT>/exts/
```

If you'd like to add a custom search path instead, open Isaac Sim, go to
**Window → Extensions → ⚙ (gear) → Extension Search Paths**, and add the
parent folder of `navable_synth_data_collector/`.

### Step 2 — Enable the extension

1. Launch Isaac Sim.
2. Open **Window → Extensions**.
3. Search for **"NavAble Synthetic Data Extension"**.
4. Toggle the extension **ON**. Tick **AUTOLOAD** so it stays enabled across
   restarts.

The main window appears under **Window → NavAble Synthetic Data Extension**.

The extension is marked `reloadable = true`, so source edits hot-reload from
the Extension Manager without restarting Isaac Sim.

### Step 3 — Configure your project paths

Open `config/config.yaml` and set the three paths to match your filesystem:

```yaml
root_folder:         "/absolute/path/to/where/captures/should/be/written"
asset_root_folder:   "/absolute/path/to/USD/asset/library"
environments_folder: "/absolute/path/to/USD/environment/stages"
```

Asset library layout — assets are scanned at
`{asset_root_folder}/{asset_class_name}/*.usd*`, e.g.

```
{asset_root_folder}/
├── handrail/
│   ├── handrail_a.usd
│   └── handrail_b.usd
├── stairs/
│   └── ...
└── ...
```

Environment stages are USD files anywhere under `environments_folder` (one
USD per environment). Both the GUI and the CLI auto-discover them.

### Step 4 (optional) — Install the CLI on a stock Python

The headless CLI lives in `navable_synth_data_collector/cli/` and ships as an
installable console script. From the repository root:

```bash
pip install -e .[test]
```

This installs the `navable-collect` command and the test extras
(`pytest`, `pyyaml`). Capture commands still need Isaac Sim's bundled Python
to run, but `navable-collect list` and the unit-test suite work with any
Python 3.11 — Isaac Sim does **not** need to be importable.

---

## Quick Start (GUI)

1. **Project Settings** — set Root Folder, Environments Folder, Asset Root,
   then pick an **Environment** and a **Class Name** from the dropdowns.
   Click **Apply Settings**. The extension scans the asset folder for that
   class and loads locations for that environment.
2. **Camera Controller** — enter the camera prim path (default
   `/World/Camera`) and click **Set**. Click **Enable Gamepad** to take
   FPS-style control.
3. **Asset Browser** — pick a location from the dropdown (or click **New
   Location**). Load an asset with **Prev / Next**, position it in the
   viewport, and click **Save Transform** to store the spawn pose.
4. **Trajectory Recording** — type a trajectory name and click **Record**.
   Fly a path. Click **Stop & Save** (or press **X** on the gamepad to
   toggle recording).
5. **Record with Trajectory** — pick a trajectory from the dropdown, pick an
   asset, set a frame step, click **Record**. The camera replays the path
   while Replicator captures each frame.
6. **Collect All Data** — click **Start** to iterate every environment ×
   location × trajectory × asset in the current class. A progress bar tracks
   the run; **Cancel** stops cleanly between units of work.

---

## UI Reference

The main window is eight collapsible sections, each backed by one file under
[navable_synth_data_collector/ui/sections/](navable_synth_data_collector/ui/sections/).
All backend logic lives on a single `Session` object — the sections only
render widgets and delegate calls.

| Section                    | File                                                                                            | Purpose                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Project Settings**       | [project.py](navable_synth_data_collector/ui/sections/project.py)                               | Root/env/class/resolution. Apply scans assets + auto-selects first location.       |
| **Camera Controller**      | [camera.py](navable_synth_data_collector/ui/sections/camera.py)                                 | Camera path, Enable/Disable gamepad, live-synced Move/Look speed sliders.          |
| **Asset Browser**          | [asset_browser.py](navable_synth_data_collector/ui/sections/asset_browser.py)                   | Location CRUD, Prev/Next, Save Transform, live position/orientation/scale readout. |
| **Trajectory Recording**   | [trajectory_record.py](navable_synth_data_collector/ui/sections/trajectory_record.py)           | Name field + Record / Stop. Saves to the active location's `trajectories/`.        |
| **Trajectory Playback**    | [trajectory_play.py](navable_synth_data_collector/ui/sections/trajectory_play.py)               | Combo of saved trajectories + Play / Stop + frame counter.                         |
| **Data Capture**           | [capture.py](navable_synth_data_collector/ui/sections/capture.py)                               | Read-only: shows active annotators, resolution, RT subframes.                      |
| **Record with Trajectory** | [record_with_trajectory.py](navable_synth_data_collector/ui/sections/record_with_trajectory.py) | Select trajectory + frame step, Record / Cancel / Record All.                      |
| **Collect All Data**       | [collect_all.py](navable_synth_data_collector/ui/sections/collect_all.py)                       | Start / Cancel the full env × loc × traj × asset matrix.                           |

### Location auto-save

When you move a saved asset in the viewport, the Asset Browser debounces the
change and writes the new transform to the current location's
`location.json` automatically — no button press required. The manual
**Save Transform** button is retained for explicit saves.

---

## Gamepad Map

Tested with a Logitech F710 in **XInput** mode; any Xbox-compatible
controller should work.

| Input              | Action                         |
| ------------------ | ------------------------------ |
| Left stick Y       | Move forward / backward        |
| Left stick X       | Strafe left / right            |
| Right stick Y      | Pitch (look up / down)         |
| Right stick X      | Yaw (look left / right)        |
| Right trigger      | Move up                        |
| Left trigger       | Move down                      |
| D-Pad Up / Down    | Increase / decrease move speed |
| D-Pad Left / Right | Decrease / increase look speed |
| Left bumper        | Toggle slow mode (50%)         |
| **X button**       | Toggle trajectory recording    |

Speed changes made via the D-pad are reflected live in the Camera Controller
sliders.

---

## CLI Reference

All CLI commands share the same `Session` as the GUI, so configuration, data
layout, and capture behavior are identical across the two.

```bash
navable-collect --help
navable-collect list             [--config PATH]
navable-collect record-all       [--config PATH] [--env NAME] [--class NAME] [--location NAME] [--frame-step N]
navable-collect collect-all      [--config PATH] [--class NAME] [--frame-step N] [--on-error skip|abort]
navable-collect collect-classes  [--config PATH] [--frame-step N]
```

### `list` — inspect a project without booting Isaac Sim

Reads the config, scans the filesystem, and prints a tree of environments,
locations, and trajectories. Does **not** boot `SimulationApp`, so it runs
in <1 s on plain Python 3.11.

```bash
navable-collect list --config config/config.yaml
```

### `record-all` — GUI "Record All Trajectories" button (headless)

Boots Isaac Sim headless, opens the configured environment, sets the
configured location, and iterates every asset × every trajectory at that
location. Every required value can come from the YAML; CLI flags only
override.

```bash
navable-collect record-all --config config/config.yaml
```

Required YAML keys: `root_folder`, `asset_root_folder`, `environments_folder`,
`environment`, `asset_class_name`, `location`. Optional: `frame_step`,
`resolution_width`/`resolution_height`, `rt_subframes`, `camera_path`,
`annotators`.

### `collect-all` — GUI "Collect All Data" button (headless)

Iterates every environment × location × asset × trajectory discovered under
`environments_folder`. Same YAML schema as `record-all`; `environment` and
`location` are ignored (the full matrix is enumerated). `Ctrl-C` cancels
cleanly.

```bash
navable-collect collect-all --config config/config.yaml
```

### `collect-classes` — multi-class headless capture

Same as `collect-all` but loops over each class in the YAML's `classes`
list inside a single SimulationApp boot. When `classes` is empty, falls
back to a one-element list containing `asset_class_name`.

```bash
navable-collect collect-classes --config config/config.yaml
```

Example YAML (shared by every command):

```yaml
root_folder:         "/path/to/collected_data"
asset_root_folder:   "/path/to/synthetic_objects"
environments_folder: "/path/to/environments"
environment:         "hotel_corridor"   # used by record-all; ignored elsewhere
asset_class_name:    "handrail"
location:            "entrance"         # used by record-all; ignored elsewhere
classes:             []                 # populate for collect-classes
frame_step:          1
resolution_width:    1280
resolution_height:   720
rt_subframes:        4
camera_path:         "/World/Camera"
parent_prim_path:    "/World"
annotators:
  rgb:                   true
  semantic_segmentation: true
  bounding_box_2d_tight: true
```

---

## Configuration

Defaults are resolved in this priority order:
**UI / CLI arguments > `config/config.yaml` > `extension.toml` carb settings >
hardcoded defaults**.

### `config/extension.toml` (Kit-level)

| Setting                     | Default          | Notes                     |
| --------------------------- | ---------------- | ------------------------- |
| `default_camera_path`       | `/World/Camera`  | Camera prim path          |
| `default_move_speed`        | `20.0`           | m/s                       |
| `default_look_speed`        | `30.0`           | deg/s                     |
| `default_resolution_width`  | `1280`           | capture width             |
| `default_resolution_height` | `720`            | capture height            |
| `default_rt_subframes`      | `4`              | RT subframes per capture  |
| `default_root_folder`       | `~/navable_data` | project root              |
| `default_environment`       | `""`             | selected env on startup   |
| `default_asset_class_name`  | `""`             | selected class on startup |

### `config/config.yaml` (user-editable)

Same keys as above plus CLI-specific ones (`annotators`, `classes`,
`frame_step`). See the file shipped in `config/config.yaml` for the full
schema.

---

## Output Layout

```
{root_folder}/{class_name}/{environment}/
├── trajectories/                         # trajectories not tied to a location
├── captures/{class}_{asset}/{trajectory}/
│   ├── rgb/
│   ├── semantic_segmentation/
│   └── bounding_box_2d_tight/
└── {location_name}/
    ├── location.json                     # spawn transform + metadata
    ├── trajectories/
    │   └── trajectory_*.json
    └── captures/…
```

Paths are derived by
[backend/paths.py](navable_synth_data_collector/backend/paths.py) — a pure-Python
module with no Isaac Sim dependencies.

---

## Trajectory JSON Format

```json
{
  "version": "1.0",
  "name": "trajectory_001",
  "environment": "hotel_corridor",
  "camera_path": "/World/Camera",
  "fps": 60,
  "frame_count": 300,
  "created": "2026-01-01T13:00:00",
  "frames": [
    {"frame": 0, "position": [1.0, 2.0, -3.0], "rotation": [10.0, -45.0, 0.0]}
  ]
}
```

Serialization helpers live in
[backend/trajectory_io.py](navable_synth_data_collector/backend/trajectory_io.py)
and are unit-tested.

---

## Developer Guide

### Architecture

```
navable_synth_data_collector/
├── extension.py               # omni.ext.IExt — on_startup / on_shutdown
├── backend/                   # pure orchestration; no omni.ui
│   ├── session.py             # Session — the single workflow facade
│   ├── stage.py               # StageController — event-driven stage swap
│   ├── capture.py             # DataRecorder + ensure_setup
│   ├── trajectory.py          # TrajectoryRecorder / TrajectoryPlayer / TrajectoryManager
│   ├── trajectory_io.py       # pure JSON helpers
│   ├── gamepad_camera.py      # XInput FPS camera
│   ├── asset_browser.py       # USD reference swap + semantic labels
│   ├── location.py            # per-environment spawn points
│   ├── config.py              # yaml / toml / defaults merge
│   ├── paths.py               # pure path derivation + collect-all planner
│   └── events.py              # tiny EventBus
├── ui/                        # widgets only; every callback is a one-liner
│   ├── window.py              # DataCollectorWindow shell
│   ├── style.py               # layout constants
│   └── sections/              # one file per collapsible frame
└── cli/                       # headless entry points
    ├── __main__.py            # argparse dispatch
    ├── bootstrap.py           # SimulationApp headless boot
    └── commands/
        ├── list_cmd.py
        ├── record_all_cmd.py
        ├── collect_all_cmd.py
        └── collect_classes_cmd.py
```

### Key design decisions

- **One Session, two frontends.** The GUI and CLI both instantiate the same
  `Session`. UI callbacks pass a `progress_cb` that writes to labels; the
  CLI passes one that writes stdout / tqdm.
- **Pure backend where possible.** `paths`, `config`, `trajectory_io`, and
  `location` guard their `carb`/`omni` imports with `try/except ImportError`,
  making them unit-testable without Isaac Sim.
- **Lazy capture setup.** `DataRecorder.ensure_setup(output_dir, …)` is
  idempotent: if the writer isn't initialized yet it calls `setup()`; if the
  output dir changed it calls `reinitialize_writer()`; otherwise it's a
  no-op.
- **Event-driven stage swaps.** `StageController.switch_to(usd_path)` runs
  pre-close hooks, closes the stage, opens the new one, warms up for N
  frames, then runs post-open hooks. Replicator is rebuilt lazily on the
  next capture against the new stage's camera.
- **EventBus for decoupling.** `backend/events.py` lets `AssetBrowser` notify
  `LocationManager` of transform deltas without holding a reference —
  essential for hot-reload safety.

### Adding a new UI section

1. Create `navable_synth_data_collector/ui/sections/my_feature.py`:
   
   ```python
   import omni.ui as ui
   
   class MyFeatureSection:
       def __init__(self, parent_vstack, session, widgets, style, refresh_cb=None):
           self.session = session
           self.widgets = widgets
           with ui.CollapsableFrame("My Feature", height=0):
               with ui.VStack(spacing=style.SPACING):
                   ui.Button("Do Thing", clicked_fn=self._on_click)
   
       def _on_click(self):
           self.session.my_workflow()
   
       def on_tick(self):
           pass
   
       def destroy(self):
           pass
   ```

2. Import and instantiate it from
   [ui/window.py](navable_synth_data_collector/ui/window.py) in
   `_build_ui`.

3. If the new feature needs backend work, add a method on `Session` —
   **never** put `omni.usd`, `pxr`, or file I/O in a section file.

---

## Running the Tests

The unit-test suite covers every pure-Python module in `backend/`. It does
**not** require Isaac Sim to be importable —
`tests/unit/conftest.py` stubs `carb`, `omni.*`, `pxr`, and `isaacsim.*`
into `sys.modules` at collection time.

### One-time setup

```bash
pip install pytest pyyaml
```

### Running

From the extension root:

```bash
python -m pytest navable_synth_data_collector/tests/unit -v
```

You can also filter by file:

```bash
python -m pytest navable_synth_data_collector/tests/unit/test_paths.py -v
python -m pytest navable_synth_data_collector/tests/unit/test_location.py::test_save_transform_preserves_metadata -v
```

### What each file covers

| File                                                                                           | Coverage                                                                                                                   |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [test_paths.py](navable_synth_data_collector/tests/unit/test_paths.py)                         | Path derivation: `class_env_dir`, `location_dir`, `run_dir`, `next_default_run_name`, `sanitize_folder_name`, `normalize`. |
| [test_config.py](navable_synth_data_collector/tests/unit/test_config.py)                       | Priority order YAML > carb > defaults; malformed YAML falls back gracefully.                                               |
| [test_location.py](navable_synth_data_collector/tests/unit/test_location.py)                   | `validate_name`, `create_location`, `save_transform`, `list_locations`, `delete_location`.                                 |
| [test_trajectory_io.py](navable_synth_data_collector/tests/unit/test_trajectory_io.py)         | JSON round-trip at schema v1.0; `list_trajectory_files` ordering.                                                          |
| [test_events.py](navable_synth_data_collector/tests/unit/test_events.py)                       | `EventBus` subscribe / publish / unsubscribe semantics.                                                                    |
| [test_gamepad_slow_mode.py](navable_synth_data_collector/tests/unit/test_gamepad_slow_mode.py) | Slow-mode multiplier and dead-zone behavior of the gamepad controller.                                                     |
| [test_cli_parser.py](navable_synth_data_collector/tests/unit/test_cli_parser.py)               | `argparse` wiring for the `navable-collect` entry point.                                                                   |

### Writing new tests

Any new module you want to unit-test must:

1. Guard external imports with
   `try: import carb\nexcept ImportError: carb = None`.
2. Keep `omni`, `pxr`, `carb`, and `isaacsim` out of the module's top-level
   scope — push them into functions or `try/except` blocks.

Drop new tests into `navable_synth_data_collector/tests/unit/` — pytest
picks them up automatically.

---

## Dependencies

### Runtime (Isaac Sim loads these via `extension.toml`)

- `omni.kit.uiapp`
- `omni.ui`
- `omni.usd`
- `omni.kit.viewport.utility`
- `omni.replicator.core`
- `isaacsim.core.api`
- `isaacsim.core.utils`

### Python (for CLI + tests)

- `pyyaml` (CLI)
- `pytest` (tests)

---

## License

This artifact is provided for the purposes of academic peer review.
