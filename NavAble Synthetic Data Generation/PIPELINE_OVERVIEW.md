# NavAble — Full Pipeline Context Overview

> **Note:** This document has been anonymized for double-blind paper review. Author and institutional information has been removed.
> 
> A research-grade context document describing the architecture, motivation,
> components, data flow, and design decisions of the **NavAble** image-to-3D
> synthetic data generation pipeline used to produce 3D
> accessibility objects for blind and low-vision navigation research.
> 
> The intended audience is an LLM that needs full context to help draft a
> research paper. Every component, file path, hyperparameter, and design
> trade-off worth citing in a paper is documented below.

---

## 1. Project Summary

**Project name:** NavAble
**Submodule package (internal codename):** `isaacnav` (`src/isaacnav/`)
**Pipeline version (config):** `2.2`
**Package version (`pyproject.toml`):** `2.0.0`
**License:** Code MIT, dataset CC BY 4.0
**Hardware target:** Linux + CUDA 12.8 + RTX 5090 (Blackwell SM_120), Python 3.11, PyTorch 2.7.1

### One-line description

A configurable, strategy-pattern-based pipeline that **crawls** images of
real-world accessibility infrastructure from the open web, **VLM-validates**
each image for class correctness and reconstruction suitability, runs
**class-aware open-vocabulary segmentation** (Grounding DINO + SAM 2.1) with
per-class **prompt expansion**, lifts each masked object into a 3D mesh
(Meta's SAM 3D Objects), and converts the result to **USD/USDZ** assets that
can be loaded directly into NVIDIA Isaac Sim — producing the NavAble
dataset.

### Why this dataset exists

The pipeline targets a problem space that has been chronically under-served
by existing 3D asset and indoor-scene datasets:

- **Blind / low-vision (BLV) navigation infrastructure** — escalators,
  handrails, tactile paving, accessible pedestrian-signal (APS) buttons,
  bus-stop signs, turnstiles, elevator buttons, ADA door-openers, etc.
- These objects rarely appear (or appear with very low diversity) in
  general-purpose 3D asset libraries (e.g. Objaverse, ShapeNet, ABO), and
  they are too small / too structural to be reliably extracted from
  large-scale scene scans.
- For sim-to-real navigation training (Isaac Sim, Habitat-style stacks),
  what is needed is a *catalog of class-labeled, USD-ready 3D primitives*
  that can be dropped into procedurally-generated environments.

The companion datasheet (`DATASHEET.md`) frames the dataset against the
"Datasheets for Datasets" (Gebru et al., 2021) checklist and describes the
intended uses, biases, and limitations.

---

## 2. Repository Layout

```
NavAble/
├── README.md
├── DATASHEET.md                    # Gebru-style datasheet
├── docs/DATASET.md                 # GuideTWSI tactile-paving dataset doc
├── pyproject.toml                  # Package metadata
├── envs/
│   ├── environment.yaml            # Conda env (CUDA 12.8 / Blackwell)
│   └── install.sh                  # One-command setup
├── configs/
│   ├── pipeline.yaml               # Master pipeline config (v2.2)
│   ├── object_categories.yaml      # Detailed taxonomy (4 super-cat × subcats)
│   └── object_categories_high_lev.yaml
├── scripts/
│   ├── run_pipeline.py             # Config-driven CLI
│   └── run_batch.py                # Manifest-tracked batch CLI
├── src/isaacnav/
│   ├── __init__.py
│   ├── pipeline.py                 # AssetPipeline orchestrator
│   ├── crawling/                   # 4 crawler backends (Wikimedia, Bing,
│   │                               #   DuckDuckGo, Multi)
│   ├── validation/                 # Gemini VLM validator
│   ├── masking/                    # SAM heuristic + Grounded SAM 2
│   ├── reconstruction/             # SAM 3D Objects wrapper
│   ├── conversion/                 # GLB/PLY → USD/USDZ converters
│   └── data/                       # Manifest + per-object output layout
├── extern/                         # Git submodules
│   ├── sam-3d-objects/             # Meta's SAM 3D Objects
│   └── Grounded-SAM-2/             # Grounding DINO + SAM 2.1
├── data/                           # Working data (gitignored)
│   ├── input/{class_name}/         # Crawled, validated images
│   └── jobs/                       # Batch manifests
├── output/                         # Per-object 3D assets (gitignored)
│   └── {class_name}/{obj_id}/
└── benchmarks/{detection,navigation,simulation}/
```

---

## 3. End-to-End Pipeline

The high-level flow is:

```
┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐
│  Crawl   │──▶│ VLM Val  │──▶│ Class-Aware  │──▶│  SAM 3D  │──▶│  GLB →   │
│ (Stage 0)│   │  (Gemini)│   │ Segmentation │   │ Objects  │   │  USDZ    │
└──────────┘   └──────────┘   └──────────────┘   └──────────┘   └──────────┘
   Wikimedia    Reject if      Grounding DINO    Single-image    Isaac Sim
   Bing/DDG     not the         + SAM 2.1         3D mesh +       compatible
   Multi        target class    + prompt          Gaussian        textured
                or unsuitable   expansion         splat PLY       USD package
                for SAM 3D      + box NMS
```

Each image belongs to **exactly one class**. The class is determined by:

1. Explicit `--class` / `--classes` CLI argument, OR
2. Parent directory name (`data/input/{class_name}/img.jpg`), OR
3. Fallback to first class in `target_classes` from config.

This monolithic, class-per-folder design is a deliberate simplification —
it keeps the pipeline traceable per-class, makes it easy to crawl /
validate / process in isolation, and matches the per-class output
directory structure (`output/{class_name}/{object_id}/`).

### 3.1 Stages and corresponding code

| Stage           | Code (file)                                                                  | Strategies                                 | Primary entry point                                                       |
| --------------- | ---------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| 0 — Crawl       | [src/isaacnav/crawling/](src/isaacnav/crawling/)                             | `wikimedia`, `bing`, `duckduckgo`, `multi` | [pipeline.py:_crawl_classes()](src/isaacnav/pipeline.py#L402)             |
| 1 — Validate    | [src/isaacnav/validation/gemini.py](src/isaacnav/validation/gemini.py)       | `gemini`                                   | [pipeline.py:_validate_images()](src/isaacnav/pipeline.py#L176)           |
| 2 — Segment     | [src/isaacnav/masking/](src/isaacnav/masking/)                               | `grounded_sam2`, `sam_heuristic`           | [pipeline.py:run_single()](src/isaacnav/pipeline.py#L269)                 |
| 3 — Reconstruct | [src/isaacnav/reconstruction/sam3d.py](src/isaacnav/reconstruction/sam3d.py) | `sam3d`                                    | [pipeline.py:run_single()](src/isaacnav/pipeline.py#L269)                 |
| 4 — Convert     | [src/isaacnav/conversion/](src/isaacnav/conversion/)                         | `glb_to_usd`, `ply_to_usd`                 | [pipeline.py:_run_conversion_subprocess()](src/isaacnav/pipeline.py#L214) |

### 3.2 Orchestrator: `AssetPipeline`

[`src/isaacnav/pipeline.py`](src/isaacnav/pipeline.py) defines the
`AssetPipeline` class. Its responsibilities:

- Load `configs/pipeline.yaml` and resolve target classes.
- Lazily instantiate each stage strategy via name-keyed registries
  ([pipeline.py:28-66](src/isaacnav/pipeline.py#L28-L66)).
- Provide three top-level entry points:
  - `run_crawl_only(classes)` — crawl + validate, no 3D processing.
  - `run_crawl_and_process(classes)` — crawl + validate + segment +
    reconstruct + convert.
  - `run_all_classes(classes)` — process images already on disk under
    `data/input/{class_name}/`.
  - `run_single(image_path, class_name)` — full per-image processing.
- Maintain a per-object output directory via `OutputLayout`
  ([data/output_layout.py](src/isaacnav/data/output_layout.py)).
- Persist a `metadata.json` per object with fields for `source`,
  `masking`, `reconstruction`, `conversion`.

### 3.3 Output layout

Per [data/output_layout.py:11-22](src/isaacnav/data/output_layout.py#L11-L22):

```
output/{class_name}/{class_name}_{image_stem}/
    ├── image.jpg                 # original
    ├── mask.png                  # selected best mask
    ├── {stem}.glb                # textured mesh
    ├── {stem}_gs.ply             # gaussian splat
    ├── asset.usd                 # intermediate USD
    ├── asset.usdz                # packaged USD-Z (Isaac Sim ready)
    ├── textures/                 # diffuse texture(s) for USD material
    └── metadata.json             # full provenance + stage metadata
```

Example (from existing `output/escalator/escalator_Escalator_bing_0003/`):

```json
{
  "object_id": "escalator_Escalator_bing_0003",
  "category": "Escalator",
  "created_at": "2026-04-17T03:09:05Z",
  "pipeline_version": "2.2",
  "masking": {
    "class_name": "Escalator",
    "confidence": 0.37,
    "bbox": [1055, 336, 2424, 1424],
    "strategy": "grounded_sam2",
    "total_detections": 1
  },
  "reconstruction": {
    "strategy": "sam3d",
    "ply_path": ".../Escalator_bing_0003_gs.ply",
    "glb_path": ".../Escalator_bing_0003.glb"
  },
  "conversion": {
    "format": "usdz",
    "has_texture": false
  }
}
```

---

## 4. Stage 0 — Image Crawling

Source: [`src/isaacnav/crawling/`](src/isaacnav/crawling/)

The pipeline acquires raw images via four interchangeable backends. All
crawlers implement [`BaseCrawler`](src/isaacnav/crawling/base.py), which
returns a list of `CrawlResult(image_path, metadata_path, source, license,
attribution)`. Each crawler also writes a sidecar `*.json` containing the
source URL, attribution, and license (where available).

### 4.1 `WikimediaCommonsCrawler` — [crawling/wikimedia.py](src/isaacnav/crawling/wikimedia.py)

- Primary CC-licensed source via the MediaWiki API.
- No API key. Polite User-Agent (`NavAbleBot/2.1`) and configurable
  `delay` (default 1.5 s) + 429 back-off with `Retry-After` honoring.
- Filters out SVG/GIF and < 300×300 px assets.
- Records full attribution (`Artist`, `LicenseShortName`) into metadata —
  important for paper compliance with CC-BY/CC-BY-SA/CC0 attribution
  requirements.

### 4.2 `BingCrawler` — [crawling/bing.py](src/isaacnav/crawling/bing.py)

- Uses the `icrawler.builtin.BingImageCrawler`. No API key.
- Over-fetches by 1.5× to compensate for download failures and
  small-image filtering (`min_size=300`).
- License is recorded as `"Unknown"` — Bing-sourced images are used
  as **input to validation**, not redistributed unverified.

### 4.3 `DuckDuckGoCrawler` — [crawling/duckduckgo.py](src/isaacnav/crawling/duckduckgo.py)

- Uses the `ddgs` Python package; falls back to `duckduckgo_search`.
- Aggressive back-off on rate limits / 403 (10 s × attempt).

### 4.4 `MultiCrawler` — [crawling/multi.py](src/isaacnav/crawling/multi.py)

- Runs N backends concurrently (`ThreadPoolExecutor`) for speed +
  *image diversity*. Each gets `(num_per_class // N) + 5` quota, and
  results are then **deduplicated by MD5 of file content**.
- Caps the final unique count to the requested `num_per_class`.
- Used to generate corpora that mix open-licensed (Wikimedia) with
  broader-coverage (Bing/DDG) sources.

### 4.5 Output structure

All crawled images are written to `data/input/{safe_class_name}/`, where
`safe_class_name = class.lower().replace(" ", "_")`. Filenames embed the
source: `{query}_wiki_{pageid}.jpg`, `{query}_bing_{idx:04d}.jpg`,
`{query}_ddg_{idx:04d}.jpg`. This lets the rest of the pipeline trivially
infer (class, source) at any point.

---

## 5. Stage 1 — VLM Validation (Gemini)

Source: [`src/isaacnav/validation/gemini.py`](src/isaacnav/validation/gemini.py)

A core innovation of this pipeline (relative to a vanilla
crawl→segment→reconstruct stack) is that **every crawled image is passed
through a vision-language model that decides whether the image is a
suitable input for a single-image 3D reconstructor**. Failures are
deleted from `data/input/`.

### 5.1 Why VLM validation matters

Open-web image search returns a long tail of failures that silently
poison a downstream image-to-3D model:

- **Diagrams / icons / CAD renders** — return clean masks but produce
  meaningless meshes.
- **Movie stills / illustrations** — break the photometric prior of
  SAM 3D Objects.
- **Heavy occlusion / extreme truncation** — produce holey or implausibly
  reconstructed geometry.
- **Stock-photo watermarks** — bake into the mesh texture.
- **Wrong sub-class** (e.g., a "moving walkway" returned for "escalator",
  a "subway tripod gate" returned for "turnstile") — pollute class labels.

The Gemini validator's system prompt (in
[gemini.py:55-71](src/isaacnav/validation/gemini.py#L55-L71)) makes these
rejection criteria explicit:

> CRITICAL REJECTION CRITERIA — Answer NO if the image has ANY of the following:
> 
> - It is a drawing, 3D render, CAD model, diagram, icon, or screenshot.
> - The object is heavily occluded by people, vehicles, or other objects.
> - The object is severely truncated.
> - Heavy intrusive stock-photo watermarks.
> - Object too distant / small to make out structural geometry.

### 5.2 Per-class prompt customization

Per-class validation prompts live in `configs/pipeline.yaml`:
[`validation.prompt.class_prompts`](configs/pipeline.yaml#L60-L123).
Falls back to a default `"Does this image contain a '{class_name}'?"` if
no class override is configured. The prompt resolution logic is in
[pipeline.py:_resolve_validation_prompt()](src/isaacnav/pipeline.py#L165).

Class prompts are written as **strict accept/reject specifications**.
Example for "Turnstile":

> ACCEPT: Full-height security turnstiles (floor-to-ceiling rotating
> steel bars) or large revolving doors at building entrances.
> REJECT STRICTLY: Waist-high subway turnstiles, tripod/three-arm
> turnstiles, paddle/flap barriers, or optical turnstiles used for basic
> ticket scanning.

This effectively functions as a **VLM-driven class-level taxonomy
disambiguation step** — the open-vocabulary search query is intentionally
loose (so we get more raw images), and the VLM enforces the precise
class definition the dataset cares about.

### 5.3 Response format

The VLM is asked to return a structured response:

```
CONTAINS: YES | NO
ACTUAL: <one-sentence factual description>
CONFIDENCE: <0.0 to 1.0>
```

…parsed by `_parse_response()` into a `ValidationResult(contains_class,
actual_content, confidence, details)`. Validation errors (network /
parsing) are intentionally **non-destructive** — the image is kept on
exception so the pipeline never silently deletes data due to API outages
([pipeline.py:204-206](src/isaacnav/pipeline.py#L204-L206)).

### 5.4 Default model

`gemini-3-flash-preview` (configurable via `validation.gemini.model`).
API key is read from `GEMINI_API_KEY` environment variable.

---

## 6. Stage 2 — Segmentation

Source: [`src/isaacnav/masking/`](src/isaacnav/masking/)

The pipeline supports two segmentation strategies, swappable via
`masking.strategy` in the config. Both implement
[`BaseMaskingStrategy`](src/isaacnav/masking/base.py), returning
`MaskResult(mask, class_name, confidence, bbox, source_strategy)`.

### 6.1 `GroundedSam2Masking` — class-aware (default, recommended)

Source: [masking/grounded_sam2.py](src/isaacnav/masking/grounded_sam2.py)

Two-stage detection-then-segmentation:

1. **Grounding DINO** (HuggingFace `IDEA-Research/grounding-dino-base`,
   ~233 M params) takes the image + a text prompt and emits class-aware
   bounding boxes with confidence scores.
2. **SAM 2.1** (Hiera-Large checkpoint, `sam2.1_hiera_large.pt`) takes
   the boxes as prompts and produces high-quality masks for each box.

Default thresholds (from [pipeline.yaml:148-202](configs/pipeline.yaml#L148-L202)):

| Param            | Default | Purpose                 |
| ---------------- | ------- | ----------------------- |
| `box_threshold`  | 0.30    | Min DINO box-confidence |
| `text_threshold` | 0.25    | Min DINO text-relevance |
| `nms_threshold`  | 0.80    | IoU for cross-class NMS |

#### 6.1.1 Per-class prompt expansion (a key contribution)

For composite structural objects, a single-word DINO prompt biases
the detector toward one sub-part. Example:

- `"escalator."` → DINO often returns *only* the moving handrail ribbon
  or *only* a step tread.
- `"handrail."` → returns the rail itself but misses brackets / mounts.

The pipeline addresses this with **per-class prompt expansion** —
configured at [pipeline.yaml:174-202](configs/pipeline.yaml#L174-L202):

```yaml
class_prompts:
  "Escalator":
    - "escalator"
    - "escalator steps"
    - "escalator balustrade"
    - "escalator handrail"
    - "moving staircase"
  "Handrail":
    - "handrail"
    - "stair railing"
    - "guard rail"
    - "handrail mount"
  "Accessible Pedestrian Signal Button":
    - "pedestrian signal button"
    - "crosswalk push button"
    - "aps button housing"
    - "accessible pedestrian signal"
  ...
```

The runtime concatenates all phrases for a target class into a single
DINO prompt (`"escalator. escalator steps. escalator balustrade. ..."`).
DINO returns boxes for each part it recognizes; the masker then runs
SAM 2.1 on every box and **unions the resulting masks per class** into a
single coherent mask
([grounded_sam2.py:194-243](src/isaacnav/masking/grounded_sam2.py#L194-L243)).
Each unioned mask records:

- `confidence` = max of constituent DINO scores,
- `bbox` = enclosing rectangle of all constituent boxes,
- `coverage` = fraction of image area covered.

This expansion+union mechanism is **the workhorse** that lets a single
frame produce a clean object-level mask for a structurally complex
object (escalator, APS button) rather than a part-level fragment.

#### 6.1.2 Label-to-class assignment

When multiple classes are requested simultaneously,
[`_match_class()`](src/isaacnav/masking/grounded_sam2.py#L99-L116)
assigns each DINO label string to the requested class with the largest
word-overlap with any of that class's prompt phrases. (In practice the
pipeline is operated one class at a time, so this is a fallback path.)

### 6.2 `SamHeuristicMasking` — class-agnostic baseline

Source: [masking/sam_heuristic.py](src/isaacnav/masking/sam_heuristic.py)

Baseline approach for ablation:

- Runs **SAM ViT-H** automatic mask generator on the image (no text
  prompt, no detection).
- Filters candidate masks by coverage (5 %–70 % of image area).
- Scores remaining masks by a weighted sum of:
  - **Size score** — proximity to a `preferred_coverage` (default 30 %).
  - **Center score** — proximity of the mask centroid to the image center.
- Returns the single highest-scoring mask.

This baseline is class-blind and strongly biased to centered, medium-sized
objects — it is included primarily to demonstrate, in ablation, why
the class-aware Grounded-SAM-2 path is necessary for accessibility-class
objects (which are often off-center, partly occluded, or composite).

### 6.3 Mask selection and persistence

After segmentation, the orchestrator picks the **single best
(highest-confidence) mask** for the target class
([pipeline.py:316](src/isaacnav/pipeline.py#L316)) and writes it as
`output/.../mask.png`. The unioned mask is what ultimately drives
3D reconstruction — meaning the reconstructor never sees fragmented
part-level masks for composite objects.

---

## 7. Stage 3 — 3D Reconstruction

Source: [`src/isaacnav/reconstruction/sam3d.py`](src/isaacnav/reconstruction/sam3d.py)

### 7.1 Backbone: Meta SAM 3D Objects

The pipeline wraps Meta's
[**SAM 3D Objects**](https://github.com/facebookresearch/sam-3d-objects)
(included as a git submodule under `extern/sam-3d-objects/`). Given an
RGB image plus a binary object mask, SAM 3D Objects produces:

- **Gaussian-splat PLY** (`{stem}_gs.ply`) — for novel-view rendering
  evaluations.
- **Textured GLB mesh** (`{stem}.glb`) — the primary asset.
- An optional mesh-only PLY (kept off by default for stability).

The reconstructor is loaded lazily via
[`Sam3dReconstruction.load_model()`](src/isaacnav/reconstruction/sam3d.py#L29).

### 7.2 Hardware-specific stability fixes

Several non-obvious fixes are encoded in `load_model()` and matter for
reproducibility:

- `os.environ.setdefault("SPCONV_ALGO", "native")` — disables `spconv`'s
  *implicit_gemm* kernels on **RTX 5090 / Blackwell SM_120**, where they
  cause floating-point exceptions. Native algos pay a perf cost but do
  not crash.
- `os.environ.setdefault("LIDRA_SKIP_INIT", "true")` — sidesteps a
  lazy-import init bug.
- After reconstruction completes, the SAM 3D output dict is
  **explicitly `del`-ed** ([sam3d.py:91-94](src/isaacnav/reconstruction/sam3d.py#L91-L94))
  before the next stage to release stale CUDA tensors that otherwise
  segfault during subsequent format conversion under the same process.

These hardware caveats are paper-relevant: the conda environment
(`envs/environment.yaml`) is pinned to **CUDA 12.8 + PyTorch 2.7.1** with
custom-compiled `flash_attn==2.8.3`, `xformers==0.0.31`,
`bitsandbytes==0.49.2`, and `MoGe@a8c37341` for SM_120 compatibility.

---

## 8. Stage 4 — Format Conversion (USD/USDZ for Isaac Sim)

Source: [`src/isaacnav/conversion/`](src/isaacnav/conversion/)

### 8.1 Why USD?

Universal Scene Description (USD) is NVIDIA Isaac Sim's native scene
format. To make assets immediately usable for sim-based BLV navigation
training, the pipeline converts every reconstructed mesh to USD/USDZ.
USDZ is the packaged single-file form (mesh + textures + manifest).

### 8.2 `GlbToUsdConverter` — [conversion/glb_to_usd.py](src/isaacnav/conversion/glb_to_usd.py)

Primary path (used for SAM 3D's GLB outputs):

1. Loads the GLB with `trimesh`; concatenates submeshes.
2. Builds a USD `UsdGeom.Mesh` under `/World/mesh` with vertices, faces,
   and per-vertex normals.
3. **Texture handling** — if the GLB carries a PIL image in
   `mesh.visual.material.image`, it:
   - Saves the image to `textures/{stem}_diffuse.png`.
   - Defines a `UsdShade.Material` + `UsdPreviewSurface` shader.
   - Defines a `UsdUVTexture` reader pointing to the diffuse texture.
   - Binds the material to the mesh, with UV coordinates written as a
     `st` primvar.
4. **Vertex-color fallback** — if no texture exists (common for SAM 3D
   outputs from low-resolution sources), per-vertex RGB colors are
   written to the `displayColor` primvar.
5. **USDZ packaging** — the resulting `.usd` is wrapped via
   `UsdUtils.CreateNewUsdzPackage()` into `asset.usdz`.

### 8.3 `PlyToUsdConverter` — [conversion/ply_to_usd.py](src/isaacnav/conversion/ply_to_usd.py)

Fallback / pure-geometry path:

- Loads the PLY via `open3d`.
- If the PLY is a point cloud, rebuilds a triangle mesh via
  `create_from_point_cloud_alpha_shape(alpha=0.03)`.
- Writes vertices, triangles, normals, and per-vertex colors to USD.
- Packages USDZ as above.

### 8.4 Subprocess isolation

Conversion runs in a **subprocess**
([pipeline.py:_run_conversion_subprocess](src/isaacnav/pipeline.py#L214)).
This is not optional:

- `pxr` (USD Python bindings) and SAM 3D's `spconv` CUDA state interact
  badly in the same process — the parent process can segfault when
  `pxr` initializes after `spconv` has touched CUDA.
- The subprocess receives the converter config as a JSON string and
  prints a single JSON line of result metadata, which the parent parses.
- Timeout = 120 s. On failure, the parent logs the last 5 stderr lines
  but continues to the next image.

This is one of the few "engineering ugly bits" worth calling out in a
paper's reproducibility appendix.

---

## 9. Configuration: `configs/pipeline.yaml`

A single YAML file
([configs/pipeline.yaml](configs/pipeline.yaml)) drives the entire
pipeline. The schema is:

```yaml
pipeline: { name, version }
target_classes: [ "Escalator", "Handrail", ... ]   # or { source: object_categories.yaml }

crawling:
  enabled: true
  strategy: wikimedia | bing | duckduckgo | multi
  num_per_class: 250
  data_dir: data/input
  multi:      { backends: [...] }
  bing:       { min_size, threads }
  wikimedia:  { delay, max_retries }
  duckduckgo: { delay, max_retries }

validation:
  enabled: true
  strategy: gemini | none
  prompt:
    default: "Does this image contain a '{class_name}'?"
    class_prompts:
      "Escalator": "<accept/reject paragraph>"
      "Handrail":  "..."
  gemini: { model, api_key_env }

masking:
  strategy: grounded_sam2 | sam_heuristic
  device:   cuda | cpu
  sam_heuristic: { checkpoint, model_type, points_per_side, ... }
  grounded_sam2:
    grounding_model: IDEA-Research/grounding-dino-base
    sam2_checkpoint: extern/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
    sam2_model_cfg:  configs/sam2.1/sam2.1_hiera_l.yaml
    box_threshold:   0.30
    text_threshold:  0.25
    nms_threshold:   0.80
    class_prompts:   { "Escalator": [phrases...], ... }

reconstruction:
  strategy: sam3d
  seed: 42
  sam3d: { sam3d_path, config_path, compile }

conversion:
  output_formats: [glb, ply, usdz]
  glb_to_usd: { scale, up_axis, output_usdz }
  ply_to_usd: { scale, up_axis, alpha_shape_alpha }

output:
  base_dir: output
  save_masks: true
  save_original: true

batch:
  max_workers: 2
  retry_on_failure: true
  max_retries: 3
  checkpoint_interval: 10
```

### 9.1 Target classes (current)

The active configuration targets a focused subset of accessibility
infrastructure (rather than the full taxonomy). Currently enabled:

- `Escalator`
- `Handrail`

Commented-out (designed-for) classes include:

- `Elevator Button`, `Door button`
- `Bus Stop Sign`
- `Accessible Pedestrian Signal Button`
- `Turnstile`

The full intended taxonomy lives in
[configs/object_categories.yaml](configs/object_categories.yaml):

| Super-category   | Subcategories                                         |
| ---------------- | ----------------------------------------------------- |
| Tactile surfaces | `tactile_paving`, `guidance_blocks`, `hazard_warning` |
| Auditory signals | `accessible_pedestrian_signal`, `talking_sign`        |
| Physical markers | `bollard`, `handrail`, `curb_cut`, `ramp`             |
| Signage          | `braille_sign`, `tactile_map`, `high_contrast_marker` |

Each subcategory carries `description`, `aliases`, and a list of
`search_queries` — the latter feeding the crawling stage.

### 9.2 Top-level taxonomy (`object_categories_high_lev.yaml`)

A simpler 9-class flat list used as the de-facto "primary classes":
Elevator, Elevator Button, Door button, Door, Cross-walk, Traffic-signal,
Pedestrian signal, Bus-stop, Bus-stop sign stand.

---

## 10. Entry Points (CLIs)

### 10.1 [`scripts/run_pipeline.py`](scripts/run_pipeline.py)

A flat argparse CLI exposing five mutually-exclusive input modes:

| Mode              | Behavior                                                                   |
| ----------------- | -------------------------------------------------------------------------- |
| `--crawl-only`    | Crawl + validate, no 3D processing                                         |
| `--crawl`         | Full pipeline (crawl → validate → segment → reconstruct → convert)         |
| `--process-all`   | Process everything already in `data/input/{class}/`                        |
| `--image PATH`    | Single-image full processing (class inferred from parent dir or `--class`) |
| `--image-dir DIR` | Process every image in one class directory                                 |

Plus runtime overrides:

- `--masking-strategy {sam_heuristic, grounded_sam2}`
- `--reconstruction-strategy {sam3d}`
- `--crawl-strategy {multi, bing, wikimedia, duckduckgo}`
- `--no-validation` — keep all crawled images, skip Gemini.
- `--output DIR` — override `output.base_dir`.
- `--classes <names…>` — restrict to a subset.

### 10.2 [`scripts/run_batch.py`](scripts/run_batch.py)

JSON-manifest-tracked batch runner with four sub-commands:
`create-manifest`, `run`, `status`, `retry`. Backed by
[`BatchManifest`](src/isaacnav/data/manifest.py), which persists a list
of `JobItem(image_path, class_names, status, retry_count, ...)` to JSON
with status transitions (`pending → masking → reconstruction →
conversion → done | failed`). Supports resume-on-interruption and
bounded retries (`batch.max_retries`, default 3). This is the path
intended for large-scale dataset generation runs.

---

## 11. Environment & Reproducibility

[`envs/environment.yaml`](envs/environment.yaml) defines a single conda
environment named `isaacnav`. Key pinned versions:

- Python 3.11
- GCC/G++ 12.4 (needed for compiling pytorch3d, gsplat, GroundingDINO
  CUDA extensions for **SM_120**)
- CUDA toolkit 12.8 (from `nvidia/label/cuda-12.8.0`)
- PyTorch 2.7.1+cu128, TorchVision 0.22.1+cu128
- `flash_attn==2.8.3`, `xformers==0.0.31`, `bitsandbytes==0.49.2`
- 3D stack: `trimesh`, `open3d==0.18.0`, `point-cloud-utils`,
  `pymeshfix`, `xatlas`, `roma`
- SAM stack: `git+https://github.com/facebookresearch/segment-anything.git`,
  `transformers`, `supervision`, `torchmetrics`
- SAM3D core: `hydra-core`, `lightning==2.3.3`, `timm==0.9.16`,
  `MoGe @ git+...@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`,
  `sentence-transformers==2.6.1`

`envs/install.sh` is a one-shot installer that builds the conda env,
fetches submodule checkpoints, and editable-installs `isaacnav`.

---

## 12. Companion Resources Referenced in the Paper

### 12.1 GuideTWSI dataset ([docs/DATASET.md](docs/DATASET.md))

A separate, much larger 2D dataset of **Tactile Walking Surface
Indicators (TWSI)** (~39.5 K images), composed of:

- **RBar-22K** — 19,925 curated real-world tactile-bar images, drawn
  from SideGuide, Tenji10K, TP, and 69 Roboflow community repos. Five
  curation passes: dedup, schema standardization, bbox/mask QA,
  format conversion (RLE for SAM 2.1, polygon for YOLOv11-seg), human
  mask-overlay verification.
- **SDome-15K** — 15,010 photorealistic synthetic truncated-dome images
  generated with a UE4 + AirSim pipeline across 10 environments,
  diverse weather/lighting, 8 ADA-compliant dome types, with full
  modality stacks (RGB + depth + semantic + instance + bbox + intrinsics).
- **RDome-2K** — 2,466 real-world truncated-dome images collected by a
  **Unitree Go2** quadruped with an Intel RealSense D435 (70° down-tilt)
  in campus / residential / suburban / rural environments,
  human-verified Roboflow auto-segmentation.

Splits: 88 / 6 / 6 train/val/test for RBar-22K and SDome-15K;
RDome-2K is held out entirely for testing.

### 12.2 BLV objects 2D corpora ([DATASET.md, Appendix B](docs/DATASET.md))

A registry of ~40 Roboflow / Mendeley / academic 2D detection &
segmentation datasets covering elevators, elevator buttons, door buttons,
crosswalks, pedestrian signals, traffic signals, bus stops, doors —
used for training the 2D detection benchmarks under
[`benchmarks/detection/`](benchmarks/detection/).

---

## 13. Benchmarks (intended)

Per [README.md:148-164](README.md#L148-L164), three benchmark slots are
defined:

- **Object detection** — `benchmarks/detection/evaluate.py --model
  yolov8 --data data/`. Evaluates accessibility-object detection in
  simulation and real images.
- **Navigation** — `benchmarks/navigation/run_eval.py --scene
  indoor_mall --agent policy.pt`. Assesses policy performance using
  detected accessibility cues as observation features.
- **Simulation** (`benchmarks/simulation/`) — Isaac Sim scene assembly
  using NavAble USD assets.

These directories exist but are scaffolded; the dataset+pipeline are the
current paper-ready artifacts.

---

## 14. Design Decisions Worth Calling Out in a Paper

1. **Strategy pattern at every stage.** Each of crawl, validate, mask,
   reconstruct, convert is an `ABC` with a fixed dataclass result type.
   The orchestrator only knows registry names. This makes the pipeline a
   *test bench* rather than a frozen system — a paper can ablate
   `sam_heuristic` vs. `grounded_sam2`, swap reconstructors (e.g. add
   TripoSR / Trellis), or change crawler mixes by editing one YAML field.

2. **Class-locked single-image processing.** Each image carries exactly
   one class label (from parent directory or CLI). All downstream stages
   restrict their attention to that class. This dramatically simplifies
   the reasoning about the mask the reconstructor receives, and matches
   the way humans curate accessibility-asset libraries.

3. **VLM-driven class disambiguation as an explicit pipeline stage.**
   Open-vocabulary search-based crawling produces enormous concept drift
   (a search for "turnstile" returns subway tripod gates, optical
   flap-barriers, full-height security turnstiles, and CAD diagrams).
   We push the disambiguation work to the VLM rather than the segmenter
   or the human curator. Paragraph-form per-class accept/reject prompts
   ([pipeline.yaml:95-123](configs/pipeline.yaml#L95-L123)) function as
   a fine-grained, machine-checkable taxonomy — they are arguably the
   most paper-novel artifact in this repo.

4. **Per-class DINO prompt expansion + mask union.** Composite
   structural objects (escalator, handrail, APS button) are not a single
   blob to an open-vocabulary detector. We hand DINO multiple
   sub-component phrases, then *fuse* the resulting masks back into one.
   This is a cheap and effective fix for a well-known failure mode of
   text-prompted segmenters on composite objects.

5. **Subprocess isolation between SAM 3D and USD.** A real engineering
   decision driven by `spconv` × `pxr` interactions on Blackwell. Worth
   a one-paragraph methods note in any reproducibility-aware paper.

6. **Hardware-aware reconstruction.** The pipeline targets **RTX 5090 /
   Blackwell SM_120**, and several env-vars and pinned package versions
   exist solely to make `spconv` / `flash_attn` / `xformers` not crash
   on the new arch. This is paper-relevant because most existing
   image-to-3D papers benchmark on Ampere/Hopper.

7. **Dataset provenance.** Every crawled image is paired with a sidecar
   JSON containing source URL, license string, and attribution. The
   dataset can be released under CC-BY 4.0 with verifiable per-asset
   attribution chains.

---

## 15. Known Limitations / Open Items

(Useful as a "limitations" / "future work" section seed.)

- **Dataset scale.** `target_classes` is currently restricted to two
  active classes (`Escalator`, `Handrail`); the broader taxonomy of
  accessibility infrastructure is configured but not yet fully
  generated.
- **VLM rejection rate is not formally measured.** Paper would benefit
  from a precision/recall study of `gemini-3-flash-preview` against a
  small human-labeled validation set per class.
- **Single-frame reconstruction limits.** SAM 3D Objects is a
  single-image model; back-of-object geometry is hallucinated.
  Multi-view extension (e.g., over crawled image *sets* of the same
  physical object) is a natural next step.
- **No automatic mesh QC.** `pymeshfix`, `xatlas`, `point-cloud-utils`
  are present in the environment but not wired into a post-recon QC
  filter. A paper-grade pipeline should add manifold-checks,
  scale-sanity, and bbox-vs-class plausibility filters.
- **USD scale is hard-coded to 1.0 m.** Per-class real-world dimensions
  (e.g., a 2.6 m escalator handrail height) are not yet inferred.
- **Crawler license heterogeneity.** Wikimedia images carry verifiable
  CC licenses; Bing/DDG images are tagged `"Unknown"`. Final dataset
  release should restrict redistribution to Wikimedia-sourced assets
  unless individual licenses are verified.
- **Class-aware NMS is global.** When multiple classes share one image,
  a high-confidence box for class A can suppress a marginal box for
  class B. In practice the pipeline is run one-class-at-a-time so this
  does not matter, but it is a paper-worthy gotcha.

---

## 16. Quick-Reference Glossary (for paper writing)

| Term                 | Meaning                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **APS**              | Accessible Pedestrian Signal — audio-tactile crosswalk button.                                                                                    |
| **TWSI**             | Tactile Walking Surface Indicator — generic term for tactile paving / truncated domes / directional bars.                                         |
| **Truncated dome**   | The North American style of detectable warning surface — raised circular bumps at curb cuts and platform edges.                                   |
| **Tenji block**      | The Japanese-origin directional / warning tactile paving (yellow with bars or domes).                                                             |
| **Grounding DINO**   | IDEA-Research's open-vocabulary text-prompt object detector. We use `grounding-dino-base` (~233 M).                                               |
| **SAM / SAM 2.1**    | Meta's Segment Anything Model. v1 ViT-H is used in the heuristic baseline; v2.1 (Hiera-L) is used as the box-prompt segmenter for Grounded-SAM-2. |
| **SAM 3D Objects**   | Meta's single-image image-to-3D mesh model (gaussian splat + textured GLB outputs).                                                               |
| **USD / USDZ**       | Pixar's Universal Scene Description; USDZ is the zip-packaged single-file form used by Apple AR Quick Look and NVIDIA Isaac Sim.                  |
| **Gaussian splat**   | Per-vertex 3D-Gaussian representation used by SAM 3D for novel-view rendering.                                                                    |
| **MoGe**             | Microsoft's Monocular Geometry estimator, used internally by SAM 3D.                                                                              |
| **prompt expansion** | Our per-class strategy of feeding Grounding DINO multiple sub-component phrases for composite objects, then unioning their masks.                 |
| **Isaac Sim**        | NVIDIA's Omniverse-based robotics simulator; consumes USD assets natively.                                                                        |

---

## 17. File-Level Index (for an LLM reader)

```
src/isaacnav/pipeline.py                       # AssetPipeline orchestrator (~490 LOC)
src/isaacnav/__init__.py                       # __version__ = "2.0.0"

src/isaacnav/crawling/base.py                  # BaseCrawler + CrawlResult
src/isaacnav/crawling/wikimedia.py             # WikimediaCommonsCrawler
src/isaacnav/crawling/bing.py                  # BingCrawler (icrawler)
src/isaacnav/crawling/duckduckgo.py            # DuckDuckGoCrawler (ddgs)
src/isaacnav/crawling/multi.py                 # MultiCrawler (concurrent + dedup)

src/isaacnav/validation/base.py                # BaseValidator + ValidationResult
src/isaacnav/validation/gemini.py              # GeminiValidator (google.genai)

src/isaacnav/masking/base.py                   # BaseMaskingStrategy + MaskResult
src/isaacnav/masking/grounded_sam2.py          # GroundedSam2Masking (DINO + SAM2.1)
src/isaacnav/masking/sam_heuristic.py          # SamHeuristicMasking (SAM ViT-H)

src/isaacnav/reconstruction/base.py            # BaseReconstructionStrategy + ReconstructionResult
src/isaacnav/reconstruction/sam3d.py           # Sam3dReconstruction (Meta SAM 3D Objects)

src/isaacnav/conversion/base.py                # BaseConverter + ConversionResult
src/isaacnav/conversion/glb_to_usd.py          # GlbToUsdConverter (trimesh + pxr)
src/isaacnav/conversion/ply_to_usd.py          # PlyToUsdConverter (open3d + pxr)

src/isaacnav/data/output_layout.py             # OutputLayout (per-object dirs + metadata.json)
src/isaacnav/data/manifest.py                  # BatchManifest + JobItem

scripts/run_pipeline.py                        # Single-image / per-class CLI
scripts/run_batch.py                           # Manifest-tracked batch CLI

configs/pipeline.yaml                          # Master config (v2.2)
configs/object_categories.yaml                 # Detailed taxonomy w/ search queries
configs/object_categories_high_lev.yaml        # 9-class high-level list

envs/environment.yaml                          # Conda env (CUDA 12.8 / SM_120)
envs/install.sh                                # One-shot installer

extern/sam-3d-objects/                         # Submodule: Meta SAM 3D Objects
extern/Grounded-SAM-2/                         # Submodule: GroundingDINO + SAM 2.1
```

---

*End of overview.*
