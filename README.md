# VCEM Analyser

A Rust desktop application for inspecting microscopy images, experimenting with cellular automata (CA), and combining feature masks. The repository also contains Python notebooks and a script for splitting image stacks into individual frames.

## Setup

### Requirements

- A graphical desktop session with working graphics drivers.
- A current stable Rust toolchain and Cargo. The project uses Rust edition 2024; its `eframe` dependency requires Rust 1.92 or newer. Use current stable for compatibility with the full dependency set.
- A C/C++ compiler, Clang/libclang, and OpenCV development libraries. OpenCV is a declared Cargo dependency and needs its native libraries during a fresh build.
- Your own images. No dataset download is required to start; the `data/` directory is excluded from Git.

Install Rust using the [official Rust installation instructions](https://rust-lang.org/tools/install/). If you already use rustup:

```sh
rustup update stable
rustup default stable
rustc --version
cargo --version
```

### Linux dependencies

On Arch Linux / Omarchy:

```sh
sudo pacman -S --needed base-devel clang llvm opencv pkgconf cmake libxcb libxkbcommon openssl
```

On Ubuntu / Debian:

```sh
sudo apt-get update
sudo apt-get install build-essential pkg-config cmake clang libclang-dev libopencv-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```

The native dependency requirements are documented upstream by [eframe](https://github.com/emilk/egui/tree/0.35.0/crates/eframe) and [opencv-rust](https://github.com/twistedfall/opencv-rust/tree/v0.100.1). For macOS or Windows, follow their platform setup instructions; the commands above cover Linux.

### Build and run

Clone or download this repository, then run these commands from its root directory (the directory containing `Cargo.toml`):

```sh
cargo run --locked --release
```

The first build downloads and compiles dependencies, including the OpenCV bindings, and can take several minutes. Release builds are recommended for image processing and CA experiments.

To build once and launch the executable directly on Linux:

```sh
cargo build --locked --release
./target/release/vcem
```

For a development build:

```sh
cargo run --locked
```

Python is optional and is not needed to run the desktop app. The app starts with an empty frame list; images are opened through its interface.

## Open and inspect images

1. Click **Add Image** in the **Frames** panel.
2. Browse folders, or enter a directory in **Location** and click **Go**.
3. Select images and click **Add selected**. **Select all** selects the supported images in that folder.
4. Click a frame in the left panel to display it. Use **Inspect** for the image processing and mask workflow.

The picker accepts PNG, JPEG (`.jpg`, `.jpeg`), and TIFF (`.tif`, `.tiff`). The decoder also supports grayscale floating-point TIFFs and normalizes them for processing. To work through the frames of an image stack individually, first extract them with the optional Python script described below.

### Viewer controls

| Control | Action |
| --- | --- |
| **Fit** | Fit the image to the viewer. |
| Mouse wheel | Zoom around the pointer. |
| Right mouse drag | Pan the image. |
| Left click | Select the corresponding intensity histogram bin. |
| Left mouse drag | Select histogram bins represented in a rectangular region. |
| **Clear Selection** | Clear the histogram selection. |

When **Draw review regions** is enabled, left dragging draws a review rectangle instead of selecting histogram bins.

### Processing resolution and downsampling

Large images are reduced to a preview of at most **1,048,576 pixels**, preserving aspect ratio. For example, a 4096 × 4096 image becomes 1024 × 1024. Histograms, CA runs, and masks operate on this prepared preview; zooming the viewer does not increase processing resolution.

Under **Downsampling**, choose a method and click **Apply Downsampling**:

| Method | Behavior |
| --- | --- |
| **Area average** | Averages the source area represented by each output pixel. Default for new sessions. |
| **Nearest neighbor** | Samples individual source pixels without averaging. |
| **Triangle (bilinear)** | Interpolates neighboring source values. |

Images already below the preview limit are not resized. Older saved configurations may show **Original behavior (legacy)**: nearest neighbor for floating-point TIFFs and triangle filtering for other decoded images.

### Prepare an image

The right panel provides rotation, histogram generation, median or Gaussian denoising, inversion, distribution equalization, and gamma adjustment. Change a control and click its corresponding **Apply** button to process the image.

The processing order is:

```text
Decode / normalize / downsample
  → rotation → denoising → inversion → equalization → gamma → CA
```

Gamma values above 1 darken normalized intensities. **Reset All Augmentations** clears the applied augmentations and CA result while keeping the applied downsampling method. Processing does not overwrite the source image.

## Run the cellular feature filter

In **Inspect**, scroll to **Cellular Feature Filter**:

1. Choose **Dark features** or **Bright features** to set the seed polarity.
2. Choose a sensing kernel size and preset, or paint a custom kernel using positive, zero, and negative weights.
3. Select a growth response: **Gaussian target**, **Survival band**, **High-pass threshold**, or **Low-pass threshold**. Adjust its **Centre** and **Width**.
4. Set **Growth**, **Decay**, **Time step**, and **Duration** under **Evolution**.
5. Click **Apply Feature Filter** and inspect the overlay. **Remove CA Overlay** removes that result from the viewer.

**Omni-directional pass** evaluates quarter-turn kernel orientations (0°, 90°, 180°, and 270°), rather than every possible angle. Kernel dimensions are measured in prepared-image pixels, so changing downsampling can change which features a configuration detects.

This filter uses image evidence throughout evolution. Its output represents candidate features selected by the settings; a mask alone does not establish that a feature is an amyloid fibril or identify the chemical source of contamination.

### Save and reuse processing settings

Under **Experiment Configuration**, set a JSON file path (default: `vcem-experiment.json` in the directory from which the app was launched).

- **Save Applied** saves the settings that produced the current image and CA result. Unapplied control edits are not saved.
- **Load** reads a configuration into the controls.
- **Apply Saved** runs that saved configuration on the selected image.

This file stores processing settings, including downsampling and CA parameters. Use a mask recipe to save a collection of mask layers and combinations.

## Capture and combine masks

The **Mask layers & review** section in Inspect and Automata Lab lets you retain several CA results for the same image and combine their binary selections. Inspect, discrete Lab, and Lenia Lab layers can be combined in the same workspace.

### Capture layers

1. Run a CA configuration using **Apply Feature Filter**.
2. Enter a descriptive name, such as `Fibril candidates`.
3. Click **Capture current CA as layer**.
4. Adjust that layer's **Mask threshold** and click **Show mask**.
5. Run another configuration, name it (for example, `Suspected contamination`), and capture another layer.

Each captured layer retains its final activation values and applied configuration. A pixel is selected when its activation is **greater than or equal to** the mask threshold. Changing the threshold updates dependent combinations without rerunning the CA. A threshold of zero selects every pixel, including zero activations.

### Boolean operations

Select layers **A** and **B**, choose an operation, enter a name for the result, and click **Create combination**.

| Operation | Selected pixels |
| --- | --- |
| **AND** | In both A and B. |
| **OR** | In either A or B, including their overlap. |
| **XOR** | In exactly one of A or B. |
| **A minus B** | In A but outside B. |
| **Create NOT A** | Outside A, across the whole prepared image; B is not used. |

Combination results become layers and can be used in further combinations. For example:

```text
Exclusions = Suspected contamination OR Blob candidates
Filtered fibril candidates = Fibril candidates minus Exclusions
```

Use **Compare A / B** before subtraction: teal shows A only, magenta shows B only, and blue shows overlap. Inspect the overlap for features you want to retain. **Current CA** returns to the latest CA overlay; **Image only** shows the image without a mask overlay.

Up to 16 layers are supported per image, including combinations. A layer referenced by another layer cannot be removed until its dependent combinations are removed.

### Review regions

Expand **Review regions**, enable **Draw review regions**, choose a category, and drag a rectangle on the image. Categories are **Likely fibrils**, **Suspected contamination**, and **Uncertain / overlap**.

The reported A, B, and overlap percentages describe selected pixels inside each rectangle. They measure coverage, not classification accuracy.

### Save and rebuild a mask workspace

Expand **Save / rebuild mask recipe**, choose a JSON path, and click **Save recipe** before closing the app. Mask workspaces are retained while switching images during a session, but are not automatically saved on exit.

A recipe stores layer names, CA configurations, thresholds, Boolean operations, review regions, and source identity. It does not embed source images or activation arrays. The default filename includes the image name and a path-derived identifier.

- **Load & rebuild** loads a recipe and reruns its CAs for the selected source image.
- **Rebuild for current view** rebuilds the current workspace after changing rotation or downsampling.
- Rebuilding uses the current rotation and downsampling, while retaining each captured layer's other processing settings.
- Masks from different images cannot be combined. Recipes check the source path, file size, and modification time; moving or changing the source can prevent a recipe from loading.
- Incompatible masks are hidden until rebuilt. Save the old recipe before choosing **Start new workspace (clear old masks)** if you want to keep it.

The current interface saves recipes; it does not export binary masks as PNG or TIFF files. Lab layers record their applied image preparation, automaton, and generation count. New recipes use version 2; existing version 1 Inspect recipes remain readable.

## Automata Lab

The **Automata Lab** tab provides a simulation workspace with **Discrete map** and **Continuous Lenia** models. Its **Image preparation** section has the same downsampling, rotation, histogram, denoising, inversion, equalization, and gamma controls as Inspect. The viewer also supports histogram selection, mask previews, and review rectangles.

1. Add an image and prepare it directly in the Lab. Apply each image change using its **Apply** button.
2. Choose **Discrete map** or **Continuous Lenia** and configure its controls.
3. Click **Initialize Lab** to create generation zero from the prepared image.
4. Use **Step** or **Play** / **Pause** to evolve the state. **Playback** controls cycles per step, playback speed, and checkpoint interval.
5. After editing model settings, click **Apply Settings & Reset**. **Reset** returns the applied model to generation zero.
6. Pause and wait for the current step to finish. In **Mask layers & review**, enter a layer name and click **Capture Lab state as layer**. Use the same thresholds, Boolean combinations, comparison view, and recipe controls as Inspect.

The Lab evolves autonomously after initialization. Changing the selected image or applying new preprocessing invalidates the current Lab session. Captured mask layers remain in their image workspace; geometry changes require a rebuild.

### Save and replay Lab configurations

Under **Lab Configuration**, choose a JSON path (default: `vcem-lab.json`).

- **Save Applied** saves the image preparation, applied automaton settings, and current generation count. Pause first; unapplied edits are not included.
- **Load** stages a saved configuration in the controls.
- **Apply Saved** prepares the selected image, initializes the saved automaton, and runs to the saved generation in the background.

Lab configurations can be applied to another image. Mask recipes remain bound to their source image. Save a mask recipe to retain multiple captured configurations and their combinations.

The **Timeline** allows checkpoint navigation; scrubbing to an earlier generation discards later history. **State bookmarks** retain named states during the session. Configuration files and mask recipes rebuild runs from generation zero; they do not save the timeline or bookmarks.

### Discrete kernels and rules

Choose the state count `q` (2–64). Initial image intensities are quantized to state IDs from `0` to `q − 1`; the initial-state histogram shows their distribution.

The **Sensing Kernel** section uses the same editor as Inspect: sizes 3–15, **Disk**, **Ring**, **Line**, **Cross**, **Crescent**, and **Clear** presets, plus an angle control for Line and Crescent. The empty, positive, and negative paint buttons sit vertically beside the grid. Click or drag to paint; right-click cycles cell types. The white outline marks the centre. Reducing the grid size removes samples outside the new grid.

**Omni-directional pass** evaluates 0°, 90°, 180°, and 270° rotations in each generation and keeps the highest mapped output state. It preserves sample order within each orientation, including for ordered rules. It has no effect on Direct rules, which ignore the kernel.

| Cell | Meaning |
| --- | --- |
| Empty | Ignore this position. |
| Positive / green | Read the neighboring state unchanged. |
| Negative / red | Reverse the state order: read `q − 1 − state`. For two states, this swaps 0 and 1. |

Negative cells reverse discrete evidence; they do not introduce negative state IDs. Rule lookup happens after this reversal.

| Rule | Input to the mapping |
| --- | --- |
| **Direct** | The current cell's own state. This rule does not use the kernel. |
| **Aggregate** | The sum or rounded mean of the sensed states. |
| **Ordered neighbourhood** | A base-`q` index from the kernel's ordered samples. Hover cells to see offsets; painting orders samples row by row. |
| **Presence triggered** | Trigger states present among the sensed samples. The chosen reduction resolves simultaneous matches; the fallback uses the current cell's state. |

The mapping graph shows inputs and their next states. Large tables are sampled for plotting. Interactive authoring is limited to 65,536 mapping entries; reduce states or samples if an ordered table exceeds that limit.

Under **Random generators**, set a seed, kernel density, and negative fraction. **Random kernel** generates sample positions and signs; **Random rule mapping** generates the active rule's outputs (including presence transitions and fallback). The same seed and settings reproduce a result. Click **Next seed** for another candidate. Ordered random kernels are capped to fit the table limit. Apply the generated settings before running them.

For mask thresholds, discrete states are normalized as `state / (q − 1)`. Higher state IDs therefore have higher mask activation; choose the rule mapping and threshold accordingly.

### Generate discrete CA candidates from a region

In **Automata Lab → Discrete map → Sensing Kernel → Candidates from region**:

1. Set the state count and kernel size. Prepare the image using the image controls if needed.
2. Click **Select example region**, then drag a rectangle around representative features on the image. Select a region with surrounding image content for comparison.
3. Click **Generate candidates**. Fitting runs in the background and can be cancelled.
4. Inspect up to four previews. Teal indicates selected pixels; the amber border marks your example. Each card reports the sampled coverage inside the example and in nearby image content.
5. Click **Preview on image** to load a candidate's kernel and rule mapping and run one generation on the full prepared image. This replaces the current Lab run; previously captured mask layers remain available.
6. Once the run finishes, use **Capture Lab state as layer** in **Mask layers & review**. Combine and save it using the existing mask recipe controls. You can also edit the candidate's kernel or mapping and run further experiments.

The generator compares local discrete-state patterns inside the example with nearby patterns outside it. It considers directional contrast kernels, a local-intensity kernel, a seeded kernel, and your current kernel when present. It fits aggregate-sum rules and ranks them by example coverage minus surrounding coverage. Candidates use Clamp boundaries and a single orientation; all their controls remain editable.

The region is an example, not a pixel-perfect annotation. The scores describe fitting-region contrast, not accuracy or biological identity. If the region and surroundings cannot be distinguished, try a tighter region, a different state count, or different preprocessing. Selecting the entire image leaves no surrounding reference and is rejected.

Regions and candidate previews are temporary. Image changes clear them and cancel outstanding generation. Capture the chosen results and save a mask recipe to retain them.

### Overlay colours and playback layout

CA results and individual masks use teal. Pixel selection and intensity highlights use amber. In **Compare A / B**, A-only is teal, B-only is magenta, and overlap is blue. The simulation controls reserve fixed rows for status, progress, and diagnostics so updates do not move the controls while running; hover truncated diagnostics to read the full text.

### Continuous Lenia visualizations

The kernel heatmap shows the spatial sensing weights; hover a cell for its normalized weight. The growth graph plots response against the weighted neighborhood state. Positive response grows the state, negative response reduces it, and the time step controls the change per generation.

See [Automata Lab architecture](docs/automata-lab-architecture.md) for the engine design and staged roadmap; some sections describe planned work.

The **Segment** and **Measure** mode buttons currently share the normal analysis view and do not provide separate workflows.

## Optional Python tools

From the repository root, create an environment and install the notebook dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

Open `Cryo-EM TEM Image Experiments.ipynb` in Jupyter and update its data paths for your local images. These dependencies are separate from the Rust application's native build dependencies.

### Split an image stack into frames

The script [segment_amyloid_fibrils_into_frames.py](segment_amyloid_fibrils_into_frames.py) extracts frames from images readable by Pillow. Despite its name, it does not detect or segment fibrils.

```sh
mkdir -p data/raw/amyloid-fibrils data/frame-segmented/amyloid-fibrils
```

Place input image stacks in `data/raw/amyloid-fibrils/`, then run:

```sh
python segment_amyloid_fibrils_into_frames.py
```

Frames are written to `data/frame-segmented/amyloid-fibrils/` and the manifest to `data/amyloid-fibrils-manifest.json`. Keep the input directory limited to readable image files. Rerunning the script can overwrite existing output frames and the manifest.

## Development checks

```sh
cargo check --locked
cargo test --locked
```

The normal test suite uses synthetic fixtures. Tests requiring a local RELION TIFF are ignored by default; their expected fixture paths are recorded in the test source. Once dependencies are cached, add `--offline` to Cargo commands to avoid network access.

| Path | Purpose |
| --- | --- |
| `src/main.rs` | Desktop entry point. |
| `src/app.rs`, `src/app/masks.rs` | Interface and background job orchestration. |
| `src/analysis.rs`, `src/downsampling.rs` | Image decoding and preprocessing. |
| `src/automata.rs`, `src/discrete_automata.rs`, `src/continuous_automata.rs` | CA engines. |
| `src/automata_lab.rs`, `src/lab_configuration.rs`, `src/app/lab.rs` | Lab sessions, saved experiments, generators, and controls. |
| `src/masks.rs` | Mask recipes and Boolean composition. |
| `src/discrete_candidates.rs`, `src/app/candidates.rs` | Region-guided candidate fitting, previews, and selection. |
| `src/app/kernel_editor.rs` | Shared Inspect and discrete kernel editor. |
| `requirements.txt` | Optional Python dependencies. |

## Troubleshooting

- **`cargo` not found:** finish the Rust installation and reopen your terminal so Cargo is on `PATH`.
- **Rust version too old:** update the stable toolchain with rustup and check `rustc --version`.
- **OpenCV or libclang cannot be found:** install the native development packages above. For a custom installation, see the [opencv-rust environment variables](https://github.com/twistedfall/opencv-rust/tree/v0.100.1#environment-variables), particularly `OpenCV_DIR`, `PKG_CONFIG_PATH`, and the linked libclang setup instructions. Installing Python's `opencv-python` does not supply the Rust build's native development setup.
- **Window or graphics initialization fails:** launch from a graphical desktop session and check the system's graphics drivers. The executable opens a native window, not a web server.
- **CA processing is slow:** use a release build and reduce kernel size or duration. Mask recipe rebuilding runs the captured CA configurations again.
- **A changed control has no visible effect:** click its **Apply** button. For saved settings, use **Apply Saved** after **Load**; for a captured mask, use **Show mask**.
