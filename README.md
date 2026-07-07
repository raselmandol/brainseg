[![Latest Release](https://img.shields.io/github/v/release/raselmandol/brainseg?label=Release&logo=github)](https://github.com/raselmandol/brainseg/releases/latest)
[![Issues](https://img.shields.io/github/issues/raselmandol/brainseg?logo=github)](https://github.com/raselmandol/brainseg/issues)
[![License](https://img.shields.io/github/license/raselmandol/brainseg?color=yellow)](https://github.com/raselmandol/brainseg/blob/main/LICENSE)
[![CI/CD](https://github.com/raselmandol/brainseg/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/raselmandol/brainseg/actions/workflows/ci-cd.yml)
[![PyPI](https://img.shields.io/pypi/v/brainseg-ai?label=PyPI&logo=pypi&logoColor=white&cacheSeconds=60)](https://pypi.org/project/brainseg-ai/)
![PyPI Downloads](https://img.shields.io/pepy/dt/brainseg-ai)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyQt6](https://img.shields.io/badge/UI-PyQt6-41CD52?logo=Qt&logoColor=white)](https://doc.qt.io/qtforpython/)

**brainseg** is a PyQt6 desktop application for brain MRI abnormality segmentation. It has a friendly multi-view interface with a PyTorch UNet backend, making it easy to inspect images, run inference with custom `.pth` models, compare against ground truth, and review statistics in one place.
![brainseg GUI Preview](https://raw.githubusercontent.com/raselmandol/brainseg/refs/heads/main/screenshots/brainseg151_1.gif)



---

## Table of Contents

1. [Highlights](#highlights)
2. [Getting Started](#getting-started)
3. [Using the App](#using-the-app)
4. [Model Management](#model-management)
5. [Customization & Settings](#customization--settings)
6. [Troubleshooting & Notes](#troubleshooting--notes)
7. [Development Workflow](#development-workflow)
8. [Roadmap](#to-do)
9. [Contributing](#contributing)
10. [Citing BrainSeg](#citing-brainseg)
11. [License](#license)

---

## Highlights

- **Three synchronized canvases** for original, mask, and highlighted overlay views with zoom/pan controls.
- **Brightness and contrast tuning** that preserves zoom state for precise inspection.
- **One-click segmentation** using your own `.pth` checkpoint; progress bar shows inference status.
- **Ground-truth loader** with thumbnail preview and automatic resizing for metric comparison.
- **Research-style statistics window** (Dice, Jaccard, Hausdorff, latency, memory, trend plots).
- **Theme, accent, and custom color palette** so users can adapt the UI to ambient lighting.
- **Robust error handling** that surfaces detailed model-loading issues via dialogs.

Browse the complete gallery in [SCREENSHOTS.md](SCREENSHOTS.md).

---

## Getting Started

### Prerequisites

- Windows (tested) with Python **3.10**
- PyTorch build that matches your hardware (CPU-only works; CUDA recommended for speed)
- Qt runtime provided by PyQt6 (installed automatically through `requirements.txt`)

### Installation

#### PyPI (Recommended)

```powershell
pip install brainseg-ai
brainseg
```

This installs the brainseg-ai package and makes the `brainseg` console entry available system-wide.

> [!Note]
> The previous torch `c10.dll` import error is resolved in the current release, so `pip install brainseg-ai` completes without warnings on supported Python >=3.10 environments.

#### Clone & Install from Source

```powershell
git clone https://github.com/raselmandol/brainseg.git
cd brainseg
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


### Build

```powershell
pip install -e .
```


### Launching the app

```powershell
python -m brainseg
# or run the legacy script if you prefer
python brainseg/main.py
```

> [!Note]
> Tip: `pip install -e .` adds the `brainseg` console entry point so you can launch the app from anywhere.
> After a PyPI install (`pip install brainseg-ai`), you can simply run `brainseg` from any shell.

---

## Using the App

1. **Open Image** – load an MRI slice (`.png`, `.jpg`, `.tif`, `.bmp`). The original view updates immediately, honoring current brightness/contrast settings.
2. **Adjust Brightness/Contrast** – use the sliders on the left dock; zoom/pan state stays intact.
3. **Load Ground Truth (optional)** – import a mask (`.png`, `.tif`). A thumbnail appears under the filename label.
4. **Select Model File** – pick a segmentation checkpoint (`.pth`). The app validates the file right away and reports issues in a modal dialog if the architecture is incompatible.
5. **Run Segmentation** – click *Run Segmentation* or press `Ctrl+Alt+R`. The progress bar indicates inference status, and statistics are recorded automatically.
6. **Save Results** – export the mask or highlighted overlay using the corresponding buttons or File menu actions.

---

## Model Management

- Default checkpoint: `brain_segmentation_model.pth`.
- Validation: selecting a model triggers a quick load test; clear dialogs explain missing keys or shape mismatches.
- Swapping models: re-open *Select Model File* at any time; the new model is cached for subsequent runs.

---

## Customization & Settings

- **Theme toggle** (light/dark) plus accent selector (Azure, Emerald, Amber, Rose).
- **Custom palette** picker applies a brand color across toolbars, docks, and controls in real time.
- **Image adjustments** -- the Settings dialog mirrors the dock sliders, so you can tweak values from either location.
- Preferences currently apply per session; wiring to `QSettings` is on the roadmap.

---

## Troubleshooting & Notes

- **Invalid model**: You'll see "Model Load Error" with the traceback if the `.pth` doesn't match the expected UNet shape.
- **Inference failure**: "Segmentation Error" dialogs include the captured stack trace. Re-select a compatible model or verify PyTorch/torchvision versions.
- **Large images**: Internally resized to the nearest multiple of 32 to satisfy encoder constraints.
- **Performance tips**: Use the CUDA-enabled PyTorch wheel when a GPU is available; CPU runs are slower but supported.

---

## Development Workflow

```powershell
# Install in editable mode
pip install -e .

# Run the GUI from source
python -m brainseg

# Quick sanity check (syntax)
python -m py_compile brainseg\model.py brainseg\main_window.py
```

---

## To-do

- [ ] Persist user preferences (theme, accent, last model path) via `QSettings`.
- [ ] Allow background model validation to keep the UI responsive with >1 GB checkpoints.
- [ ] Add CLI hooks or REST mode for batch inference.
- [ ] Extend statistics window with export-to-CSV and per-run notes.
- [ ] ONNX Runtime support (load `.onnx` models alongside `.pth`)
- [x] Segmentation list
- [ ] Asynchronous (non-blocking) segmentation via a worker thread
- [ ] Fetch models from remote sources
- [ ] Support semantic and instance segmentation (parallel workflows)

---

## Contributing

Pull requests and feature suggestions are welcome. Please file an issue describing the change, branch from `main`, and keep PRs focused. For major UI adjustments, sharing mockups or screenshots helps align expectations.

---

## Citing BrainSeg

If this project supports your research or clinical workflow, please cite it using the metadata in [CITATION.cff](CITATION.cff):

> Mandol, Md Rasel. **BrainSeg: Deep Learning-based Brain MRI Segmentation and Annotation.** Version 1.5.5, 3 Jan 2026. MIT License. https://github.com/raselmandol/brainseg

GitHub's "Cite this repository" button also reads the same CFF file, so you can export BibTeX or other formats directly.

---

## License

This project is released under the [MIT License](LICENSE).


**Maintainer:** Md. Rasel Mandol (Smart Systems & Connectivity Lab, NIT Meghalaya)