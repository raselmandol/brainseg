# brainaseg

Brain abnormality segmentation application (PyQt6).

## Structure

- `brainaseg/`
  - `__init__.py` — Package init
  - `canvas.py` — Graphics canvas and UI components
  - `image_utils.py` — Image utility functions
  - `model.py` — Model loading and inference
  - `worker.py` — Worker thread for segmentation
  - `main_window.py` — Main window and application logic
  - `main.py` — Entry point for the app
- `setup.py` — Package setup
- `requirements.txt` — Python dependencies

## Usage

Install with:
```
pip install .
```
Run with:
```
brainaseg
```
