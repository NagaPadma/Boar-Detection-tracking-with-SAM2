# Boar Detection and Tracking with SAM 2

We use **OWLv2** to detect animals and save bounding boxes. Then **SAM 2** uses those boxes to segment and track the animals across frames.

Run the commands below from the folder containing `detect.py`, `segment.py` and `requirements.txt`.

## 1. Create the environment and install requirements

```bash
conda create -n boar_demo python=3.11 -y
conda activate boar_demo
python -m pip install -r requirements.txt
```

If `boar_demo` already exists, skip the create command and activate it.

## 2. Install official SAM 2 and download the checkpoint

Use official SAM 2 for adding new animals after tracking starts.

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_official

```
If this source and checkpoint are already installed/downloaded, reuse their paths in `segment.py`.

## 3. Object detection — detect.py

At the top of `detect.py`, set the input video:

```python
INPUT = Path('/absolute/path/to/boar_video.mp4')
MAX_FRAMES = 50       # Short test; None processes the full input
FRAME_STEP = 1       # Save every frame
DETECT_EVERY = 5     # Detect every fifth saved frame
```

A folder of numeric-named JPEGs can also be used:

```python
INPUT = Path('/absolute/path/to/image_folder')
FOLDER_FPS = 6.0      # Example: original 30 FPS / extraction stride 5
```

For videos, FPS is read automatically. Run:

```bash
python detect.py
```

Results are saved in `detection_output/`:

- `detections.json`: bounding boxes, scores and prepared-frame information.
- `frames_<run>/`: prepared JPEG frames used by SAM 2.
- `boxes_<run>/`: bounding-box preview images and XYWH text files.

Inspect the box previews before running segmentation.

## 4. Segmentation and tracking — segment.py

At the top of `segment.py`, set:

```python
BASE = Path(__file__).resolve().parent
DETECTIONS = BASE / 'detection_output/detections.json'
SAM_REPO = BASE / 'sam2_official'
CHECKPOINT = BASE / 'checkpoints/sam2.1_hiera_small.pt'
MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
```

`MODEL_CONFIG` must be a relative string, not a filesystem `Path`.

Run:

```bash
python segment.py
```

This reads the saved detections and frames; it does not run OWLv2 again. It assigns tracking IDs, propagates masks, and confirms later-arriving animals across two detection scans before adding them.

Results are saved in `tracking_output/<run>/`:

| Output | Contents |
| --- | --- |
| `annotated.mp4` | Video with coloured masks, bounding boxes and animal IDs |
| `masks/animal_N/` | Binary mask images for each track |
| `tracks.csv` | Frame, source time, ID, box coordinates and mask area |
| `tracks.json` | Track initialization details and latest stored position |

Keep the prepared frames unchanged between detection and segmentation.

This is a prototype: IDs can change during occlusion or re-entry. Behaviour classification, such as sleeping, walking or head movement, is not implemented yet.

## Tracking Demo

[Watch the tracking demo](annotated.mp4)
