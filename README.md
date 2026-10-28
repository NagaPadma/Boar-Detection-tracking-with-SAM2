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

We use **OWLv2** (`google/owlv2-base-patch16-ensemble`) to detect wild boars using text prompts. The script saves bounding boxes, confidence scores and preview images.

Set `INPUT` in `detect.py` to a video or a folder of numeric-named JPEG images:

```python
INPUT = Path('/absolute/path/to/boar_video.mp4')
```

For an image folder, also set `FOLDER_FPS` to the original video FPS divided by the extraction stride. Video FPS is read automatically. The default test processes 50 frames; use `MAX_FRAMES = None` for the full input.

```bash
python detect.py
```

Results are saved in `detection_output/`: `detections.json`, prepared frames and box previews. Check the previews before running segmentation.
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
