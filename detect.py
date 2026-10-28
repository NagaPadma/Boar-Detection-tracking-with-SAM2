"""Step 1: python detect.py. Edit only the settings below.
Uses OWLv2 only; no SAM imports. Check the saved box previews before Step 2.
Input folder: numeric JPEG frames, e.g. 0000001.jpg. An MP4 also works.
Local I/O was tested with mock models; real GPU inference still needs testing.
"""
from pathlib import Path
from datetime import datetime
import json
import math

# ---------- EDIT THESE SETTINGS ----------
INPUT = Path('video2/Schnapsie Training.mp4')
OUTPUT = Path(__file__).resolve().parent / 'detection_output'
FOLDER_FPS = 6.0  # EXAMPLE: original 30 FPS / extraction stride 5. Verify this!
MAX_FRAMES = 300  # Short first test. None processes the entire input.
FRAME_STEP = 1
MAX_WIDTH = 1280
DETECT_EVERY = 1  # in saved/prepared frames
PROMPTS = ['picture of a animal', 'picture of a boar', 'picture of a wild pig', 'picture of a wild boar']
CONFIDENCE = 0.40
MODEL = 'google/owlv2-base-patch16-ensemble'


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def remove_duplicates(boxes):
    kept = []
    for item in sorted(boxes, key=lambda item: item['score'], reverse=True):
        if all(iou(item['box'], other['box']) < 0.5 for other in kept):
            kept.append(item)
    return sorted(kept, key=lambda item: item['box'][0])


def main():
    import cv2
    import numpy as np
    import scipy  # required by OWLv2 image resizing
    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    if not INPUT.exists():
        raise FileNotFoundError(f'Input not found: {INPUT}')
    if FRAME_STEP < 1 or DETECT_EVERY < 1 or (MAX_FRAMES is not None and MAX_FRAMES < 1):
        raise ValueError('FRAME_STEP, DETECT_EVERY and MAX_FRAMES must be positive.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate boar_demo and check nvidia-smi.')
    print(f'GPU: {torch.cuda.get_device_name(0)} | SciPy: {scipy.__version__}', flush=True)
    print('Loading OWLv2...', flush=True)
    processor = Owlv2Processor.from_pretrained(MODEL)
    # Test preprocessing before extracting frames or downloading model weights.
    processor(text=[PROMPTS], images=Image.new('RGB', (64, 48)), return_tensors='pt')
    model = Owlv2ForObjectDetection.from_pretrained(MODEL, use_safetensors=True).to('cuda').eval()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    frames_dir = OUTPUT / f'frames_{stamp}'
    preview_dir = OUTPUT / f'boxes_{stamp}'
    frames_dir.mkdir(parents=True)
    preview_dir.mkdir()
    cap = None
    if INPUT.is_dir():
        images = sorted([p for p in INPUT.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'} and p.stem.isdigit()], key=lambda p: int(p.stem))
        if not images:
            raise ValueError('No numeric-named JPEG images found in INPUT.')
        fps = FOLDER_FPS
        source = ((i, cv2.imread(str(path))) for i, path in enumerate(images))
    else:
        cap = cv2.VideoCapture(str(INPUT))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'Could not open video: {INPUT}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        def video_frames():
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                yield index, image
                index += 1
        source = video_frames()

    frames, scans, shape = [], {}, None
    try:
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Source FPS is invalid. Set FOLDER_FPS for image sequences.')
        for source_index, image in source:
            if source_index % FRAME_STEP:
                continue
            if image is None:
                raise ValueError(f'Unreadable source image at index {source_index}.')
            height, width = image.shape[:2]
            scale = min(1.0, MAX_WIDTH / width) if MAX_WIDTH else 1.0
            size = (int(width*scale)//2*2, int(height*scale)//2*2)
            if min(size) < 2:
                raise ValueError('Image is too small.')
            if size != (width, height):
                image = cv2.resize(image, size)
            if shape is None:
                shape = image.shape[:2]
            if image.shape[:2] != shape:
                raise ValueError('All prepared frames must have the same size.')
            index = len(frames)
            filename = f'{index:07d}.jpg'
            if not cv2.imwrite(str(frames_dir / filename), image):
                raise OSError('Could not save prepared frame.')
            frames.append({'file': filename, 'source_index': source_index, 'time_s': source_index/fps})
            if index % DETECT_EVERY == 0:
                # Detect on exactly the JPEG pixels SAM will read in Step 2.
                with Image.open(frames_dir / filename) as saved:
                    rgb = saved.convert('RGB')
                inputs = processor(text=[PROMPTS], images=rgb, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    outputs = model(**inputs)
                    result = processor.image_processor.post_process_object_detection(
                        outputs, threshold=CONFIDENCE, target_sizes=[shape])[0]
                boxes = []
                for box, score in zip(result['boxes'], result['scores']):
                    values = box.detach().cpu().numpy().astype(float)
                    values[[0, 2]] = np.clip(values[[0, 2]], 0, shape[1])
                    values[[1, 3]] = np.clip(values[[1, 3]], 0, shape[0])
                    if np.isfinite(values).all() and values[2]-values[0] >= 4 and values[3]-values[1] >= 4:
                        boxes.append({'box': values.tolist(), 'score': float(score.item())})
                boxes = remove_duplicates(boxes)
                scans[str(index)] = boxes
                preview = cv2.imread(str(frames_dir / filename))
                lines = []
                for item in boxes:
                    x1, y1, x2, y2 = item['box']
                    lines.append(','.join(f'{v:.2f}' for v in [x1, y1, x2-x1, y2-y1]))
                    cv2.rectangle(preview, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
                    cv2.putText(preview, f'{item["score"]:.2f}', (round(x1), max(20, round(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                if not cv2.imwrite(str(preview_dir / filename), preview):
                    raise OSError('Could not write detection preview.')
                (preview_dir / f'{index:07d}.txt').write_text('\n'.join(lines))
                print(f'Frame {index}: {len(boxes)} animal detections', flush=True)
            if MAX_FRAMES is not None and len(frames) >= MAX_FRAMES:
                break
    finally:
        if cap is not None:
            cap.release()
    if not frames:
        raise RuntimeError('No frames could be decoded.')
    # All paths become strings here; no metadata logger can mask a model error.
    payload = {'frames_dir': str(frames_dir.resolve()), 'frames': frames,
               'fps': fps/FRAME_STEP, 'height': shape[0], 'width': shape[1],
               'detect_every': DETECT_EVERY, 'scans': scans, 'model': MODEL,
               'prompts': PROMPTS, 'confidence': CONFIDENCE}
    temporary = OUTPUT / 'detections.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(OUTPUT / 'detections.json')
    print(f'DONE. Inspect boxes: {preview_dir}\nSaved: {OUTPUT / "detections.json"}', flush=True)


if __name__ == '__main__':
    main()
