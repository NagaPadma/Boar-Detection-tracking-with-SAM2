"""Step 2: python segment.py. Reads saved boxes; does not load OWLv2.
Edit SAM_REPO and CHECKPOINT below to your real existing locations.
If SAM is not installed, use a clean official clone (commands below in comments):
  git clone https://github.com/facebookresearch/sam2.git sam2_repo
  git -C sam2_repo checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
  SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps -e ./sam2_repo
Then set SAM_REPO = BASE / 'sam2_repo'. Use a matching SAM 2.1 small checkpoint.

Later arrivals need two consecutive saved detections. Matching uses box overlap,
not appearance re-identification. Review masks/IDs before presenting results.
Keep the prepared frames unchanged after detection. IDs start at confirmation,
with no backward filling. Empty starts work; long absences can produce new IDs.
Local I/O/matching were tested with mock models, not real GPU inference.
"""
from pathlib import Path
from datetime import datetime
import csv
import json
import sys

# ---------- EDIT THESE SETTINGS ----------
BASE = Path(__file__).resolve().parent
DETECTIONS = BASE / 'detection_output/detections.json'
SAM_REPO = Path('/home/veerla/2026_Projects/Boar_Simple/sam2_off')  # official clean clone, not a fork
CHECKPOINT = Path('/home/veerla/2026_Projects/Boar_Simple/sam2_off/checkpoints/sam2.1_hiera_small.pt')
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
USE_FP16 = True  # V100/P100: use FP16 or float32, not BF16
NEW_OBJECT_CONFIRMATIONS = 2  # Set 1 for immediate arrivals, with more false positives
MIN_MASK_PIXELS = 25
MAX_TRACKS = 20
MATCH_IOU = 0.30
DUPLICATE_IOU = 0.20
PENDING_IOU = 0.20
RECENT_FRAMES = 10  # short-gap fallback; not long-term identity recovery


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def mask_box(mask):
    import numpy as np
    ys, xs = np.where(mask)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]


def match_boxes(frame, boxes, current_boxes, tracks, pending):
    """Match recent tracks; return new/recovery prompts and next pending boxes."""
    known = {i: t['box'] for i, t in tracks.items() if frame-t['last_seen'] <= RECENT_FRAMES}
    pairs = sorted([(iou(d['box'], box), j, i) for j, d in enumerate(boxes) for i, box in known.items()], reverse=True)
    used_boxes, used_ids, prompts = set(), set(), []
    for overlap, j, object_id in pairs:
        if overlap < MATCH_IOU:
            break
        if j in used_boxes or object_id in used_ids:
            continue
        used_boxes.add(j); used_ids.add(object_id)
        if current_boxes.get(object_id) is None:
            prompts.append((object_id, boxes[j]['box']))  # re-prompt short-gap recovery
    next_pending, used_pending = [], set()
    for j, detection in enumerate(boxes):
        box = detection['box']
        if j in used_boxes or any(iou(box, previous) >= DUPLICATE_IOU for previous in known.values()):
            continue
        options = [(iou(box, p['box']), k) for k, p in enumerate(pending) if k not in used_pending]
        overlap, k = max(options, default=(0, -1))
        hits = pending[k]['hits']+1 if overlap >= PENDING_IOU else 1
        if overlap >= PENDING_IOU:
            used_pending.add(k)
        needed = 1 if frame == 0 else NEW_OBJECT_CONFIRMATIONS
        if hits < needed:
            next_pending.append({'box': box, 'hits': hits})
            continue
        if len(tracks) >= MAX_TRACKS:
            raise RuntimeError('MAX_TRACKS reached. Inspect boxes for duplicates before increasing it.')
        object_id = max(tracks, default=0)+1
        tracks[object_id] = {'box': box, 'last_seen': frame, 'start_frame': frame,
                             'initial_box': box, 'initial_score': detection['score']}
        prompts.append((object_id, box))
        print(f'NEW animal track {object_id} at prepared frame {frame}', flush=True)
    return prompts, next_pending


def main():
    import cv2
    import numpy as np

    if not DETECTIONS.is_file():
        raise FileNotFoundError('Run detect.py first; detections.json is missing.')
    if NEW_OBJECT_CONFIRMATIONS < 1 or MAX_TRACKS < 1 or MIN_MASK_PIXELS < 1:
        raise ValueError('Confirmation count, MAX_TRACKS and MIN_MASK_PIXELS must be positive.')
    data = json.loads(DETECTIONS.read_text())
    frames_dir = Path(data['frames_dir'])
    entries, step = data['frames'], data['detect_every']
    height, width = data['height'], data['width']
    if not entries or step < 1 or data['fps'] <= 0:
        raise ValueError('Invalid detection result. Rerun detect.py.')
    if set(data['scans']) != {str(i) for i in range(0, len(entries), step)}:
        raise ValueError('Some saved detection scans are missing. Rerun detect.py.')
    expected_frames = {f'{i:07d}.jpg' for i in range(len(entries))}
    actual_frames = {p.name for p in frames_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'}}
    if actual_frames != expected_frames:
        raise ValueError('Prepared images are missing or extra frames were added. Rerun detect.py.')
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {CHECKPOINT}. Set its path at the top of segment.py.')
    package = SAM_REPO.resolve() / 'sam2'
    if not (package / '__init__.py').is_file():
        raise FileNotFoundError(f'SAM source missing: {package}. Set SAM_REPO at the top of segment.py.')
    if (package / 'sam2').is_dir():
        raise RuntimeError(f'Unexpected nested folder: {package / "sam2"}. Use a clean official clone; keep the upstream guard.')
    import torch
    sys.path.insert(0, str(SAM_REPO.resolve()))
    import sam2
    print(f'SAM source: {sam2.__file__}', flush=True)
    if not sam2.__file__ or Path(sam2.__file__).resolve() != (package / '__init__.py').resolve():
        raise RuntimeError('Wrong SAM package loaded. Start a fresh Python process and verify SAM_REPO.')
    from sam2.build_sam import build_sam2_video_predictor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Activate boar_demo and check nvidia-smi.')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    predictor = build_sam2_video_predictor(MODEL_CONFIG, str(CHECKPOINT), device='cuda:0',
        apply_postprocessing=False, hydra_overrides_extra=[
            '++model.non_overlap_masks=true', '++model.add_all_frames_to_correct_as_cond=true'])
    output = BASE / 'tracking_output' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    writer = cv2.VideoWriter(str(output / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), data['fps'], (width, height))
    if not writer.isOpened():
        raise RuntimeError('Could not create the output video.')
    state, tracks, pending = None, {}, []
    palette = [(0,255,0), (255,100,0), (0,150,255), (255,0,255), (255,255,0), (0,255,255)]

    def propagate(start, end):
        # SAM uses an inclusive offset: end-start, not the number of frames.
        for frame, ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=end-start):
            yield frame, [int(i) for i in ids], (logits[:, 0] > 0).cpu().numpy()

    def observe(frame, ids, masks):
        current = {}
        for object_id, mask in zip(ids, masks):
            box = mask_box(mask)
            current[object_id] = box
            if box is not None:
                tracks[object_id].update(box=box, last_seen=frame)
        return current

    print(f'Segmentation output: {output}', flush=True)
    try:
        with (output / 'tracks.csv').open('w', newline='') as handle, torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=USE_FP16):
            table = csv.writer(handle)
            table.writerow(['frame', 'source_time_s', 'track_id', 'status', 'x', 'y', 'width', 'height', 'mask_area_px'])
            for start in range(0, len(entries), step):
                # Finish this probe before changing IDs; do not mutate a running iterator.
                current = {}
                if tracks:
                    probe = list(propagate(start, start))
                    if len(probe) != 1 or probe[0][0] != start:
                        raise RuntimeError('SAM returned an unexpected boundary frame.')
                    current = observe(*probe[0])
                prompts, pending = match_boxes(start, data['scans'][str(start)], current, tracks, pending)
                if prompts and state is None:
                    state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
                for object_id, box in prompts:
                    predictor.add_new_points_or_box(inference_state=state, frame_idx=start, obj_id=object_id, box=np.array(box, np.float32))
                end = min(start+step-1, len(entries)-1)
                predictions = propagate(start, end) if tracks else ((i, [], []) for i in range(start, end+1))
                expected = start
                for frame, ids, masks in predictions:
                    if frame != expected or set(ids) != set(tracks):
                        raise RuntimeError('SAM returned unexpected frames or object IDs.')
                    current = observe(frame, ids, masks)
                    image = cv2.imread(str(frames_dir / entries[frame]['file']))
                    if image is None or image.shape[:2] != (height, width):
                        raise ValueError('Prepared image is unreadable or its dimensions changed.')
                    for object_id, mask in zip(ids, masks):
                        box = current[object_id]
                        area = int(mask.sum())
                        folder = output / f'masks/animal_{object_id}'
                        folder.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(folder / f'{frame:07d}.png'), mask.astype(np.uint8)*255):
                            raise OSError('Could not save mask.')
                        status = 'valid' if box else ('empty' if area == 0 else 'small')
                        coords = ['', '', '', '']
                        if box:
                            x1, y1, x2, y2 = box
                            coords = [x1, y1, x2-x1, y2-y1]
                            color = palette[(object_id-1) % len(palette)]
                            image[mask] = (image[mask]*0.55 + np.array(color)*0.45).astype(np.uint8)
                            cv2.rectangle(image, (x1,y1), (x2-1,y2-1), color, 2)
                            cv2.putText(image, f'Animal {object_id}', (x1,max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                        table.writerow([frame, entries[frame]['time_s'], object_id, status, *coords, area])
                    writer.write(image)
                    expected += 1
                if expected != end+1:
                    raise RuntimeError('SAM stopped before the end of this window.')
                print(f'Processed {end+1}/{len(entries)} frames | initialized tracks: {len(tracks)}', flush=True)
    finally:
        writer.release()
    (output / 'tracks.json').write_text(json.dumps(tracks, indent=2))
    print(f'DONE. Play: {output / "annotated.mp4"}', flush=True)
    print('Track IDs are temporary labels, not verified animal identities. No behaviour classification yet.')


if __name__ == '__main__':
    main()
