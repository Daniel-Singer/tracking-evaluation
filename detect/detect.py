import os
import sportslabkit as slk
from sportslabkit.mot import SORTTracker

def track_players(model=None, path_to_csv=None, path_to_mp4=None, cam=None):
    
    bbdf_gt = slk.load_df(path_to_csv)
    
    if bbdf_gt.index[0] == 0:
        bbdf_gt.index += 1
    bbdf_gt = bbdf_gt[:100]
    
    # setup tracker
    det_model = slk.detection_model.load(
        model_name='yolov8',
        model='yolov8x.pt',
        conf=0.25,
        iou=0.6,
        imgsz=960,
        device='mps',
        classes=0,
        augment=True,
        max_det=35
    )
    
    motion_model = slk.motion_model.load(
    model_name='kalmanfilter',
    dt=1/30,
    process_noise=500,
    measurement_noise=10,
    confidence_scaler=1
    )

    matching_fn = slk.matching.SimpleMatchingFunction(
        metric=slk.metrics.IoUCMM(use_pred_box=True),
        gate=0.9
    )

    tracker = SORTTracker(
        detection_model=det_model,
        motion_model=motion_model,
        matching_fn=matching_fn,
        max_staleness=2,
        min_length=2
    )
    
    frames = cam[:100]

    bbdf_pred = tracker.track(frames)
    hota = slk.metrics.hota_score(bbdf_gt, bbdf_pred)["HOTA"]
    print(f"HOTA Score before Tuning (SORTTracker): {hota:.3f}")