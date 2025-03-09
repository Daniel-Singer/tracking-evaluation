import os
import optuna
import sportslabkit as slk
from sportslabkit.mot import SORTTracker, DeepSORTTracker, BYTETracker

def track_players(model_name=None, path_to_csv=None, path_to_mp4=None, cam=None, tracker_type=None):
        
    bbdf_gt = slk.load_df(path_to_csv)
    
    if bbdf_gt.index[0] == 0:
        bbdf_gt.index += 1
    bbdf_gt = bbdf_gt
    
    # setup tracker
    det_model = slk.detection_model.load(
        model_name=model_name,
        model=f'{model_name}.pt',
        conf=0.2,
        iou=0.3,
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

    
    tracker = None
    
    if tracker_type == 'SORT':

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
        
    elif tracker_type == 'Deep':
        
        slk.logger.set_log_level('INFO')
        det_model = slk.detection_model.load(
            model_name=model_name,
            model=f'{model_name}.pt',
            conf=0.25,
            iou=0.6,
            imgsz=960,
            device='mps',
            classes=0,
            augment=True,
            max_det=35
        )

        image_model = slk.image_model.load(
            model_name='mobilenetv2_x1_0',
            image_size=(32,32),
            device='mps'
        )

        motion_model = slk.motion_model.load(
            model_name='kalmanfilter',
            dt=1/30,
            process_noise=500,
            measurement_noise=10,
            confidence_scaler=1
        )

        matching_fn = slk.matching.MotionVisualMatchingFunction(
            motion_metric=slk.metrics.IoUCMM(use_pred_box=True),
            motion_metric_gate=0.2,
            visual_metric=slk.metrics.CosineCMM(),
            visual_metric_gate=0.2,
            beta=0.9,
        )

        tracker = DeepSORTTracker(
            detection_model=det_model,
            image_model=image_model,
            motion_model=motion_model,
            matching_fn=matching_fn,
            max_staleness=2,
            min_length=2
        )
    
    elif tracker_type == 'Byte':
        slk.logger.set_log_level('INFO')
        det_model = slk.detection_model.load(
            model_name=model_name,
            model=f'{model_name}.pt',
            conf=0.25,
            iou=0.6,
            imgsz=960,
            device='mps',
            classes=0,
            augment=True,
            max_det=35
        )

        image_model = slk.image_model.load(
            model_name='mobilenetv2_x1_0',
            image_size=(32,32),
            device='mps'
        )

        motion_model = slk.motion_model.load(
            model_name='kalmanfilter',
            dt=1/30,
            process_noise=500,
            measurement_noise=10,
            confidence_scaler=1
        )

        first_matching_fn = slk.matching.MotionVisualMatchingFunction(
            motion_metric=slk.metrics.IoUCMM(use_pred_box=True),
            motion_metric_gate=0.2,
            visual_metric=slk.metrics.CosineCMM(),
            visual_metric_gate=0.2,
            beta=0.9,
        )

        second_matching_fn = slk.matching.SimpleMatchingFunction(
            metric=slk.metrics.IoUCMM(use_pred_box=True),
            gate=0.9,
        )

        tracker = BYTETracker(
            detection_model=det_model,
            image_model=image_model,
            motion_model=motion_model,
            first_matching_fn=first_matching_fn,
            second_matching_fn=second_matching_fn,
            detection_score_threshold=0.6,
            max_staleness=2,
            min_length=2
        )
    else:
        print('Please provide tracker type')
        
    frames = cam
    
    bbdf_pred = tracker.track(frames)
    hota = slk.metrics.hota_score(bbdf_gt, bbdf_pred)
    
    mota = slk.metrics.mota_score(bbdf_gt, bbdf_pred)
    
    print(mota)
    

    