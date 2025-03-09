import sportslabkit as slk
import argparse

from sportslabkit.metrics import mota_score, hota_score, identity_score

from download import download_dataset, define_model
from detect import track_players


# save_path = "assets/tracking_results.mp4"
# res.visualize_frames(cam.video_path, save_path)

# The tracking data is now ready for analysis

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Arguments needed to determine which function to execute')
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolov8x', 'yolov9e', 'yolo11x'],
        default='yolov8x'
    )
    
    parser.add_argument(
        '--task', 
        type=str, 
        choices=['download', 'track', 'metrics', 'detect'], 
        help='Please provide information about which task to execute',
        default='detect'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        choices=['mota', 'hota', 'id'],
        help='Please provide information about the desired metric',
    )
    
    parser.add_argument(
        '--tracker_type',
        type=str,
        choices=['SORT', 'Deep', 'Byte'],
        help='Please provide tracker type',
        default='SORT'
    )
    
    args = parser.parse_args()
    
    # print available dataset paths`
    slk.datasets.get_path()
    
    
    dataset_path = slk.datasets.get_path('wide_view')
    path_to_csv = sorted(dataset_path.glob('annotations/*.csv'))[0]
    path_to_mp4 = sorted(dataset_path.glob('videos/*.mp4'))[0]
    
    cam = slk.Camera(path_to_mp4)
    
    bbdf = slk.load_df(path_to_csv)
    
    model = define_model(args.model)
    
    if args.task == 'download':
        download_dataset()
        
    elif args.task == 'detect':
        print('detect')
        
    elif args.task == 'track':
        track_players(args.model, path_to_csv, path_to_mp4, cam, args.tracker_type)
    
    elif args.task == 'metrics':
        
        
        gt_bbdf = bbdf[:5]
        pred_bbdf = bbdf[1:5]
        
        
        score = None
        
        if args.metric == 'mota':
            score = mota_score(pred_bbdf, gt_bbdf)
        elif args.metric == 'hota':
            score = hota_score(pred_bbdf, gt_bbdf)
        elif args.metric == 'id':
            score = identity_score(pred_bbdf, gt_bbdf)
                    
        print(score)
        
    else:
        print('No task defined in arguments')
        