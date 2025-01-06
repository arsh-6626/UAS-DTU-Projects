# MOTOR ALERTNESS INFERRENCES

## ULTRALYTICS YOLO

```
from ultralytics import YOLO
model = YOLO('bestbody.pt')
results = model.predict(source=image_path, save=False, conf=0.25)

```

### oh no error
`
[ WARN:0@7.474] global cap_ffmpeg_impl.hpp:1541 grabFrame packet read max attempts exceeded, if your video have multiple streams (video, audio) try to increase attempt limit by setting environment variable OPENCV_FFMPEG_READ_ATTEMPTS (current value is 4096)
`
* fix: 
` 
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'
`

## yolo human detection code

```
from ultralytics import YOLO
import os
import cv2
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

model = YOLO('/home/cha0s/ViTPose/demo/best_body.pt')
cap = cv2.VideoCapture('/home/cha0s/vid1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, save=False, conf=0.25)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## MMPose
* ![image](https://github.com/user-attachments/assets/2d49b4d3-cfee-47cf-bdc2-9a363b37fcda)
* TBC
 
## Variances of the code
* ```
    if prev_pose_results is not None and len(pose_results) > 0 and len(prev_pose_results) > 0:
                prev_keypoints = prev_pose_results[0]['keypoints']
                current_keypoints = pose_results[0]['keypoints']
                variances = np.var(current_keypoints - prev_keypoints, axis=0)
                print(f"Keypoint variances between frames: {variances}")
  ```
  ![image](https://github.com/user-attachments/assets/073b16a5-52dc-4254-9df7-24702abc0398)

  ```
  import os
import warnings
from argparse import ArgumentParser
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

def visualize():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path', default="/home/cha0s/D05_G1_S3.MP4")
    parser.add_argument('--burns-index-list', type=list, required=False)
    parser.add_argument('--lacerations-index-list', type=list, required=False)
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='sample',
        help='Root of the output video file.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')

    # Build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
       '../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', 
       'vitpose_small.pth', 
       device=args.device.lower()
    )

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Failed to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    person_model = YOLO('best_body.pt')

    prev_pose_results = None

    while cap.isOpened():
        flag, img = cap.read()
        img = cv2.resize(img, (1024, 1024))
        if not flag:
            break

        result_person = person_model.predict(img, verbose=False)
        for r in result_person:
            box = r.boxes.xyxy.to("cpu").numpy()
            cls = r.boxes.cls.to("cpu").numpy()
            if box.shape[0] == 0:
                continue

            if cls[0] != 0:  # Skip non-person detections
                continue

            widths = box[:, 2] - box[:, 0]
            heights = box[:, 3] - box[:, 1]
            areas = widths * heights
            max_area_index = np.argmax(areas)
            max_area_box = box[max_area_index]        
            x1_body, y1_body, x2_body, y2_body = map(int, max_area_box)

            person_results = [{'bbox': np.array([x1_body, y1_body, x2_body, y2_body])}]

            # Perform pose estimation on the detected person
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None
            )

            # Calculate variances between two frames
            if prev_pose_results is not None and len(pose_results) > 0 and len(prev_pose_results) > 0:
                prev_keypoints = prev_pose_results[0]['keypoints']
                current_keypoints = pose_results[0]['keypoints']
                variances = current_keypoints - prev_keypoints
                for i, variance in enumerate(variances):
                    print(f"Variance for keypoint {i}: {variance}")

            # Label each keypoint on the image
            if len(pose_results) > 0:
                for i, (x, y, score) in enumerate(pose_results[0]['keypoints']):
                    if score > args.kpt_thr:
                        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            prev_pose_results = pose_results

            # Visualize the pose estimation results
            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False
            )

            # Save the frame to the output video
            if save_out_video:
                videoWriter.write(vis_img)

            # Optionally display the frame
            if args.show:
                cv2.imshow('Pose Estimation', vis_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize()
    
```

* meow
