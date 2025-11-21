import argparse
import glob
from pathlib import Path
import os
import cv2
import numpy as np

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from visual_utils import open3d_vis_utils as V

# --- nuScenes Devkit 로드 (카메라 연동용) ---
# ‼️ 사용자님의 v1.0-mini 경로로 설정되어 있는지 확인하세요
NUSCENES_DATAROOT = '/home/omen16/workspace/OpenPCDet/data/nuscenes/v1.0-mini'
NUSCENES_VERSION = 'v1.0-mini'
print(f"Loading nuScenes devkit for camera sync (this may take a moment)...")
try:
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    print("nuScenes devkit loaded.")
except Exception as e:
    print(f"Error: nuScenes devkit 로드 실패. {e}")
    nusc = None
# ---


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True, help='config file for the model')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint file to load')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample in the validation set to visualize')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def show_6_cameras(sample_idx, test_set, logger):
    """지정된 인덱스의 6개 카메라 이미지를 띄웁니다."""
    if nusc is None:
        return
        
    try:
        # 1. 'test_set'의 info에서 LiDAR 파일 상대 경로 가져오기
        #    test_set.infos[idx]에는 'lidar_path'가 있음 (예: 'samples/LIDAR_TOP/...')
        lidar_relative_path = test_set.infos[sample_idx]['lidar_path']
        
        # 2. devkit으로 LiDAR 토큰 찾기
        lidar_sd_token = nusc.field2token('sample_data', 'filename', lidar_relative_path)[0]
        lidar_sample_data = nusc.get('sample_data', lidar_sd_token)
        
        # 3. 'sample' (키프레임) 토큰 찾기
        sample_token = lidar_sample_data['sample_token']
        sample_record = nusc.get('sample', sample_token)
        
        # 4. 6개 카메라 채널 리스트 정의
        cam_channels = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        # 5. 각 카메라 채널을 순회하며 이미지 띄우기
        for channel in cam_channels:
            cam_sd_token = sample_record['data'].get(channel)
            if not cam_sd_token:
                logger.warning(f"Warning: {channel} 데이터를 찾을 수 없습니다.")
                continue
            
            cam_sample_data = nusc.get('sample_data', cam_sd_token)
            cam_filename = cam_sample_data['filename']
            cam_image_path = os.path.join(NUSCENES_DATAROOT, cam_filename)
            
            img = cv2.imread(cam_image_path)
            if img is None:
                logger.warning(f"Warning: {cam_image_path} 이미지를 로드할 수 없습니다.")
                continue
            
            scale = 0.3 # 30%
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            img_resized = cv2.resize(img, (width, height))
            
            cv2.imshow(f'{channel}', img_resized)

        cv2.waitKey(100) # 100ms 대기
    
    except Exception as e:
        print(f"Error: 카메라 이미지를 로드/표시할 수 없습니다: {e}")


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------10-Sweep Demo of OpenPCDet-------------------------')

    # 1. NuScenesDataset 객체 생성 (test.py와 동일한 방식)
    #    이렇게 하면 10-sweep 로직이 자동으로 적용됩니다.
    logger.info(f"Loading dataset: {cfg.DATA_CONFIG.DATASET}")
    
    # 데모용으로 Augmentation과 Shuffle 비활성화
    cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ['ALL']
    if 'SHUFFLE_ENABLED' in cfg.DATA_CONFIG.DATA_PROCESSOR[1]:
         cfg.DATA_CONFIG.DATA_PROCESSOR[1].SHUFFLE_ENABLED['test'] = False
            
    test_set = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        logger=logger
    )
    
    logger.info(f'Total number of samples in val set: {len(test_set)}')
    if args.sample_idx >= len(test_set):
        logger.error(f"Error: --sample_idx {args.sample_idx} is out of range (max is {len(test_set) - 1})")
        return

    # 2. 모델 빌드 및 체크포인트 로드 (test.py와 동일)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    # 3. 사용자가 요청한 단일 샘플 가져오기
    #    (이때 test_set[idx]가 10-sweep 로직을 실행합니다!)
    logger.info(f"Loading sample index: {args.sample_idx}")
    data_dict = test_set[args.sample_idx]
    
    # 4. 카메라 이미지 띄우기 (수정된 demo.py 로직)
    show_6_cameras(args.sample_idx, test_set, logger)

    # 5. 추론 실행 (demo.py와 동일)
    logger.info("Running inference...")
    data_dict = test_set.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    SCORE_THRESH = 0.5  # (이 값을 0.2 ~ 0.5 사이로 조절해보세요)
    pred_scores = pred_dicts[0]['pred_scores']
    mask = pred_scores >= SCORE_THRESH

    # 6. 10-sweep 포인트 클라우드와 예측 결과 시각화
    logger.info("Displaying Open3D visualization...")

    print("===== 모델 예측 결과 (Raw Labels) =====")
    print(pred_dicts[0]['pred_labels'].cpu().numpy())
    print("========================================")

    V.draw_scenes(
        points=data_dict['points'][:, 1:],  # 0번째 'x'좌표 외의 피처
        ref_boxes=pred_dicts[0]['pred_boxes'],
        ref_scores=pred_scores[mask], 
        ref_labels=pred_dicts[0]['pred_labels']
    )

    logger.info('Demo done. Close Open3D window to exit.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
