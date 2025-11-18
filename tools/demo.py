import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import cv2  
from nuscenes.nuscenes import NuScenes
import os

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

NUSCENES_DATAROOT = '/home/omen16/workspace/OpenPCDet/data/nuscenes/v1.0-mini'
NUSCENES_VERSION = 'v1.0-mini'
print(f"Loading nuScenes devkit for camera sync (this may take a moment)...")
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
print("nuScenes devkit loaded.")

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.logger is not None:
            self.logger.info(f'Visualized sample index: \t{index + 1}')

        # 1. 5개 컬럼(x, y, z, intensity, ring_index)을 모두 로드
        points_full = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)

        # 2. 앞의 4개(x, y, z, intensity)만 사용
        points_xyzi = points_full[:, :4]

        # 3. 5번째 컬럼인 timestamp를 0.0으로 직접 생성
        #    (데모는 1개 프레임만 보므로 시간=0.0)
        timestamps = np.zeros((points_xyzi.shape[0], 1), dtype=np.float32)

        # 4. [x, y, z, intensity, timestamp] 5개로 합치기
        points = np.concatenate((points_xyzi, timestamps), axis=1)

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                pred_dicts, _ = model.forward(data_dict)
            
            try:
                # 1. LiDAR 파일 경로 가져오기 (절대 경로로)
                lidar_full_path = os.path.abspath(demo_dataset.sample_file_list[idx])
                lidar_relative_path = str(Path(lidar_full_path).relative_to(NUSCENES_DATAROOT)).replace(os.sep, '/')
                
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
                    
                    # 6. OpenCV로 이미지 읽기
                    img = cv2.imread(cam_image_path)
                    if img is None:
                        logger.warning(f"Warning: {cam_image_path} 이미지를 로드할 수 없습니다.")
                        continue
                    
                    # (선택) 이미지가 너무 크므로 작게 리사이즈
                    scale = 0.3 # 30% (6개를 띄워야 하니 더 작게)
                    width = int(img.shape[1] * scale)
                    height = int(img.shape[0] * scale)
                    img_resized = cv2.resize(img, (width, height))
                    
                    # 7. 개별 창에 띄우기
                    cv2.imshow(f'{channel}', img_resized)

                # 8. 모든 창이 뜰 수 있도록 잠시 대기
                cv2.waitKey(100) # 100ms 대기
            
            except Exception as e:
                print(f"Error: 카메라 이미지를 로드/표시할 수 없습니다: {e}")

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
