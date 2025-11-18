import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image


class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION # 데이터셋의 진짜 루트 경로를 설정
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        ) # 부모 클래스 초기화  -> 부모 클래스 것을 쓸 수 있다!
        self.infos = [] # 이 데이터셋이 사용할 모든 sample의 정보(.pkl) 파일의 내용을 담을 빈 리스트를 생성

        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False  # 카메라 부분

        self.include_nuscenes_data(self.mode) # mode : train or test에 따라 .pkl 파일을 디스크에서 열어 내용물을 self.infos 리스트에 채워넣음.

        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos) # 해당 함수를 호출 -> bicycle처럼 희귀한 클래스가 포함된 info를 중복 복사하여 그 sample을 부풀린다 !!
            # train 모드일 때만 실행 

    def include_nuscenes_data(self, mode): # .pkl 파일 로더 
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]: # yaml 파일에서 mode를 키로다가 파일 경로가 담긴 리스트를 가져옴.
            info_path = self.root_path / info_path # nuscenes_infos_10sweeps_train.pkl이 할당
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f: # binary 읽기 모드로 열고 파일이 생성될 때의 원본 파이썬 객체로 완벽하게 복원
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos) # 복원한 sample 정보를 self.infos 리스트에 채워넣음.
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos): # 훈련 데이터 재조정기 -> train.py 시에만 호출되며 클래스 불균형 문제를 해결
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492 
        """ 
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names} # 10개 클래스 이름을 key로 하고 빈 리스트를 값으로 가지는 버킷 딕셔너리를 만든다. 
        for info in infos:
            for name in set(info['gt_names']): # 해당 sample에 등장하는 모든 정답 클래스를 가져온다.
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()} # 원본 데이터셋의 클래스 분포를 계산

        sampled_infos = []

        frac = 1.0 / len(self.class_names) # 모든 클래스가 10%씩 등장하는 "이상적인(target) 분포" 값
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios): # 리샘플링 실행 
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos
    # 323개의 sample info -> 1630개의 증강된 info

    # 핵심 역할 : 과거의 한 시점(Sweep)"에 찍힌 라이다 데이터를 불러와서, "현재 시점의 차량 위치 기준"으로 좌표를 변환(이동)
    def get_sweep(self, sweep_info): # sweep_info에는 nuscenes_utils.py가 미리 계산해서 .pkl에 저장해 둔 딕셔너리 -> 
        # 그 과거 프레임의 파일 경로(lidar_path), 현재 프레임과의 시간 차이(time_lag), 그리고 가장 중요한 좌표 변환 행렬(transform_matrix) 있음 !
       
        def remove_ego_points(points, center_radius=1.0): # 내부 헬퍼 함수, 라이다 센서 바로 주변 (반경 1m 정사각형 안)에 찍힌 포인트들을 제거
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        # ======================= 수정 포인트 !!!! =============================
        # 과거 데이터를 읽고 ring_index를 버림.
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        # reshape([-1, 5]): 원본 데이터는 5개 컬럼 (x, y, z, intensity, ring_index)을 가지고 있습니다.
        # [:, :4]: 이 부분에서 5번째 컬럼인 ring_index가 삭제되고, 앞의 4개 (x, y, z, intensity)만 남습니다.

        points_sweep = remove_ego_points(points_sweep).T # 차체 노이즈를 제거하고 전치 행렬을 만듦. (4, N)

        if sweep_info['transform_matrix'] is not None: # [핵심] 좌표 변환 (과거 ➡️ 현재)
            num_points = points_sweep.shape[1] 
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :] # 동차 좌표를 만들어 4x4 변환 행렬을 곱함 -> 과거의 포인트들이 현재 차량이 있는 위치 기준으로 이동 및 회전하게 된다.

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1])) # 시간 정보 생성 -> 과거 프레임이 현재로부터 얼마나 전인지 모든 포인트에 똑같이 부여
        return points_sweep.T, cur_times.T # 원래 모양의로 뒤집어서 변환된 포인트 좌표와 시간 정보 반환

    # 핵심 역할은 "현재 시점의 메인 라이다 데이터"와 "과거 시점의 보조 라이다 데이터(Sweeps)"를 불러와서 하나의 거대하고 빽빽한 포인트 클라우드로 합체시키는 것
    def get_lidar_with_sweeps(self, index, max_sweeps=1):  # max_sweeps=1: 총 몇 개의 프레임을 합칠지 결정
        info = self.infos[index] # 해당 인덱스의 메타데이터를 가져옴. (라이다 파일 경로, 과거 프레임 정보 리스트, 정답 박스 등)
        lidar_path = self.root_path / info['lidar_path'] # 현재 프레임의 실제 파일 경로 완성
        # ===================== 중요 ===========================
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        # 역할 : "현재" 프레임의 바이너리 데이터를 읽고, 5번째 정보를 버립니다. (🚨 매우 중요)
        # .reshape([-1, 5]): 일렬로 나열된 데이터를 포인트당 5개 값 (x, y, z, intensity, ring_index)을 가지는 형태로 재배열
        # [:, :4]: [졸업 프로젝트 수정 포인트 1]
            # 모든 포인트(:)에 대해, 앞의 4개 컬럼(:4 ➡️ 0, 1, 2, 3번 인덱스)만 남깁니다.
            # x, y, z, intensity만 남기고 ring_index는 이 순간 삭제

        sweep_points_list = [points] # 합체할 데이터를 담을 **바구니(리스트)**를 준비
        sweep_times_list = [np.zeros((points.shape[0], 1))] # "현재" 포인트들의 시간 차이(time_lag)는 0초이므로, 포인트 개수만큼의 0.0을 담은 배열을 만들어 담습니다.

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False): # 과거 프레임(Sweeps)을 무작위로 선택하여 순회
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k]) # 선택된 과거 프레임 하나를 불러오고 변환 -> 이 함수가 과거 데이터를 읽고, ring_index를 버리고, 현재 좌표계로 변환한 뒤 반환
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep) # 불러온 과거 포인트와 시간 정보를 바구니에 추가

        points = np.concatenate(sweep_points_list, axis=0) # points: 10개 프레임의 모든 포인트 (x, y, z, intensity)가 세로로 길게 합쳐집니다. (N, 4)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype) # times: 10개 프레임의 모든 시간 정보가 세로로 길게 합쳐집니다. (N, 1)

        points = np.concatenate((points, times), axis=1) # 포인트 정보와 시간 정보를 가로로 합칩니다. -> [x, y, z, intensity, timestamp] 데이터가 완성
        
        return points # 완성된 10프레임 겹친 데이터를 반환

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index): # 0 ~ 1629개의 index를 받음.
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos) # 에포크 기반이 아니라 iter 기반 train을 하는 경우, 인덱스가 전체 데이터 개수를 넘어가면 다시 0번부터 시작하도록 조정

        info = copy.deepcopy(self.infos[index]) # 메모리에서 로드해 둔 self.infos를 deep copy -> 이후 데이터 증강 과정에서 값이 막 바뀌므로 원본을 보호하기 위함.

        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        # 디스크에 있는 라이다 원본 파일 .bin 파일들을 실제로 읽어서, 10프레임이 하나로 합쳐진 거대한 포인트 클라우드 (N, 5) 크기의 넘파이 배열 [x, y, z, intensity, timestamp])를 반환
        # VLP-16 시뮬레이션 데이터를 위해 제거해야하는 ring_index 삭제 코드가 get_lidar_with_sweeps에 있음 !!

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        } # 모델 파이프라인에 전달할 표준 데이터 패키지 구성 / points: 방금 로드한 10프레임 포인트 클라우드. / frame_id: 파일 이름 (예: 'n008-2018...') / metadata: 나중에 평가 결과를 제출할 때 필요한 nuScenes 고유 토큰.

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1) # GT 박스 안에 라이다 포인트가 0개보다 많은 박스만 True가 되는 마스크를 만든다. 
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            }) # mask 연산을 통해 유효한 정답을 가진 GT box 정보만 info_dict 패키지에 추가

        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info) # 카메라 부분
 
        data_dict = self.prepare_data(data_dict=input_dict) # NuScenesDataset의 부모인 DatasetTemplate에 정의된 함수
        # 이 함수가 .yaml의 DATA_AUGMENTOR (데이터 증강 부분) + DATA_PROCESSOR (전처리: 범위 필터링, Voxel/Pillar 변환)를 순서대로 모두 실행
        # 이걸 거치면 원본 데이터(input_dict)이 train에 투입되는 형태, data_dict으로 완전히 탈바꿈 !

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes # 혹시나 데이터 증강 과정에서 속도 정보 값이 NaN이 되면 0으로 바꿔주는 부분

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]] # 속도 예측 사용 안하면 속도 정보 인덱스를 삭제하는 부분

        return data_dict # 이게 train.py에서 collate_batch 함수를 거쳐 GPU로 올라가고 forward path 진행 !!

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10): # used_classes가 기본 None으로 지정되어있어서 5개로 수정할 때 손 봐야함.
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo' # 오려낸 객체 .bin 파일을 저장할 폴더 경로
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl' # 오려낸 객체들의 목록 역할을 할 .pkl 파일 경로

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))): # 모든 train sample을 tqdm과 함께 순회
            sample_idx = idx
            info = self.infos[idx] # 현재 sample의 정보 딕셔너리를 가져옴.
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps) # 현재 sample로부터 10프레임이 겹쳐진 dense한 포인트 클라우드 ([x, y, z, i, timestamp])를 로드
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names'] # 현재 sample의 정답 박스와 정답 이름을 가져옴.

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu( # 고도로 최적화된 GPU 함수
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(), # 씬의 "모든" 포인트 (points[:, 0:3], 즉 x,y,z)
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda() # 씬의 "모든" 정답 박스 (gt_boxes[:, 0:7], 즉 x,y,z,l,w,h,yaw)
            ).long().squeeze(dim=0).cpu().numpy()
            # 출력 배열은 "씬의 1번 포인트는 2번 박스 안에 있다", "2번 포인트는 아무 박스에도 속하지 않는다(-1)", "3번 포인트는 0번 박스 안에 있다" ... 처럼 모든 포인트가 어떤 박스에 속하는지 알려주는 "소속 지도"

            for i in range(gt_boxes.shape[0]): # 현재 모든 sample 안의 모든 정답 박스를 순회
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i) # 이 객체를 저장할 고유 파일 이름
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i] #box_idxs_of_pts를 이용해 sample의 모든 points 중에서 오직 i번째 박스에 속하는 포인트만 gt_points로 추출(필터링)

                gt_points[:, :3] -= gt_boxes[i, :3] # 오려낸 포인트들의 (x, y, z) 좌표에서 **해당 박스의 중심 좌표(x, y, z)**를 뺍니다. -> 오려낸 객체가 (0,0,0)으로 이동!
                    # gt_sampling을 할 때 원점에 있는 객체를 가져와야 붙여넣기가 쉬움!
                with open(filepath, 'w') as f:
                    gt_points.tofile(f) # 개별 .bin으로 저장

                if (used_classes is None) or gt_names[i] in used_classes: # 이 객체가 .yaml에서 받아온 used_classes에 포함되어 있는지 확인
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]} # 오려낸 객체에 대한 카탈로그 정보 딕셔너리를 만듬. (이름, .bin 파일 경로, 원본 박스 정보, 포인트 개수 등)
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info) # 이 카탈로그 정보를 all_db_infos 마스터 딕셔너리에 추가
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v))) # tqdm 루프가 (323개 샘플) 모두 끝나면, all_db_infos 딕셔너리를 순회하며 최종 통계를 터미널에 print

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f) # 완성된 "카탈로그" (all_db_infos 딕셔너리)를 **nuscenes_dbinfos...pkl 파일로 저장


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
    # 1. import 및 초기 설정
    from nuscenes.nuscenes import NuScenes # nuScenes 원본 .json 메타데이터를 읽기 위한 nuScenes 공식 devkit 라이브러리를 임포트
    from nuscenes.utils import splits # nuScenes가 공식적으로 "어떤 씬이 훈련용이고 어떤 씬이 검증용인지" 나눠놓은 씬 이름 리스트
    from . import nuscenes_utils # 핵심 로직이 들어있는 nuscenes_utils.py import
    data_path = data_path / version # 원본 데이터 경로 완성
    save_path = save_path / version

    # 2. train/val 용 Scene 목록 정의
    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'] # 정해진 버전만 사용
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train # 이 변수에는 미니 훈련용 씬 8개 저장
        val_scenes = splits.mini_val # 미니 검증용 씬 2개 저장
    else:
        raise NotImplementedError

    # 3. 사용 가능한 Scene 필터링 (궁금한 점 : Scene 정보는 어떻게 보는지, 각 씬은 몇 프레임으로 이루어져 있는지?)
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True) # nuScenes devkit 객체를 초기화 -> v1.0-mini의 모든 메타데이터(.json)를 읽어들여, 모든 샘플, 센서, 정답 정보에 접근
    available_scenes = nuscenes_utils.get_available_scenes(nusc)  # nusc가 알고 있는 씬 목록 중, 실제로 .bin 파일이 디스크에 존재하는 씬만 찾아달라고 요청 -> 데이터를 일부만 다운받았을 경우를 대비.. (이러면 part만 받아도 되는 거 아닌가?)
    available_scene_names = [s['name'] for s in available_scenes] 
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes)) # splits에서 가져온 "공식 훈련 씬 목록"과, "디스크에 실제 파일이 있는 씬 목록"을 비교하여 교집합만 남김.
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes]) # 처리하기 편하도록 씬의 이름('scene-0016')을 씬의 고유 ID('token')로 변환하여 최종 train_scenes 세트(set)를 완성
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    # 4. 작업 위임 및 저장 
    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes))) # train 씬과 val 씬 개수 출력

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
    ) # 이 nusc 객체와, 8개의 train_scenes, 2개의 val_scenes를 줄 테니, max_sweeps=10 설정을 따라서 모든 정보가 담긴 훈련/검증용 리스트(infos)를 만들어줘"라고 요청
    # nuscenes_utils.py의 fill_trainval_infos 함수가 nuScenes의 모든 샘플(404개)을 tqdm (프로그레스 바)으로 순회하며, 10프레임 변환 행렬, 정답 박스, 클래스 이름 매핑 등을 수행
    # train_nusc_infos (323개)와 val_nusc_infos (81개) 리스트를 반환

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f) # 최종 결과물(정보 리스트)을 pickle을 사용해 .pkl 파일로 저장(직렬화)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)
    # 결과적으로 nuscenes_infos_10sweeps_train.pkl과 nuscenes_infos_10sweeps_val.pkl 파일 2개가 디스크에 생성 완료!!!

# python -m pcdet.datasets.nuscenes.nuscenes_dataset ... 명령어를 처음 실행했을 때 파이썬이 가장 먼저 실행하는 코드
if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    # 1. 스크립트 실행 준비 (Argument Parsing)
    parser = argparse.ArgumentParser(description='arg parser') # 터미널 명령어를 해석할 argparse 객체를 생성
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset') 
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args() # 터미널에 입력한 argment 불러오는 부분 : --cfg_file: 사용할 .yaml 설정 파일 경로, --func: 실행할 함수의 이름 (create_nuscenes_infos), --version: 사용할 데이터셋 버전. 
    
    # 2. .pkl 파일 생성 실행 (Offline Pre-processing)
    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))  # nuscenes_dataset.yaml의 딕셔너리 값들을 객체처럼 바꿔서 .(점)으로 접근할 수 있게 해줌!
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve() # 프로젝트 ROOT의 절대 경로 저장
        dataset_cfg.VERSION = args.version # yaml에서 읽어온 설정의 VERSION 값을 터미널에서 입력 받은 값으로 덮어 씌운다. 
        
        # 핵심 작업 1. ...infos.pkl 데이터베이스 파일 생성 / create_nuscenes_info 함수를 호출 -> 자세한 건 위에서, 결과로 pkl 뽑아냄.
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            with_cam=args.with_cam
        ) 
        # 핵심 작업 2(준비) NuScenesDataset 클래스의 객체를 생성
        # class_names=None 부분이 5개 클래스 필터링에 실패한 이유 !! ==> yaml의 CLASS_NAMES을 읽어오지 않고 None을 전달해서.. --
        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, 
            # --------------------------------- Class 5개 줄이는 거 고려할 부분 -------------------------------------
            class_names=None,
            # --------------------------------- Class 5개 줄이는 거 고려할 부분 -------------------------------------
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True
        )
        # 핵심 작업 2. ...dbinfos.pkl 데이터베이스 파일 생성 / create_groundtruth_database 함수를 호출 -> Data Augmentation 기법에 사용되는 객체 GT Database pkl
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
