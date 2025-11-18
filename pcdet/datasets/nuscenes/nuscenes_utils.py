"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
} # class 바꾸려면.. 이것도 건드려야할 것 같은 느낌이 싹 도는데?


cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
} # result_nusc.json 파일에 attribute를 끼워넣기 위해 미리 정의해놓은 attribute 데이터 시트!


def get_available_scenes(nusc): # 메타데이터 DB에는 있지만, 디스크에는 없는 유령 Scene을 걸러내는 역할
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene: # nusc DB에 있는 모든 씬을 하나씩 반복
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token) # 그 씬의 첫 번째 sample을 가져옴.
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])  # 그 씬의 첫 번째 sample을 가져옴.
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP']) # 그 "첫 번째 키프레임"에 연결된 LIDAR_TOP 센서 데이터의 정보
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token']) # "첫 번째 키프레임"의 LIDAR_TOP 데이터가 저장된 실제 하드디스크 경로를 가져옴.
            if not Path(lidar_path).exists(): # 실제로 디스크에 존재하는지 그 경로에 가서 확인
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue # 만약 파일이 없었다면, available_scenes 리스트에 그 씬을 추가하지 않고 건너뛴다. 
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes # 실제로 존재하는 씬만 리턴


def get_sample_data(nusc, sample_data_token, selected_anntokens=None): # 좌표계 변환기 역할
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token) # LIDAR_TOP의 고유 ID를 사용해 해당 센서 데이터의 상세 정보를 DB에서 조회
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token']) # 차량 대비 어디에 부착되었는지에 대한 정보
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token']) # 차량이 맵 기준으로 어디에 위치해있는지에 대한 변환 정보

    data_path = nusc.get_sample_data_path(sample_data_token) # 실제 파일 경로

    if sensor_record['modality'] == 'camera': # 카메라 내부 파라미터 준비하는 부분
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # 모든 정답 박스를 nusc DB에서 가져옴. 
    if selected_anntokens is not None: 
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes: # 글로벌 좌표계의 정답 박스를 하나씩 모두 순회
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system (global -> 차량)
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system (차량 -> 센서)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic # 라이다 .bin 경로와 라이다 센서 좌표계 기준으로 변환된 GT box 리스트


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw # 단위 x 벡터를 쿼터니언 회전시킨 다음, 그 각도를 x,y 평면에 대해 계산해서 yaw 값을 계산!
    
# 센서 좌표계 통일하는 부분 -> 카메라 퓨전용이라 안 봐도 됨!
def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep


def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10, with_cam=False): # nuscenes_dataset.py에서 nusc 객체, 씬 목록(train_scenes, val_scenes), max_sweeps=10 설정을 전달받음.
    train_nusc_infos = []
    val_nusc_infos = [] # 최종 결과물인 정보 딕셔너리를 담을 빈 리스트 2개를 생성
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True) # progress bar 생성

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.
    # LIDAR_TOP 센서를 기준으로 삼아 모든 좌표 변환과 프레임 합치기를 수행하겠다! (아마 멀티모달 퓨전 관련해서 사용하는것 것이라 추측)

    for index, sample in enumerate(nusc.sample): # nusc 객체에 들어있는 모든 키프레임을 하나씩 순회 (mini 기준 404개)
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan] # 현재 sample 딕셔너리에서 LIDAR_TOP 센서의 데이터 ID(ref_sd_token)를 가져옵니다.
        ref_sd_rec = nusc.get('sample_data', ref_sd_token) # 해당 ID의 상세 정보(ref_sd_rec, 예: 파일명, 타임스탬프)를 가져옵니다.
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token']) # 현재" LIDAR_TOP 센서가 차량(Ego) 대비 어디에 부착되었는지 (x, y, z, rotation)에 대한 보정(Calibration) 정보를 가져옵니다.
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token']) # "현재" **차량(Ego)**이 글로벌 맵(Global) 대비 어디에 있는지 (x, y, z, rotation)에 대한 위치(Pose) 정보
        ref_time = 1e-6 * ref_sd_rec['timestamp'] # 현재 키프레임의 "기준 시간"을 초(second) 단위로 저장

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token) 
        # ref_sd_token을 사용해 "현재" 라이다 .bin 파일의 경로(ref_lidar_path)와, "현재" 씬의 모든 정답 박스(ref_boxes) 목록을 LIDAR_TOP 좌표계로 변환하여 가져옵니다.

        ref_cam_front_token = sample['data']['CAM_FRONT'] # 퓨전 모델을 위해 전방 카메라 정보도 가지고 온다. 
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
        ) # (Ego ➡️ Sensor) : 이 행렬은 "현재" 차량(Ego) 좌표계의 점을 "현재" LIDAR_TOP 좌표계로 변환하는 4x4 행렬

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
        )  # (Global ➡️ Ego) : 이 행렬은 "글로벌 맵" 좌표계의 점을 "현재" 차량(Ego) 좌표계로 변환하는 4x4 행렬

        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
            'cam_intrinsic': ref_cam_intrinsic,
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        } # sample의 모든 정보를 저장할 메인 딕셔너리를 생성

        if with_cam:
            info['cams'] = dict()
            l2e_r = ref_cs_rec["rotation"]
            l2e_t = ref_cs_rec["translation"],
            e2g_r = ref_pose_rec["rotation"]
            e2g_t = ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info["cams"].update({cam: cam_info})
        # 카메라 퓨전 모델용 부분이라 넘어가기

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token) # 현재 LIDAR_TOP 데이터로 초기화
        sweeps = []
        while len(sweeps) < max_sweeps - 1: # 9개의 과거 프레임을 찾을 때까지 반복
            if curr_sd_rec['prev'] == '': # 만약 이게 씬의 첫 프레임이라 이전 프레임이 없으면, 이전 프레임을 복제하여 가짜 10프레임을 만든다. 
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1]) # 아니면 첫 프레임을 복사
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev']) # 현재 프레임에서 한 프레임 과거로 이동

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token']) # 과거 시점의 차량의 위치 정보를 가지고 옴
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                ) # 과거 차량 좌표계 -> 글로벌 맵 좌표계로 변환하는 행렬

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                ) # 과거 시점의 라이다 센서 보정 정보를 가지고 온다. 
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                ) # (Sensor ➡️ Ego): "과거" 라이다 좌표계 ➡️ "과거" 차량 좌표계로 변환하는 행렬을 계산

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                # "과거" 라이다 좌표계의 점을 "현재" 라이다 좌표계로 한 번에 변환 
                # 흐름 : 과거 센서 -> 과거 차량 -> Global -> 현재 차량 -> 현재 라이다(ref)

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])
                # "과거" 프레임의 .bin 파일 경로
                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']
                #"기준 시간"(ref_time)과의 시간 차이(time_lag, 예: -0.05초)를 계산

                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm, # 과거 센서 -> 현재 센서 변환 행렬
                    'global_from_car': global_from_car, # 과거 차량 -> 글로벌
                    'car_from_current': car_from_current, # 과거 센서 -> 과거 차량
                    'time_lag': time_lag, # 타임스탬프 차이
                }
                sweeps.append(sweep) # sweeps 리스트에 sweep 딕셔너리 저장

        info['sweeps'] = sweeps # 9개의 과거 프레임 정보가 담긴 sweeps 리스트를 메인 info 딕셔너리에 저장

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
            # v1.0-test가 아닐 경우, sample에 속한 모든 정답 목록을 가져온다. 

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0) # 포인트가 0개인 가짜 정답 박스를 걸러내기 위한 mask를 생성

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3) 
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh) KITTI는 w,l,h 순서인데 nusc는 l,w,h라서 교환
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1) # 쿼터니언 회전값을 yaw로 변환
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            # nuScenes 객체 리스트에서 OpenPCDet가 사용하는 numpy 배열 포맷으로 변환
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)
            # 변환된 것들을 다 합쳐 (N,9) 크기의 최종 정답 박스 배열을 만든다!

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask] 
            # 맨 위에 map_name_from_general_to_detection 딕셔너리를 통해 gt_name을 변환해준다. 
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_radar_pts[mask] 
            # mask 필터링을 해주고

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info) # 이렇게 처리한 sample의 sence_token이 train_scenes 목록에 있으면 train_train_nusc_infos에 저장
        else:
            val_nusc_infos.append(info) # val이면 val_nusc_infos에 저장

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos # 이후에 nuscenes_dataset.py의 create_nuscenes_info 함수로 return -> pkl 파일로 저장


# ============== 여기까지가 Offline pre-processing 부분 ================

def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details
