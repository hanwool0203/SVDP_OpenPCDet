import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module): # 하나의 Pillar(기둥) 안에 들어있는 여러 개의 점들을 하나의 특징 벡터로 요약
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False): # 초기화
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2 
            # VoxelNet 방식을 써놓은 것. PointPillars는 개별 필라 인코딩 후 바로 넘기는데, VoxelNet은 그 복셀 내의 모든 포인트의 특징을 결합해서 다음 층으로 넘김 -> 그래서 1/2를 해주어야함 !!

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False) # 선형 레이어 정의
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01) # BN 정의
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000 # 한 번에 처리할 최대 Pillar 개수

    def forward(self, inputs): # 실행 (inputs: 현재 처리할 모든 Pillar의 포인트 데이터입니다. 크기는 (M, 20, 11) 정도 됩니다. (M: Pillar 개수, 20: 포인트 수, 11: 특징 수))
        if inputs.shape[0] > self.part: # Pillar 개수가 너~~~무 많은 경우 VRAM이 딸림.. -> 데이터를 쪼개서 계산했다가 합치는 아이디어
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs) # 일반적인 경우, Pillar 수가 5만개 이하이면, (M, 20, 11) -> Linear(11, 64) -> (M, 20, 64)
            # 논문에서는 point별 특징을 9차원으로 정의하였으나 OpenPCDet에서는 11차원으로 확대함. (timestamp, z - z_pillar)
            # 여기서 궁금한 점 : PointNet에서는 Conv1D로 Shared-MLP를 구현하였으나 여기서는 Linear?
            # nn.Linear는 입력 데이터의 가장 마지막 차원을 봄. 즉 point 별 feature인 11차원의 data를 가지고 연산을 하는 것이므로 앞의 (M, 20)은 건드리지 않음.
            # 즉, M x 20개의 point가 서로 독립적으로, 동일한 가중치로 계산이 되는 것! 
            # Conv1d는 주로 (B,C,N) 형태로 다룰 때 사용, 지금은 (B,N,C)
            
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x # BatchNorm1d -> Batch의 데이터 분포를 일정하게 맞춰주어 학습을 안정화한다.
        torch.backends.cudnn.enabled = True
        x = F.relu(x) # 활성화 함수 (ReLU)
        x_max = torch.max(x, dim=1, keepdim=True)[0] # Max Pooling 결과 : (M, 1, 64) -> M : 비어 있지 않은 Pillar의 개수!

        if self.last_vfe: # PointPillars 방식 -> yaml에 필터를 하나만 지정했으므로 한 개의 PFN layer만 만든다!
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1) # VoxelNet 방식 -> 다수의 VFE layer를 사용하므로 필터가 여러개일 것!
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate): # PFN layer를 사용하여 전체 Point Cloud를 처리하는 Class
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs): # 초기화 : 공장 설비 세팅
        super().__init__(model_cfg=model_cfg) # 부모 클래스 초기화
        # .yaml 파일에서 설정을 읽어옴.
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ

        # 1. point 별 feature 세팅
        num_point_features += 6 if self.use_absolute_xyz else 3 # point 별 특징을 5개에서 11개로 확장하겠다는 이야기
        if self.with_distance: # 원점(라이다 센서)으로부터의 거리를 추가할 경우 1차원이 더 늘어남 (보통 안 씀)
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS # 필터 개수
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) # num_filters: [11, 64] 리스트가 됩니다. (입력 11채널 ➡️ 출력 64채널)

        # 2. PFN Layer 사용
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i] # 11
            out_filters = num_filters[i + 1] # 64
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            ) # PFN layer 실행
        self.pfn_layers = nn.ModuleList(pfn_layers) # (M, 1, 64) 저장

        # 각 Pillar의 기하학적 중심 좌표를 계산하기 위한 offset 값들을 미리 계산
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs): # 실행 : 실제 데이터 처리
        # 데이터(batch_dict)가 들어오면 실제로 변환하고 압축하는 핵심 과정
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # voxels : 실제 point data (M, 20, 5) / voxel_num_points : 각 Pillar에 실제로 들어 있는 Point 개수 / voxel_coord : 각 Pillar의 위치 주소(index)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1) 
        # 각 Pillar 내에 있는 모든 포인트의 (x, y, z) 좌표 합을 구하고, 실제 포인트 개수로 나눠서 Pillar별 평균 좌표 (무게중심) 구함.
        f_cluster = voxel_features[:, :, :3] - points_mean # Feature 1: 평균으로부터의 거리

        f_center = torch.zeros_like(voxel_features[:, :, :3]) # Feature 2: Pillar 중심으로부터의 거리
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center] # 특징 모으기 
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # point 당 11개의 특징을 가지도록 (M, 32, 11)을 만듬.

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        # Piiar에 포인트가 Pillar에 포인트가 32개 꽉 차지 않은 경우(예: 5개만 있음), 나머지 빈 공간(27개)의 쓰레기 값을 모두 0으로 만들어 버립니다.
        # 지금은 비어있지 않은 부분에 대해서만 가공해주고 있는 것! 뒤에 모듈에서 비어있는 Pillar의 부분을 0으로 초기화.

        # PFN Layer 실행 부분
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict # 다음 모듈로 (M, 64)차원의 데이터 전달
