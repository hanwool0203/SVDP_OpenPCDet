import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs): # 초기화
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # BEV image의 채널 수
        self.nx, self.ny, self.nz = grid_size # 격자 크기 (여기서는 0.2m x 0.2m로 되 있을 것이라고 예상..)
        assert self.nz == 1 # PointPillars는 Z축을 하나로 합치므로 nz는 항상 1이어야 한다!

    def forward(self, batch_dict, **kwargs): # 실행
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # pillar_features: VFE를 통과한 각 Pillar의 특징 벡터입니다. 크기는 (M, 64) / coords: 각 Pillar의 좌표 (batch_idx, z_idx, y_idx, x_idx)입니다. 크기는 (M, 4)
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1 # 배치 크기를 계산
        for batch_idx in range(batch_size): # batch_size = 현재 묶음 내에서의 출석 번호 ~! (몇 번째 sample인가?)
            spatial_feature = torch.zeros( # 배치 내 각 샘플에 대해 2D BEV 이미지를 저장할 빈 텐서를 생성 , 64, 1 * nx * ny)
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # 각 Pillar가 1D로 펼쳐진 BEV 이미지에서 어디에 위치해야 하는지 계산 -> 컴퓨터 구조상 인덱싱 기법임
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars # 계산된 인덱스 위치에 Pillar 특징을 채워 넣는다. 빈 공간은 0으로 유지
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # 각 sample의 BEV 이미지를 하나로 합쳐 batch 형태 (Batch, 64, nx*ny)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # 1D로 펼쳐진 이미지를 다시 2D 형태 (Batch, 64, ny, nx)로 변환
        batch_dict['spatial_features'] = batch_spatial_features # 최종 생성된 2D BEV 이미지를 batch_dict에 저장
        return batch_dict

# Z축이 1이 아닌 경우를 처리하기 위한 클래스 -> 근데 PointPillars에는 Z축을 1로 설정하여 사용
class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict