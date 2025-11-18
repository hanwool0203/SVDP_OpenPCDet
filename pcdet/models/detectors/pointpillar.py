from .detector3d_template import Detector3DTemplate # 부모 클래스를 상속

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset) # 부모 클래스를 초기화
        self.module_list = self.build_networks() 
        # build_networks() 함수가 .yaml 파일의 MODEL 섹션을 읽어서 필요한 부품을 읽는다.
        # (VFE, MapToBEV, Backbone2D, DenseHead)을 순서대로 조립하여 self.module_list에 올려둠!

    def forward(self, batch_dict): # forward path 부분
        for cur_module in self.module_list: # module_list에 올려둔 부품들을 하나씩 순서대로 가동
            batch_dict = cur_module(batch_dict) # 각 단계가 끝날 때마다 처리된 데이터가 batch_dict에 계속 추가

        if self.training: # train.py 모드
            loss, tb_dict, disp_dict = self.get_training_loss() # Loss를 계산하고 반환 -> 이 Loss로 backpropagation 진행
            # tb_dict: **텐서보드(TensorBoard)**에 기록할 로그용 딕셔너리입니다. (예: 현재 step의 분류 loss는 얼마, 회귀 loss는 얼마...)
            # disp_dict: 터미널의 프로그레스 바(tqdm) 옆에 실시간으로 보여줄 간단한 정보 딕셔너리
            ret_dict = {
                'loss': loss
            } # train 엔진에 전달할 최종 결과 딕셔너리 -> 가장 중요한 Loss 텐서가 여기 있음!
            return ret_dict, tb_dict, disp_dict
        
        else: # test.py, demo.py 모드
            pred_dicts, recall_dicts = self.post_processing(batch_dict) # Loss는 필요 없고 최종 박스 결과가 필요 -> 
             # 부모 클래스의 self.post_processing 함수를 호출 -> .yaml의 POST_PROCESSING 설정(NMS 등)을 적용하여 최종적으로 깔끔한 박스들만 남겨서 반환
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss() # 모든 Loss는 dense_head (AnchorHeadMulti)에서 나옴. -> 최종 Loss
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        } # 로깅용 딕셔너리

        loss = loss_rpn # 계산된 총 Loss 텐서입니다. (분류 Loss + 위치 Loss + 방향 Loss 합산값)
        return loss, tb_dict, disp_dict
