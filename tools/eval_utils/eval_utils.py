import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    # test.py로 부터 인자를 받고 평가 결과를 저장할 폴더를 만듬.
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True) # --save_to_file이 켜져 있으면, 모델이 예측한 박스 좌표 등을 텍스트 파일로 저장할 폴더를 만듬.
        # 이 옵션 키면 다른 모듈에서 받아서 쓰기가 좀 편할 것 같은데?

    # Recall 통계 초기화 
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset # dataset: NuScenesDataset 객체
    class_names = dataset.class_names
    det_annos = [] # det_annos: 모델이 예측한 **모든 결과(답안지)**를 모아둘 빈 리스트

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval() # 모델을 평가 모드로 전환

    # ========== 평가(inference Loop) 루프 ===========
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time() # 전체 평가 시간 측정 시작

    # 데이터 로드 : __getitem__과 collate_batch를 통해 만든 batch_dict를 하나 가져와서 GPU로 옮깁니다.
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        # 추론 실행 
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict) 
            # PointPillar.forward가 실행 , ret_dict: Recall 계산을 위한 중간 정보 (ROI, RCNN 단계별 예측 결과)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict) # 통계 업데이트: 방금 배치에서 나온 Recall 정보를 전체 통계(metric)에 누적
        
        # 답안지 작성 : 모델의 예측 결과(pred_dicts)를 사람이 읽을 수 있는 형태(딕셔너리)로 변환
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    # ============== 결과 집계 및 Recall 출력 ================
    # 전체 평가에 걸린 시간을 샘플 수로 나누어, 샘플당 처리 시간을 출력
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    
    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]
    # Recall 계산: 누적된 Recall 카운트를 총 정답 개수(gt_num)로 나누어 최종 Recall 비율을 계산하고 로그에 출력
    # (참고: PointPillars는 1-stage 모델이라 roi_recall은 항상 0이고, rcnn_recall이 실제 최종 Recall입니다.)
    # recall_roi는 2-stage detector에서 1단계 RoI의 Recall을 의미해서 여기선 0이 나옴!
    gt_num_cnt = metric['gt_num'] # 연못에 있는 총 물고기 수 (실제 정답 개수)
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: # [0.3, 0.5, 0.7] 기준을 순회하며
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        # 결과 출력: "Recall (IoU 0.5 기준): 0.65" -> "0.5 이상 겹친 기준으로 65% 찾았습니다!"
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    # 샘플당 평균 몇 개의 객체를 예측했는지 통계를 출력
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # ========= 최종 평가 ===============
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f) # 답안지 백업

    # 채점 요청 전체 답안지(det_annos)를 넘겨주고, 최종 성적표를 받아온다. 
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict # 최종 성적표(mAP, NDS 등)를 로그에 출력하고 반환하며 평가를 마침.


if __name__ == '__main__':
    pass
