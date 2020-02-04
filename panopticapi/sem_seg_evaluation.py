import os
import numpy as np
import json
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id, merge_thing_instances

OFFSET = 256 * 256 * 256
VOID = 0


class SemSegStat():
    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._N = num_classes + 1
        self._tp = np.zeros(self._N, np.int64)
        self._pos_gt = np.zeros(self._N, np.int64)
        self._pos_pred = np.zeros(self._N, np.int64)

    def __iadd__(self, sem_seg_stat):
        self._tp += sem_seg_stat._tp
        self._pos_gt += sem_seg_stat._pos_gt
        self._pos_pred += sem_seg_stat._pos_pred
        return self

    def sem_seg_average(self, categories, isthing):
        continuous_ids = []
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            continuous_ids.append(label_info['continuous_id'])

        assert len(continuous_ids) == len(set(continuous_ids))
        assert len(continuous_ids) > 0 and 0 not in continuous_ids

        acc = np.zeros(len(continuous_ids), dtype=np.float)
        iou = np.zeros(len(continuous_ids), dtype=np.float)

        tp = self._tp[continuous_ids].astype(np.float)
        pos_gt = self._pos_gt[continuous_ids].astype(np.float)
        pos_pred = self._pos_pred[continuous_ids].astype(np.float)

        class_weights = pos_gt / np.sum(pos_gt)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        return {'macc': macc, 'miou': miou, 'fiou': fiou, 'pacc': pacc}


@get_traceback
def sem_seg_compute_single_core(
    proc_id, annotation_set, gt_folder, pred_folder, categories
):
    sem_seg_stat = SemSegStat(len(categories))

    merge_things = True

    for gt_ann, pred_ann in annotation_set:
        pan_gt = np.array(
            Image.open(os.path.join(gt_folder, gt_ann['file_name'])),
            dtype=np.uint32
        )
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(
            Image.open(os.path.join(pred_folder, pred_ann['file_name'])),
            dtype=np.uint32
        )
        pan_pred = rgb2id(pan_pred)

        gt_segms_info = gt_ann['segments_info']
        pred_segms_info = pred_ann['segments_info']
        if merge_things:
            pan_gt, gt_segms_info = merge_thing_instances(
                pan_gt, gt_segms_info, is_gt=True
            )
            pan_pred, pred_segms_info = merge_thing_instances(
                pan_pred, pred_segms_info, is_gt=False
            )

        gt_segms = {el['id']: el for el in gt_segms_info}
        pred_segms = {el['id']: el for el in pred_segms_info}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_segms_info)
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        gt_id2continuous_id = {
            label: categories[label_info['category_id']]['continuous_id']
            for label, label_info in gt_segms.items()
        }
        pred_id2continuous_id = {
            label: categories[label_info['category_id']]['continuous_id']
            for label, label_info in pred_segms.items()
        }

        gt_id2continuous_id[VOID] = 0
        pred_id2continuous_id[VOID] = 0
        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_continuous_id = gt_id2continuous_id[gt_id]
            pred_continuous_id = pred_id2continuous_id[pred_id]

            # skip predictions on VOID ground-truth pixels
            if gt_continuous_id == 0:
                pred_continuous_id = 0

            sem_seg_stat._pos_gt[gt_continuous_id] += intersection
            sem_seg_stat._pos_pred[pred_continuous_id] += intersection
            if gt_continuous_id == pred_continuous_id:
                sem_seg_stat._tp[gt_continuous_id] += intersection

    return sem_seg_stat


def sem_seg_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(sem_seg_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    sem_seg_stat = SemSegStat(len(categories))
    for p in processes:
        sem_seg_stat += p.get()
    return sem_seg_stat


def sem_seg_metrics_compute(
    gt_json_file, pred_json_file, gt_folder=None, pred_folder=None,
):
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')

    categories = gt_json['categories']
    for i, label_info in enumerate(categories):
        label_info['continuous_id'] = i + 1
    categories = {el['id']: el for el in categories}

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    sem_seg_stat = sem_seg_compute_multi_core(
        matched_annotations_list, gt_folder, pred_folder, categories
    )

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name] = sem_seg_stat.sem_seg_average(
            categories, isthing=isthing
        )

    return results
