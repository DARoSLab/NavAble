"""Custom BLV evaluation metric with semantic and component-level scores."""

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.registry import METRICS

from blv_pipeline.class_mapping_ade20k import DEFAULT_ADE20K_TO_BLV
from blv_pipeline.constants import IGNORE_INDEX
from blv_pipeline.data_utils import extract_semantic_instances, rle_iou
from blv_pipeline.runtime import write_json


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _precision_envelope_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = np.maximum(mpre[index - 1], mpre[index])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


@METRICS.register_module()
class BLVMetric(BaseMetric):
    """BLV metric bundle for mIoU, precision, recall, and mask AP50-95."""

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        iou_metrics: List[str] = ['mIoU', 'mAP50-95', 'Prec', 'Rec'],
        zero_shot_remap: bool = False,
        ade20k_to_blv: Optional[Dict[int, int]] = None,
        num_classes: Optional[int] = None,
        segformer_conf_threshold: float = 0.1,
        mask_fg_conf_threshold: float = 0.3,
        segformer_extra_channel_fg_only: bool = False,
        synthesize_bg_channel: bool = False,
        gt_label_map: Optional[Dict[int, int]] = None,
        output_metrics_path: Optional[str] = None,
        excluded_class_indices: Optional[List[int]] = None,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.metric_names = list(iou_metrics)
        self.zero_shot_remap = zero_shot_remap
        self.ade20k_to_blv = ade20k_to_blv or DEFAULT_ADE20K_TO_BLV
        self.num_classes = num_classes
        self.segformer_conf_threshold = max(float(segformer_conf_threshold), 0.0)
        self.mask_fg_conf_threshold = max(float(mask_fg_conf_threshold), 0.0)
        self.segformer_extra_channel_fg_only = bool(segformer_extra_channel_fg_only)
        self.synthesize_bg_channel = bool(synthesize_bg_channel)
        self.gt_label_map = {int(k): int(v) for k, v in (gt_label_map or {}).items()}
        self.output_metrics_path = output_metrics_path
        # Class indices to exclude from mean computations (kept in per-class
        # output for transparency, but NaN'd before _safe_mean so they don't
        # drag the mean down). Use for classes with zero GT coverage in the
        # eval set (e.g. turnstile in BLV val/test).
        self.excluded_class_indices = sorted({int(c) for c in (excluded_class_indices or [])})

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = self.num_classes or len(self.dataset_meta['classes'])
        for sample_index, data_sample in enumerate(data_samples):
            gt_label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy().astype(np.int64)
            gt_label = self._remap_gt_labels(gt_label)
            pred_label, score_maps = self._prepare_prediction(data_sample, num_classes)

            confusion = self._confusion_matrix(pred_label, gt_label, num_classes)
            true_positive, pred_hist, gt_hist = self._per_class_counts(
                pred_label,
                gt_label,
                num_classes,
            )
            pred_instances = extract_semantic_instances(
                pred_label,
                score_maps=score_maps,
                ignore_index=self.ignore_index,
                json_serializable=False,
            )
            gt_instances = extract_semantic_instances(
                gt_label,
                score_maps=None,
                ignore_index=self.ignore_index,
                json_serializable=False,
            )

            image_id = data_sample.get('img_path', f'image_{len(self.results)}_{sample_index}')
            for instance in pred_instances:
                instance['image_id'] = image_id
            for instance in gt_instances:
                instance['image_id'] = image_id

            self.results.append(
                dict(
                    confusion=confusion,
                    true_positive=true_positive,
                    pred_hist=pred_hist,
                    gt_hist=gt_hist,
                    pred_instances=pred_instances,
                    gt_instances=gt_instances,
                )
            )

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        num_classes = self.num_classes or len(self.dataset_meta['classes'])
        class_names = list(self.dataset_meta['classes'])

        total_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_true_positive = np.zeros(num_classes, dtype=np.float64)
        total_pred_hist = np.zeros(num_classes, dtype=np.float64)
        total_gt_hist = np.zeros(num_classes, dtype=np.float64)
        all_pred_instances: List[dict] = []
        all_gt_instances: List[dict] = []
        for result in results:
            total_confusion += result['confusion']
            total_true_positive += result.get('true_positive', np.zeros(num_classes, dtype=np.float64))
            total_pred_hist += result.get('pred_hist', np.zeros(num_classes, dtype=np.float64))
            total_gt_hist += result.get('gt_hist', np.zeros(num_classes, dtype=np.float64))
            all_pred_instances.extend(result['pred_instances'])
            all_gt_instances.extend(result['gt_instances'])

        # Prefer explicit per-class counts computed before ignore filtering.
        # This ensures predictions mapped to ignore_index are treated as misses
        # (false negatives) instead of being dropped from denominator terms.
        if np.any(total_gt_hist):
            true_positive = total_true_positive
            pred_hist = total_pred_hist
            gt_hist = total_gt_hist
        else:
            true_positive = np.diag(total_confusion).astype(np.float64)
            pred_hist = total_confusion.sum(axis=0).astype(np.float64)
            gt_hist = total_confusion.sum(axis=1).astype(np.float64)
        union = pred_hist + gt_hist - true_positive

        iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
        precision = np.divide(
            true_positive,
            pred_hist,
            out=np.full_like(true_positive, np.nan),
            where=pred_hist > 0,
        )
        recall = np.divide(
            true_positive,
            gt_hist,
            out=np.full_like(true_positive, np.nan),
            where=gt_hist > 0,
        )
        precision[(pred_hist == 0) & (gt_hist > 0)] = 0.0

        ap = self._compute_map50_95(all_pred_instances, all_gt_instances, num_classes)

        # Mask excluded class indices to NaN before computing means. Per-class
        # values still show the (possibly 0) score so the user sees there were
        # FPs but no GTs — but the class no longer drags down the average.
        iou_for_mean = iou.copy()
        ap_for_mean = ap.copy()
        precision_for_mean = precision.copy()
        recall_for_mean = recall.copy()
        for cls_idx in self.excluded_class_indices:
            if 0 <= cls_idx < num_classes:
                iou_for_mean[cls_idx] = np.nan
                ap_for_mean[cls_idx] = np.nan
                precision_for_mean[cls_idx] = np.nan
                recall_for_mean[cls_idx] = np.nan

        summary = OrderedDict(
            mIoU=round(_safe_mean(iou_for_mean) * 100.0, 2),
            **{'mAP50-95': round(_safe_mean(ap_for_mean) * 100.0, 2)},
            Prec=round(_safe_mean(precision_for_mean) * 100.0, 2),
            Rec=round(_safe_mean(recall_for_mean) * 100.0, 2),
        )

        per_class = OrderedDict(
            Class=class_names,
            IoU=np.round(np.nan_to_num(iou, nan=0.0) * 100.0, 2).tolist(),
            **{'AP50-95': np.round(np.nan_to_num(ap, nan=0.0) * 100.0, 2).tolist()},
            Prec=np.round(np.nan_to_num(precision, nan=0.0) * 100.0, 2).tolist(),
            Rec=np.round(np.nan_to_num(recall, nan=0.0) * 100.0, 2).tolist(),
        )

        table = PrettyTable()
        for key, values in per_class.items():
            table.add_column(key, values)
        print_log('per class results:', logger)
        print_log('\n' + table.get_string(), logger=logger)

        if self.output_metrics_path and is_main_process():
            payload = dict(
                summary=summary,
                per_class={name: dict(
                    IoU=per_class['IoU'][idx],
                    AP50_95=per_class['AP50-95'][idx],
                    Prec=per_class['Prec'][idx],
                    Rec=per_class['Rec'][idx],
                ) for idx, name in enumerate(class_names)},
            )
            write_json(Path(self.output_metrics_path), payload)

        metrics = dict(summary)
        metrics['per_class_iou'] = {
            name: per_class['IoU'][idx] for idx, name in enumerate(class_names)
        }
        metrics['per_class_ap'] = {
            name: per_class['AP50-95'][idx] for idx, name in enumerate(class_names)
        }
        return metrics

    def _prepare_prediction(self, data_sample: dict, num_classes: int) -> Sequence[np.ndarray]:
        if 'seg_logits' in data_sample:
            seg_logits = data_sample['seg_logits']['data'].squeeze().float().cpu()
        else:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze().long().cpu()
            one_hot = torch.nn.functional.one_hot(
                pred_label.clamp(min=0, max=num_classes - 1),
                num_classes=num_classes,
            )
            seg_logits = one_hot.permute(2, 0, 1).float()

        if self.zero_shot_remap:
            probabilities = torch.softmax(seg_logits, dim=0).numpy()
            remapped_scores = np.zeros((num_classes, *probabilities.shape[1:]), dtype=np.float32)
            for source_id, target_id in self.ade20k_to_blv.items():
                if source_id >= probabilities.shape[0] or target_id >= num_classes:
                    continue
                remapped_scores[target_id] += probabilities[source_id]
            pred_label = remapped_scores.argmax(axis=0).astype(np.uint8)
            missing = remapped_scores.max(axis=0) <= 0
            pred_label[missing] = self.ignore_index
            return pred_label, remapped_scores

        probabilities = torch.softmax(seg_logits, dim=0).numpy()

        # Bg-inclusive eval for heads with no learned bg channel (M2F, SAN):
        # synthesize bg as 1 - max(fg) so a 12-class argmax can be computed
        # against GT where bg is class (num_classes - 1). The synthesized
        # channel is only appended when the shape is short by exactly one.
        if (
            self.synthesize_bg_channel
            and probabilities.shape[0] == num_classes - 1
        ):
            bg = np.clip(1.0 - probabilities.max(axis=0, keepdims=True), 0.0, 1.0)
            probabilities = np.concatenate([probabilities, bg.astype(probabilities.dtype)], axis=0)

        # SegFormer with extra bg channel (num_classes=11, fg_only=True):
        # strip bg channel and argmax over fg classes only.
        if probabilities.shape[0] > num_classes and self.segformer_extra_channel_fg_only:
            fg_probs = probabilities[:num_classes]
            max_fg = fg_probs.max(axis=0)
            pred_label = fg_probs.argmax(axis=0).astype(np.uint8)
            if self.segformer_conf_threshold > 0.0:
                pred_label[max_fg < self.segformer_conf_threshold] = self.ignore_index
            probabilities = fg_probs
        elif probabilities.shape[0] > num_classes:
            # Safety fallback: if a head still outputs more channels than
            # num_classes (e.g. old config without per-query filtering),
            # argmax over ALL channels and map bg predictions to ignore_index.
            full_pred = probabilities.argmax(axis=0)
            pred_label = full_pred.astype(np.uint8)
            pred_label[full_pred >= num_classes] = self.ignore_index
            fg_probs = probabilities[:num_classes]
            if self.mask_fg_conf_threshold > 0.0:
                max_fg = fg_probs.max(axis=0)
                pred_label[max_fg < self.mask_fg_conf_threshold] = self.ignore_index
            probabilities = fg_probs
        else:
            # Standard path: heads output exactly num_classes channels.
            # BLVMask2FormerHead and BLVSideAdapterCLIPHead with per-query
            # filtering land here — simple argmax over fg classes.
            max_prob = probabilities.max(axis=0)
            pred_label = probabilities.argmax(axis=0).astype(np.uint8)
            if self.segformer_conf_threshold > 0.0:
                pred_label[max_prob < self.segformer_conf_threshold] = self.ignore_index

        return pred_label, probabilities

    def _remap_gt_labels(self, gt_label: np.ndarray) -> np.ndarray:
        if not self.gt_label_map:
            return gt_label

        remapped = np.full_like(gt_label, fill_value=self.ignore_index)
        remapped[gt_label == self.ignore_index] = self.ignore_index
        for source_id, target_id in self.gt_label_map.items():
            remapped[gt_label == source_id] = target_id
        return remapped

    def _per_class_counts(
        self,
        pred_label: np.ndarray,
        gt_label: np.ndarray,
        num_classes: int,
    ) -> Sequence[np.ndarray]:
        # Honest counting: predictions on gt==ignore_index pixels still count
        # toward pred_hist (false positives), so spurious fg predictions on
        # background regions are penalized. gt_hist is unaffected because
        # those pixels have no GT class. Pixels where pred==ignore_index
        # (low-confidence preds) are naturally excluded by pred_mask below.
        # Only pixels where gt==ignore_index AND pred==ignore_index are
        # treated as "no signal" and ignored.
        true_positive = np.zeros(num_classes, dtype=np.float64)
        pred_hist = np.zeros(num_classes, dtype=np.float64)
        gt_hist = np.zeros(num_classes, dtype=np.float64)

        gt_valid = gt_label != self.ignore_index
        for class_id in range(num_classes):
            gt_mask = gt_label == class_id
            pred_mask = pred_label == class_id
            gt_hist[class_id] = float(np.count_nonzero(gt_mask))
            pred_hist[class_id] = float(np.count_nonzero(pred_mask))
            true_positive[class_id] = float(np.count_nonzero(gt_mask & pred_mask & gt_valid))

        return true_positive, pred_hist, gt_hist

    def _confusion_matrix(
        self,
        pred_label: np.ndarray,
        gt_label: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        valid = gt_label != self.ignore_index
        pred = pred_label[valid]
        gt = gt_label[valid]
        valid_pred = pred != self.ignore_index
        pred = pred[valid_pred]
        gt = gt[valid_pred]
        encoded = num_classes * gt + pred
        return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    def _compute_map50_95(
        self,
        predictions: Sequence[dict],
        ground_truths: Sequence[dict],
        num_classes: int,
    ) -> np.ndarray:
        thresholds = np.arange(0.5, 1.0, 0.05)
        ap_per_class = np.full(num_classes, np.nan, dtype=np.float64)
        preds_by_class: Dict[int, List[dict]] = defaultdict(list)
        gts_by_class: Dict[int, List[dict]] = defaultdict(list)

        for prediction in predictions:
            preds_by_class[int(prediction['category_id'])].append(prediction)
        for ground_truth in ground_truths:
            gts_by_class[int(ground_truth['category_id'])].append(ground_truth)

        for class_id in range(num_classes):
            class_gts = gts_by_class.get(class_id, [])
            if not class_gts:
                continue
            class_preds = preds_by_class.get(class_id, [])
            threshold_scores = []
            for threshold in thresholds:
                threshold_scores.append(self._average_precision_at_threshold(class_preds, class_gts, threshold))
            ap_per_class[class_id] = float(np.mean(threshold_scores))
        return ap_per_class

    def _average_precision_at_threshold(
        self,
        predictions: Sequence[dict],
        ground_truths: Sequence[dict],
        threshold: float,
    ) -> float:
        if not ground_truths:
            return float('nan')
        if not predictions:
            return 0.0

        gt_by_image: Dict[object, List[dict]] = defaultdict(list)
        matched: Dict[object, List[bool]] = {}
        for ground_truth in ground_truths:
            gt_by_image[ground_truth['image_id']].append(ground_truth)
        for image_id, gt_instances in gt_by_image.items():
            matched[image_id] = [False] * len(gt_instances)

        ordered_predictions = sorted(
            predictions,
            key=lambda item: float(item.get('score', 0.0)),
            reverse=True,
        )
        true_positives = np.zeros(len(ordered_predictions), dtype=np.float64)
        false_positives = np.zeros(len(ordered_predictions), dtype=np.float64)

        for index, prediction in enumerate(ordered_predictions):
            candidates = gt_by_image.get(prediction['image_id'], [])
            if not candidates:
                false_positives[index] = 1.0
                continue

            ious = rle_iou(prediction['rle'], [candidate['rle'] for candidate in candidates])
            best_iou = -1.0
            best_match = -1
            for gt_index, iou in enumerate(ious.tolist()):
                if matched[prediction['image_id']][gt_index]:
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_index

            if best_match >= 0 and best_iou >= threshold:
                matched[prediction['image_id']][best_match] = True
                true_positives[index] = 1.0
            else:
                false_positives[index] = 1.0

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        recalls = cumulative_tp / max(len(ground_truths), 1)
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        return _precision_envelope_ap(recalls, precisions)
