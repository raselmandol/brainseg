import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import find_objects
from scipy.spatial.distance import directed_hausdorff


@dataclass
class QualityRecord:
    image_id: Optional[str]
    timestamp: float
    aggregate: Dict[str, Optional[float]]
    regions: List[Dict[str, Any]]
    counts: Dict[str, int]

class StatisticsTracker:
    def __init__(self):
        self.latencies = deque(maxlen=100)
        self.memory_peaks = deque(maxlen=100)
        self.accuracies = deque(maxlen=100)
        self.model_load_times = deque(maxlen=10)
        self.quality_records = deque(maxlen=50)
        self.lock = threading.Lock()
        self.model_loaded = False
        self.model_load_start = None
        self.model_load_end = None
        self.candidate_metrics = None  # For comparison

    def start_model_load(self):
        self.model_load_start = time.perf_counter()

    def end_model_load(self):
        self.model_load_end = time.perf_counter()
        load_time = self.model_load_end - self.model_load_start
        with self.lock:
            self.model_load_times.append(load_time)
        self.model_loaded = True

    def record_segmentation(self, latency, memory_peak, accuracy=None):
        with self.lock:
            self.latencies.append(latency)
            self.memory_peaks.append(memory_peak)
            if accuracy is not None:
                self.accuracies.append(accuracy)

    @staticmethod
    def _ensure_binary(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        data = np.array(mask)
        if data.ndim > 2:
            if data.shape[2] == 4:
                if data.dtype != np.uint8:
                    data = data.astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
            else:
                if data.dtype != np.uint8:
                    data = data.astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = np.squeeze(data)
        if data.dtype == np.bool_:
            return data
        if np.issubdtype(data.dtype, np.floating):
            thresholded = data > 0.5
        else:
            thresholded = data > 0
        return thresholded.astype(bool)

    @staticmethod
    def _compute_basic_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)

        tp = int(np.logical_and(pred_bool, gt_bool).sum())
        fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
        fn = int(np.logical_and(~pred_bool, gt_bool).sum())

        if tp == 0 and fp == 0 and fn == 0:
            dice = 1.0
            jaccard = 1.0
        else:
            denom = (2 * tp + fp + fn)
            dice = (2 * tp / denom) if denom > 0 else 0.0
            union = tp + fp + fn
            jaccard = (tp / union) if union > 0 else 0.0

        pred_coords = np.argwhere(pred_bool)
        gt_coords = np.argwhere(gt_bool)
        if pred_coords.size == 0 and gt_coords.size == 0:
            hausdorff = 0.0
        elif pred_coords.size == 0 or gt_coords.size == 0:
            hausdorff = np.nan
        else:
            hausdorff = max(
                directed_hausdorff(pred_coords, gt_coords)[0],
                directed_hausdorff(gt_coords, pred_coords)[0],
            )

        return {
            "dice": float(dice),
            "jaccard": float(jaccard),
            "hausdorff": float(hausdorff) if not np.isnan(hausdorff) else np.nan,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def record_quality(self, pred_mask: Optional[np.ndarray], gt_mask: Optional[np.ndarray], image_id: Optional[str] = None) -> Optional[QualityRecord]:
        pred_bool = self._ensure_binary(pred_mask)
        gt_bool = self._ensure_binary(gt_mask)
        if pred_bool is None or gt_bool is None:
            return None
        if pred_bool.shape != gt_bool.shape:
            gt_resized = cv2.resize(gt_bool.astype(np.uint8), (pred_bool.shape[1], pred_bool.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_bool = gt_resized.astype(bool)

        overall = self._compute_basic_metrics(pred_bool, gt_bool)

        labels, n_regions = cc_label(gt_bool)
        regions: List[Dict[str, Any]] = []
        if n_regions > 0:
            slices = find_objects(labels)
            for idx, slc in enumerate(slices, start=1):
                if slc is None:
                    continue
                region_gt = (labels[slc] == idx)
                region_pred = pred_bool[slc]
                region_metrics = self._compute_basic_metrics(region_pred, region_gt)
                regions.append(
                    {
                        "label": f"Region {idx}",
                        "pixels": int(region_gt.sum()),
                        **region_metrics,
                    }
                )

        record = QualityRecord(
            image_id=image_id,
            timestamp=time.time(),
            aggregate={
                "dice": overall["dice"],
                "jaccard": overall["jaccard"],
                "hausdorff": overall["hausdorff"],
            },
            regions=regions,
            counts={
                "tp": overall["tp"],
                "fp": overall["fp"],
                "fn": overall["fn"],
            },
        )

        with self.lock:
            self.quality_records.append(record)

        return record

    def get_metrics(self):
        with self.lock:
            lat = list(self.latencies)
            mem = list(self.memory_peaks)
            acc = list(self.accuracies)
            load = list(self.model_load_times)
            quality = list(self.quality_records)
        dice_scores = [rec.aggregate["dice"] for rec in quality if rec.aggregate["dice"] is not None]
        jaccard_scores = [rec.aggregate["jaccard"] for rec in quality if rec.aggregate["jaccard"] is not None]
        hausdorff_scores = [rec.aggregate["hausdorff"] for rec in quality if rec.aggregate["hausdorff"] is not None and not np.isnan(rec.aggregate["hausdorff"]) ]
        return {
            'latency_p50': np.percentile(lat, 50) if lat else None,
            'latency_p95': np.percentile(lat, 95) if lat else None,
            'memory_peak': max(mem) if mem else None,
            'accuracy_mean': np.mean(acc) if acc else None,
            'accuracy_delta': (np.mean(acc) - np.mean(self.candidate_metrics['accuracies'])) if self.candidate_metrics and acc else None,
            'model_load_time': load[-1] if load else None,
            'latencies': lat,
            'memory_peaks': mem,
            'accuracies': acc,
            'model_load_times': load,
            'dice_scores': dice_scores,
            'jaccard_scores': jaccard_scores,
            'hausdorff_scores': hausdorff_scores,
            'quality_history': quality,
        }

    def get_quality_summary(self):
        with self.lock:
            records = list(self.quality_records)
        if not records:
            return {
                "records": [],
                "aggregate": None,
                "latest": None,
            }

        dice_vals = np.array([rec.aggregate["dice"] for rec in records], dtype=float)
        jaccard_vals = np.array([rec.aggregate["jaccard"] for rec in records], dtype=float)
        hausdorff_vals = np.array([
            rec.aggregate["hausdorff"] if rec.aggregate["hausdorff"] is not None else np.nan
            for rec in records
        ], dtype=float)

        def _safe_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
            mask = ~np.isnan(values)
            if not mask.any():
                return {"mean": None, "median": None, "min": None, "max": None}
            valid = values[mask]
            return {
                "mean": float(np.mean(valid)),
                "median": float(np.median(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            }

        aggregate = {
            "dice": _safe_stats(dice_vals),
            "jaccard": _safe_stats(jaccard_vals),
            "hausdorff": _safe_stats(hausdorff_vals),
            "count": len(records),
        }

        return {
            "records": records,
            "aggregate": aggregate,
            "latest": records[-1],
        }

    def set_candidate_metrics(self, latencies, memory_peaks, accuracies):
        self.candidate_metrics = {
            'latencies': latencies,
            'memory_peaks': memory_peaks,
            'accuracies': accuracies
        }

    def run_statistical_tests(self):
        # Simple t-test for accuracy delta
        from scipy.stats import ttest_ind
        if self.candidate_metrics and self.accuracies:
            t_stat, p_val = ttest_ind(list(self.accuracies), self.candidate_metrics['accuracies'], equal_var=False)
            return {'t_stat': t_stat, 'p_value': p_val}
        return None

# Singleton instance
statistics_tracker = StatisticsTracker()
