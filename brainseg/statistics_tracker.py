import time
import threading
import numpy as np
import psutil
from collections import deque

class StatisticsTracker:
    def __init__(self):
        self.latencies = deque(maxlen=100)
        self.memory_peaks = deque(maxlen=100)
        self.accuracies = deque(maxlen=100)
        self.model_load_times = deque(maxlen=10)
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

    def get_metrics(self):
        with self.lock:
            lat = list(self.latencies)
            mem = list(self.memory_peaks)
            acc = list(self.accuracies)
            load = list(self.model_load_times)
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
            'model_load_times': load
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
