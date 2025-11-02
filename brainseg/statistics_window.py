from PyQt6 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .statistics_tracker import statistics_tracker

class StatisticsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation Statistics")
        self.setMinimumSize(700, 500)
        layout = QtWidgets.QVBoxLayout(self)
        self._add_all_content(layout)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

    def _add_all_content(self, layout):
        metrics = statistics_tracker.get_metrics()
        stats = statistics_tracker.run_statistical_tests()
        def fmt(val, unit=None, places=3):
            if val is None:
                return "N/A"
            if isinstance(val, float):
                s = f"{val:.{places}f}"
            else:
                s = str(val)
            return f"{s} {unit}" if unit else s

        # Metrics summary
        text = f"""
        <b>Latency p50:</b> {fmt(metrics['latency_p50'], 's')}<br>
        <b>Latency p95:</b> {fmt(metrics['latency_p95'], 's')}<br>
        <b>Memory Peak:</b> {fmt(metrics['memory_peak'], 'MB', 1)}<br>
        <b>Accuracy Mean:</b> {fmt(metrics['accuracy_mean'])}<br>
        <b>Accuracy Delta:</b> {fmt(metrics['accuracy_delta'])}<br>
        <b>Model Load Time:</b> {fmt(metrics['model_load_time'], 's')}<br>
        """
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)

        # Graphs
        graphs_widget = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(graphs_widget)
        # Latency, Memory, Accuracy, Model Load Time
        n_graphs = 2
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            n_graphs += 1
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
            n_graphs += 1
        fig, axs = plt.subplots(1, n_graphs, figsize=(4*n_graphs, 3))
        if n_graphs == 1:
            axs = [axs]
        axs[0].plot(metrics['latencies'], label='Latency (s)')
        axs[0].set_title('Latency')
        axs[0].legend()
        axs[1].plot(metrics['memory_peaks'], label='Memory (MB)', color='orange')
        axs[1].set_title('Memory Peak')
        axs[1].legend()
        idx = 2
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            axs[idx].plot(metrics['accuracies'], label='Accuracy', color='green')
            axs[idx].set_title('Accuracy')
            axs[idx].legend()
            idx += 1
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
            axs[idx].plot(metrics['model_load_times'], label='Model Load Time (s)', color='purple')
            axs[idx].set_title('Model Load Time')
            axs[idx].legend()
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        v.addWidget(canvas)
        layout.addWidget(graphs_widget)

        # Comparison
        if stats:
            comp_text = f"<b>t-statistic:</b> {fmt(stats['t_stat'])}<br><b>p-value:</b> {fmt(stats['p_value'])}"
        else:
            comp_text = "<b>Comparison:</b> N/A (no candidate model metrics set or no accuracy data)"
        comp_label = QtWidgets.QLabel(comp_text)
        comp_label.setWordWrap(True)
        layout.addWidget(comp_label)
