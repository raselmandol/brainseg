from PyQt6 import QtWidgets
from PyQt6 import QtCore
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
        btn_close.setFixedWidth(100)
        btn_close.setStyleSheet("margin-top: 16px;")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
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

            # --- Metrics Section ---
            metrics_box = QtWidgets.QGroupBox("Session Metrics")
            metrics_box.setStyleSheet("font-weight: 600; font-size: 13px; margin-bottom: 8px;")
            grid = QtWidgets.QGridLayout(metrics_box)
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(6)
            labels = [
                ("Latency p50", fmt(metrics['latency_p50'], 's')),
                ("Latency p95", fmt(metrics['latency_p95'], 's')),
                ("Memory Peak", fmt(metrics['memory_peak'], 'MB', 1)),
                ("Accuracy Mean", fmt(metrics['accuracy_mean'])),
                ("Accuracy Delta", fmt(metrics['accuracy_delta'])),
                ("Model Load Time", fmt(metrics['model_load_time'], 's')),
            ]
            for i, (name, value) in enumerate(labels):
                lbl_name = QtWidgets.QLabel(name + ":")
                lbl_name.setStyleSheet("font-weight: 500;")
                lbl_val = QtWidgets.QLabel(value)
                lbl_val.setStyleSheet("font-family: monospace; font-size: 13px;")
                grid.addWidget(lbl_name, i, 0)
                grid.addWidget(lbl_val, i, 1)
            layout.addWidget(metrics_box)

            # --- Graphs Section ---
            graphs_box = QtWidgets.QGroupBox("Performance Graphs")
            graphs_box.setStyleSheet("font-weight: 600; font-size: 13px; margin-bottom: 8px;")
            v = QtWidgets.QVBoxLayout(graphs_box)
            n_graphs = 2
            if metrics['accuracies'] and len(metrics['accuracies']) > 0:
                n_graphs += 1
            if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
                n_graphs += 1
            fig, axs = plt.subplots(1, n_graphs, figsize=(4*n_graphs, 3))
            if n_graphs == 1:
                axs = [axs]
            axs[0].plot(metrics['latencies'], label='Latency (s)', color='#1976d2')
            axs[0].set_title('Latency', fontsize=11)
            axs[0].grid(True, linestyle='--', alpha=0.3)
            axs[0].legend()
            axs[1].plot(metrics['memory_peaks'], label='Memory (MB)', color='#ffa726')
            axs[1].set_title('Memory Peak', fontsize=11)
            axs[1].grid(True, linestyle='--', alpha=0.3)
            axs[1].legend()
            idx = 2
            if metrics['accuracies'] and len(metrics['accuracies']) > 0:
                axs[idx].plot(metrics['accuracies'], label='Accuracy', color='#43a047')
                axs[idx].set_title('Accuracy', fontsize=11)
                axs[idx].grid(True, linestyle='--', alpha=0.3)
                axs[idx].legend()
                idx += 1
            if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
                axs[idx].plot(metrics['model_load_times'], label='Model Load Time (s)', color='#8e24aa')
                axs[idx].set_title('Model Load Time', fontsize=11)
                axs[idx].grid(True, linestyle='--', alpha=0.3)
                axs[idx].legend()
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            v.addWidget(canvas)
            layout.addWidget(graphs_box)

            # --- Comparison Section ---
            comp_box = QtWidgets.QGroupBox("Model Comparison")
            comp_box.setStyleSheet("font-weight: 600; font-size: 13px; margin-bottom: 8px;")
            comp_layout = QtWidgets.QHBoxLayout(comp_box)
            if stats:
                comp_text = f"t-statistic: <b>{fmt(stats['t_stat'])}</b>   |   p-value: <b>{fmt(stats['p_value'])}</b>"
            else:
                comp_text = "Comparison: N/A (no candidate model metrics set or no accuracy data)"
            comp_label = QtWidgets.QLabel(comp_text)
            comp_label.setStyleSheet("font-size: 13px; font-family: monospace;")
            comp_layout.addWidget(comp_label)
            layout.addWidget(comp_box)

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