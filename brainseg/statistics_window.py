from PyQt6 import QtWidgets
from PyQt6 import QtCore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .statistics_tracker import statistics_tracker

class StatisticsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation Statistics")
        self.setMinimumSize(700, 500)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scrollable content area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(12)
        self._add_all_content(content_layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Single close button at bottom
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setFixedWidth(100)
        btn_close.clicked.connect(self.close)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        btn_row.addStretch(1)
        main_layout.addLayout(btn_row)

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
        metrics_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        grid = QtWidgets.QFormLayout(metrics_box)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        grid.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        labels = [
            ("Latency p50", fmt(metrics['latency_p50'], 's')),
            ("Latency p95", fmt(metrics['latency_p95'], 's')),
            ("Memory Peak", fmt(metrics['memory_peak'], 'MB', 1)),
            ("Accuracy Mean", fmt(metrics['accuracy_mean'])),
            ("Accuracy Delta", fmt(metrics['accuracy_delta'])),
            ("Model Load Time", fmt(metrics['model_load_time'], 's')),
        ]
        for name, value in labels:
            lbl_name = QtWidgets.QLabel(name + ":")
            lbl_name.setStyleSheet("font-weight: 500;")
            lbl_val = QtWidgets.QLabel(value)
            lbl_val.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 13px;")
            grid.addRow(lbl_name, lbl_val)
        layout.addWidget(metrics_box)

        # --- Graphs Section ---
        graphs_box = QtWidgets.QGroupBox("Performance Graphs")
        graphs_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        v = QtWidgets.QVBoxLayout(graphs_box)

        # Apply a professional matplotlib style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass
        plt.rcParams.update({
            'axes.edgecolor': '#cccccc',
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.frameon': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.family': 'Segoe UI'
        })

        # Determine grid layout (up to 2 columns)
        n_graphs = 2
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            n_graphs += 1
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
            n_graphs += 1
        cols = 2 if n_graphs > 1 else 1
        rows = int(np.ceil(n_graphs / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5.2*cols, 2.8*rows), constrained_layout=True)
        axs = np.array(axs).reshape(-1) if isinstance(axs, (list, np.ndarray)) else np.array([axs])

        # Plot 1: Latency (with p50/p95 bands if data exists)
        ax = axs[0]
        ax.plot(metrics['latencies'], label='Latency (s)', color='#1976d2', linewidth=1.8)
        ax.set_title('Latency per Run')
        ax.set_xlabel('Run #')
        ax.set_ylabel('Seconds')
        if metrics['latencies']:
            p50 = np.percentile(metrics['latencies'], 50)
            p95 = np.percentile(metrics['latencies'], 95)
            ax.axhline(p50, color='#455a64', linestyle='--', linewidth=1, label='p50')
            ax.axhline(p95, color='#b71c1c', linestyle='--', linewidth=1, label='p95')
        ax.legend(loc='best')

        # Plot 2: Memory Peak
        ax = axs[1 if n_graphs > 1 else 0]
        ax.plot(metrics['memory_peaks'], label='Memory (MB)', color='#fb8c00', linewidth=1.8)
        ax.set_title('Memory Peak per Run')
        ax.set_xlabel('Run #')
        ax.set_ylabel('MB')
        ax.legend(loc='best')

        plot_idx = 2
        # Plot 3: Accuracy (if available)
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            ax = axs[plot_idx]
            ax.plot(metrics['accuracies'], label='Accuracy', color='#43a047', linewidth=1.8)
            ax.set_title('Accuracy per Run')
            ax.set_xlabel('Run #')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.legend(loc='best')
            plot_idx += 1

        # Plot 4: Model Load Time (if available)
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0 and plot_idx < len(axs):
            ax = axs[plot_idx]
            ax.bar(range(1, len(metrics['model_load_times'])+1), metrics['model_load_times'],
                   color='#8e24aa', width=0.6, label='Load time (s)')
            ax.set_title('Model Load Time')
            ax.set_xlabel('Load #')
            ax.set_ylabel('Seconds')
            ax.legend(loc='best')

        canvas = FigureCanvas(fig)
        v.addWidget(canvas)
        layout.addWidget(graphs_box)

        # --- Comparison Section ---
        comp_box = QtWidgets.QGroupBox("Model Comparison")
        comp_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        comp_layout = QtWidgets.QHBoxLayout(comp_box)
        if stats:
            comp_text = f"t-statistic: <b>{fmt(stats['t_stat'])}</b>   |   p-value: <b>{fmt(stats['p_value'])}</b>"
        else:
            comp_text = "Comparison: N/A (no candidate model metrics set or no accuracy data)"
        comp_label = QtWidgets.QLabel(comp_text)
        comp_label.setStyleSheet("font-size: 13px; font-family: Consolas, 'Courier New', monospace;")
        comp_layout.addWidget(comp_label)
        layout.addWidget(comp_box)
