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
        quality_summary = statistics_tracker.get_quality_summary()
        def fmt(val, unit=None, places=3):
            if val is None:
                return "N/A"
            if isinstance(val, float):
                s = f"{val:.{places}f}"
            else:
                s = str(val)
            return f"{s} {unit}" if unit else s
        def fmt_stat_block(stats_dict, unit=None):
            if not stats_dict or stats_dict.get('mean') is None:
                return "N/A"
            mean_s = fmt(stats_dict.get('mean'), unit)
            median_s = fmt(stats_dict.get('median'), unit)
            min_s = fmt(stats_dict.get('min'), unit)
            max_s = fmt(stats_dict.get('max'), unit)
            return f"mean {mean_s} | median {median_s} | range {min_s} - {max_s}"

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
            lbl_val.setStyleSheet("font-family: 'DejaVu Sans Mono', 'Liberation Mono', Menlo, Monaco, Consolas, 'Courier New', monospace; font-size: 13px;")
            grid.addRow(lbl_name, lbl_val)
        layout.addWidget(metrics_box)

        # --- Segmentation Quality Section ---
        if quality_summary and quality_summary.get('records'):
            quality_box = QtWidgets.QGroupBox("Segmentation Quality")
            quality_box.setStyleSheet("font-weight: 600; font-size: 13px;")
            q_layout = QtWidgets.QVBoxLayout(quality_box)
            q_layout.setSpacing(8)

            agg_form = QtWidgets.QFormLayout()
            agg_form.setHorizontalSpacing(16)
            agg_form.setVerticalSpacing(6)
            agg = quality_summary.get('aggregate') or {}
            agg_form.addRow("Dice", QtWidgets.QLabel(fmt_stat_block(agg.get('dice'))))
            agg_form.addRow("Jaccard", QtWidgets.QLabel(fmt_stat_block(agg.get('jaccard'))))
            agg_form.addRow("Hausdorff (px)", QtWidgets.QLabel(fmt_stat_block(agg.get('hausdorff'), 'px')))
            count_label = QtWidgets.QLabel(str(agg.get('count', 0)))
            count_label.setStyleSheet("font-family: 'DejaVu Sans Mono', 'Liberation Mono', Menlo, Monaco, Consolas, 'Courier New', monospace;")
            agg_form.addRow("Evaluated Runs", count_label)
            q_layout.addLayout(agg_form)

            latest = quality_summary.get('latest')
            if latest:
                run_name = latest.image_id or f"Run {agg.get('count', 0)}"
                latest_summary = QtWidgets.QLabel(
                    f"Latest run — <b>{run_name}</b>: Dice {fmt(latest.aggregate.get('dice'))}, "
                    f"Jaccard {fmt(latest.aggregate.get('jaccard'))}, Hausdorff {fmt(latest.aggregate.get('hausdorff'), 'px')}"
                )
                latest_summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
                latest_summary.setStyleSheet("font-size: 12px; font-weight: 400;")
                q_layout.addWidget(latest_summary)

                if latest.regions:
                    table = QtWidgets.QTableWidget(len(latest.regions), 5)
                    table.setHorizontalHeaderLabels([
                        "Region",
                        "Pixels",
                        "Dice",
                        "Jaccard",
                        "Hausdorff (px)",
                    ])
                    for row, region in enumerate(latest.regions):
                        table.setItem(row, 0, QtWidgets.QTableWidgetItem(region.get('label', f"Region {row+1}")))
                        table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(region.get('pixels', 0))))
                        table.setItem(row, 2, QtWidgets.QTableWidgetItem(fmt(region.get('dice'))))
                        table.setItem(row, 3, QtWidgets.QTableWidgetItem(fmt(region.get('jaccard'))))
                        table.setItem(row, 4, QtWidgets.QTableWidgetItem(fmt(region.get('hausdorff'), 'px')))
                    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
                    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
                    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
                    table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                    q_layout.addWidget(table)
                else:
                    q_layout.addWidget(QtWidgets.QLabel("No labeled regions detected in the latest ground truth mask."))

            layout.addWidget(quality_box)

        # --- Graphs Section ---
        graphs_box = QtWidgets.QGroupBox("Performance Graphs")
        graphs_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        v = QtWidgets.QVBoxLayout(graphs_box)

        # Apply a professional matplotlib style
        try:
            plt.style.use('seaborn-v0_8-white')
        except Exception:
            try:
                plt.style.use('seaborn-white')
            except Exception:
                pass
        plt.rcParams.update({
            'axes.edgecolor': '#cccccc',
            'axes.grid': False,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.frameon': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.family': ['Ubuntu', 'Cantarell', 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
        })

        graph_types = ['latency', 'memory']
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            graph_types.append('accuracy')
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
            graph_types.append('load_time')
        if metrics['dice_scores'] and len(metrics['dice_scores']) > 0:
            graph_types.append('dice')

        n_graphs = len(graph_types)
        cols = 2 if n_graphs > 1 else 1
        rows = int(np.ceil(n_graphs / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5.2*cols, 2.8*rows), constrained_layout=True)
        axs = np.array(axs).reshape(-1) if isinstance(axs, (list, np.ndarray)) else np.array([axs])

        for idx, graph_type in enumerate(graph_types):
            ax = axs[idx]
            ax.grid(False)
            if graph_type == 'latency':
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
            elif graph_type == 'memory':
                ax.plot(metrics['memory_peaks'], label='Memory (MB)', color='#fb8c00', linewidth=1.8)
                ax.set_title('Memory Peak per Run')
                ax.set_xlabel('Run #')
                ax.set_ylabel('MB')
                ax.legend(loc='best')
            elif graph_type == 'accuracy':
                ax.plot(metrics['accuracies'], label='Accuracy', color='#43a047', linewidth=1.8)
                ax.set_title('Accuracy per Run')
                ax.set_xlabel('Run #')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                ax.legend(loc='best')
            elif graph_type == 'load_time':
                ax.bar(range(1, len(metrics['model_load_times'])+1), metrics['model_load_times'],
                       color='#8e24aa', width=0.6, label='Load time (s)')
                ax.set_title('Model Load Time')
                ax.set_xlabel('Load #')
                ax.set_ylabel('Seconds')
                ax.legend(loc='best')
            elif graph_type == 'dice':
                ax.plot(metrics['dice_scores'], label='Dice', color='#0097a7', linewidth=1.8)
                ax.set_title('Dice per Run')
                ax.set_xlabel('Run #')
                ax.set_ylabel('Dice')
                ax.set_ylim(0, 1)
                if quality_summary and quality_summary.get('aggregate'):
                    dice_mean = quality_summary['aggregate']['dice'].get('mean')
                    if dice_mean is not None:
                        ax.axhline(dice_mean, color='#006064', linestyle='--', linewidth=1, label='Mean')
                ax.legend(loc='best')

        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(320)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        graphs_box.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
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
        comp_label.setStyleSheet("font-size: 13px; font-family: 'DejaVu Sans Mono', 'Liberation Mono', Menlo, Monaco, Consolas, 'Courier New', monospace;")
        comp_layout.addWidget(comp_label)
        layout.addWidget(comp_box)
