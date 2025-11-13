from PyQt6 import QtWidgets, QtCore


class HelpWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help - BrainSeg")
        self.setMinimumSize(600, 500)
        self.setModal(False)
        # Allowing theme-specific QSS to target this dialog if needed
        self.setObjectName("HelpWindow")
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Tab widget for organized content
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._create_shortcuts_tab(), "Keyboard Shortcuts")
        tabs.addTab(self._create_usage_tab(), "How to Use")
        tabs.addTab(self._create_about_tab(), "About")
        
        main_layout.addWidget(tabs)
        
        # Close button at bottom
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        button_layout.setContentsMargins(10, 10, 10, 10)
        
        main_layout.addLayout(button_layout)
        
    def _create_shortcuts_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        shortcuts_html = """
        <h2>Keyboard Shortcuts</h2>
        <table cellpadding="8" style="width: 100%;">
            <tr>
                <th align="left" style="padding: 8px;"><b>Shortcut</b></th>
                <th align="left" style="padding: 8px;"><b>Action</b></th>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+O</code></td>
                <td style="padding: 8px;">Open Image</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+R</code></td>
                <td style="padding: 8px;">Run Segmentation</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+S</code></td>
                <td style="padding: 8px;">Save Mask</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+Q</code></td>
                <td style="padding: 8px;">Exit Application</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+F</code></td>
                <td style="padding: 8px;">Fit all views to window</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+1</code></td>
                <td style="padding: 8px;">Reset zoom to 1:1 (all views)</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Mouse Wheel</code></td>
                <td style="padding: 8px;">Zoom in/out on image</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Space (hold)</code></td>
                <td style="padding: 8px;">Pan/drag image view</td>
            </tr>
        </table>
        """
        
        label = QtWidgets.QLabel(shortcuts_html)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)
        layout.addStretch()
        
        return widget
    
    def _create_usage_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        usage_html = """
        <h2>How to Use BrainSeg</h2>
        
        <h3>Step 1: Select Model File</h3>
        <p>Click the <b>"Select Model File"</b> button in the control panel to choose a trained model file (.pth).</p>
        <p>The default model is <code>brain_segmentation_model.pth</code> in the application directory.</p>
        
        <h3>Step 2: Load an Image</h3>
        <p>Click <b>"Open Image"</b> or press <code>Ctrl+Alt+O</code> to select a brain MRI image.</p>
        <p>Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP</p>
        
        <h3>Step 3: Tune the View</h3>
        <p>Use the <b>Brightness</b> and <b>Contrast</b> sliders in the control panel to match your preferred viewing profile. Adjustments update in real time without disturbing your zoom or pan state.</p>

        <h3>Step 4: Run Segmentation</h3>
        <p>Click <b>"Run Segmentation"</b> or press <code>Ctrl+Alt+R</code> to process the image.</p>
        <p>The application will display:</p>
        <ul>
            <li><b>Original Image</b> - Your input image</li>
            <li><b>Segmented Mask</b> - Binary mask of detected abnormalities</li>
            <li><b>Highlighted Tumor</b> - Original image with abnormalities outlined in red</li>
        </ul>
        
        <h3>Step 5: Save Results</h3>
        <p>Use <b>"Save Mask"</b> (Ctrl+Alt+S) or <b>"Save Highlight"</b> to save the processed images.</p>
        <p>Output files are automatically named: <code>originalname_mask.png</code> and <code>originalname_highlight.png</code></p>
        
        <h3>Navigation Tips</h3>
        <ul>
            <li>Use mouse wheel to zoom in/out on any view</li>
            <li>Hold Space and drag to pan across the image</li>
            <li>Press F to fit all views to window</li>
            <li>Press 1 to reset zoom to actual size</li>
        </ul>
        """
        
        label = QtWidgets.QLabel(usage_html)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
        layout.addWidget(scroll)
        
        return widget
    
    def _create_about_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        
        about_html = """
        <div style="max-width: 520px; margin: 0 auto;">
            <h2 style="text-align: center; letter-spacing: 0.8px;">BrainSeg</h2>
            <p style="text-align: center; font-size: 14px; color: #5f6368;">Precision brain abnormality segmentation for research-grade imaging pipelines.</p>
            <p style="text-align: center; font-size: 13px;"><b>Release:</b> 0.1.0 &nbsp;|&nbsp; <b>Last Update:</b> 2025</p>

            <hr style="margin: 24px 0;">

            <h3>Purpose</h3>
            <p>BrainSeg streamlines the evaluation of brain MRI studies by coupling state-of-the-art deep learning with an interactive analytical interface. Designed for medical imaging researchers, it accelerates hypothesis testing, model benchmarking, and exploratory analysis.</p>

            <h3>What You Can Expect</h3>
            <ul>
                <li><b>Robust segmentation:</b> U-Net (EfficientNet-B7 backbone) tuned for focal lesion delineation.</li>
                <li><b>Live analytics:</b> Latency, memory, load profiling, and ground-truth quality comparisons in one dashboard.</li>
                <li><b>Research ergonomics:</b> Fine-grained zoom/pan, brightness control, and theme-aware presentation for prolonged review sessions.</li>
                <li><b>Seamless export:</b> Save binary masks and annotated overlays with experiment-friendly naming.</li>
            </ul>

            <h3>Technology Stack</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 6px 0; width: 35%;"><b>Application</b></td>
                    <td style="padding: 6px 0;">PyQt6</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0;"><b>Inference</b></td>
                    <td style="padding: 6px 0;">PyTorch · segmentation-models-pytorch</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0;"><b>Imaging</b></td>
                    <td style="padding: 6px 0;">OpenCV · NumPy · SciPy</td>
                </tr>
                <tr>
                    <td style="padding: 6px 0;"><b>Analytics</b></td>
                    <td style="padding: 6px 0;">Matplotlib · SciPy statistics</td>
                </tr>
            </table>

            <h3>Authorship</h3>
            <p>
                <b>Md. Rasel Mandol</b><br>
                Smart Systems & Connectivity Lab, National Institute of Technology Meghalaya
            </p>

            <h3>Usage & License</h3>
            <p>This build is distributed under the MIT License for research and educational use. Please acknowledge SSC Lab when publishing results generated with BrainSeg.</p>

            <p style="margin-top: 32px; text-align: center; color: #95a5a6; font-size: 12px;">
                © 2025 Smart Systems & Connectivity Lab · BrainSeg
            </p>
        </div>
        """
        
        label = QtWidgets.QLabel(about_html)
        label.setWordWrap(True)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
        layout.addWidget(scroll)
        
        return widget
