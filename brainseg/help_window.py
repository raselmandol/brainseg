from PyQt6 import QtWidgets, QtCore


class HelpWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help - BrainSeg")
        self.setMinimumSize(600, 500)
        self.setModal(False)
        
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
        <h2 style="color: #2c3e50;">Keyboard Shortcuts</h2>
        <table cellpadding="8" style="width: 100%;">
            <tr style="background-color: #f0f0f0;">
                <th align="left" style="padding: 8px;"><b>Shortcut</b></th>
                <th align="left" style="padding: 8px;"><b>Action</b></th>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+O</code></td>
                <td style="padding: 8px;">Open Image</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px;"><code>Ctrl+Alt+R</code></td>
                <td style="padding: 8px;">Run Segmentation</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Ctrl+Alt+S</code></td>
                <td style="padding: 8px;">Save Mask</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px;"><code>Ctrl+Alt+Q</code></td>
                <td style="padding: 8px;">Exit Application</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>F</code></td>
                <td style="padding: 8px;">Fit all views to window</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 8px;"><code>1</code></td>
                <td style="padding: 8px;">Reset zoom to 1:1 (all views)</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><code>Mouse Wheel</code></td>
                <td style="padding: 8px;">Zoom in/out on image</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
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
        <h2 style="color: #2c3e50;">How to Use BrainSeg</h2>
        
        <h3 style="color: #34495e;">Step 1: Select Model File</h3>
        <p>Click the <b>"Select Model File"</b> button in the control panel to choose a trained model file (.pth).</p>
        <p>The default model is <code>brain_segmentation_model.pth</code> in the application directory.</p>
        
        <h3 style="color: #34495e;">Step 2: Load an Image</h3>
        <p>Click <b>"Open Image"</b> or press <code>Ctrl+O</code> to select a brain MRI image.</p>
        <p>Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP</p>
        
        <h3 style="color: #34495e;">Step 3: Run Segmentation</h3>
        <p>Click <b>"Run Segmentation"</b> or press <code>Ctrl+R</code> to process the image.</p>
        <p>The application will display:</p>
        <ul>
            <li><b>Original Image</b> - Your input image</li>
            <li><b>Segmented Mask</b> - Binary mask of detected abnormalities</li>
            <li><b>Highlighted Tumor</b> - Original image with abnormalities outlined in red</li>
        </ul>
        
        <h3 style="color: #34495e;">Step 4: Save Results</h3>
        <p>Use <b>"Save Mask"</b> (Ctrl+S) or <b>"Save Highlight"</b> to save the processed images.</p>
        <p>Output files are automatically named: <code>originalname_mask.png</code> and <code>originalname_highlight.png</code></p>
        
        <h3 style="color: #34495e;">Navigation Tips</h3>
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
        <h2 style="color: #2c3e50; text-align: center;">BrainSeg</h2>
        <p style="text-align: center; color: #7f8c8d; font-size: 14px;">Brain Abnormality Segmentation Application</p>
        <p style="text-align: center; font-size: 13px;"><b>Version:</b> 0.1.0</p>
        
        <hr style="margin: 20px 0;">
        
        <h3 style="color: #34495e;">About</h3>
        <p>BrainSeg is a specialized tool for detecting and segmenting abnormalities in brain MRI images using deep learning.</p>
        <p>The application uses a U-Net architecture with EfficientNet-B7 encoder for accurate segmentation of brain tumors and other abnormalities.</p>
        
        <h3 style="color: #34495e;">Features</h3>
        <ul>
            <li>Real-time brain MRI segmentation</li>
            <li>Interactive image viewing with zoom and pan</li>
            <li>Multiple output formats (mask and highlighted views)</li>
            <li>Support for various image formats</li>
            <li>GPU acceleration support (when available)</li>
            <li>Model hot-swapping capability</li>
        </ul>
        
        <h3 style="color: #34495e;">Technology Stack</h3>
        <ul>
            <li><b>Framework:</b> PyQt6</li>
            <li><b>Deep Learning:</b> PyTorch, Segmentation Models PyTorch</li>
            <li><b>Image Processing:</b> OpenCV, NumPy, SciPy</li>
        </ul>
        
        <h3 style="color: #34495e;">Developer</h3>
        <p><b>Md. Rasel Mandol</b><br>
        Smart Systems & Connectivity Lab<br>
        National Institute of Technology Meghalaya</p>
        
        <h3 style="color: #34495e;">License</h3>
        <p>This software is provided for research and educational purposes.</p>
        
        <p style="margin-top: 30px; text-align: center; color: #95a5a6; font-size: 12px;">
        © 2025 BrainSeg. All rights reserved.
        </p>
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
