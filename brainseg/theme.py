from PyQt6 import QtGui

LIGHT_THEME = """
QMainWindow, QWidget {
    background: #f0f1ec;
    color: #2c3e50;
    font-family: 'Segoe UI', Helvetica, Arial;
    font-size: 14px;
}
QDockWidget {
    background: #f7f7f7;
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
}
QDockWidget::title {
    background: #f0f0f0;
    padding: 6px;
    font-weight: 600;
}
QPushButton {
    background-color: #fafafa;
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background-color: #e8f0fe; border-color: #b9d0ff; }
QLabel#hint { color: #666; font-size: 12px; }
QToolBar {
    background: #f0f1ec;
    border-bottom: 1px solid #e6e6e6;
    spacing: 6px;
}
QStatusBar {
    background: #f0f1ec;
    border-top: 1px solid #e6e6e6;
}
/* View titles */
QLabel#viewTitle {
    color: #1a1a1a;
}
/* Help window content defaults */
QDialog QLabel, QDialog QScrollArea QLabel {
    color: #2c3e50;
}
QDialog a { color: #1a73e8; }
"""

DARK_THEME = """
QMainWindow, QWidget {
    background: #23272e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Helvetica, Arial;
    font-size: 14px;
}
QDockWidget {
    background: #23272e;
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
}
QDockWidget::title {
    background: #23272e;
    color: #e0e0e0;
    padding: 6px;
    font-weight: 600;
}
QPushButton {
    background-color: #2c313a;
    color: #e0e0e0;
    border: 1px solid #444a57;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background-color: #3a3f4b; border-color: #b9d0ff; }
QLabel#hint { color: #b0b0b0; font-size: 12px; }
QToolBar {
    background: #23272e;
    border-bottom: 1px solid #444a57;
    spacing: 6px;
}
QStatusBar {
    background: #23272e;
    border-top: 1px solid #444a57;
}
/* View titles */
QLabel#viewTitle {
    color: #ffffff;
}
/* Help window content defaults in dark */
QDialog QLabel, QDialog QScrollArea QLabel {
    color: #e0e0e0;
}
QDialog a { color: #8ab4f8; }
"""

def get_icon_path(theme: str, assets_dir: str, style: int = 1):
    if theme == "light":
        return f"{assets_dir}/night-mode{style}.png"
    else:
        return f"{assets_dir}/white-mode{style}.png"
