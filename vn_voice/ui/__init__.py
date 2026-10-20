"""Qt user interface. This module holds the shared theme."""
from __future__ import annotations

import hashlib

ACCENT = "#b48cff"
ACCENT_2 = "#ff8fb8"
GOOD = "#6ee7b7"
WARN = "#fbbf24"
BAD = "#f87171"
MUTED = "#9a93ad"

STYLE = f"""
* {{ font-family: "Segoe UI", sans-serif; font-size: 10pt; color: #ece9f5; }}
QMainWindow, QDialog {{ background: #15131c; }}
QWidget#central {{ background: #15131c; }}
QFrame#card {{ background: #1e1b27; border: 1px solid #332e44; border-radius: 10px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 9pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#speaker {{ font-size: 18pt; font-weight: 700; }}
QLabel#line {{ font-size: 14pt; }}
QLabel#delivery {{ color: {MUTED}; font-style: italic; }}
QLabel#muted, QLabel#meta {{ color: {MUTED}; }}
QLabel#banner {{ background: #3b1f2b; border: 1px solid {BAD}; border-radius: 8px; padding: 8px; }}
QLabel#chip {{ border-radius: 9px; padding: 2px 10px; font-weight: 600; color: #15131c; }}
QLabel#preview {{ background: #0f0d14; border: 1px dashed #332e44; border-radius: 6px; color: {MUTED}; }}
QPushButton {{ background: #2a2638; border: 1px solid #3b3550; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ background: #352f47; }}
QPushButton:pressed {{ background: #403857; }}
QPushButton:disabled {{ color: #5d576e; }}
QPushButton#primary {{ background: {ACCENT}; color: #15131c; font-weight: 700; border: none; }}
QPushButton#primary:hover {{ background: #c4a3ff; }}
QPushButton#primary:checked {{ background: {ACCENT_2}; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: #262233; border: 1px solid #3b3550; border-radius: 6px; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: #262233; selection-background-color: #403857; }}
QTableWidget, QListWidget {{ background: #191621; border: 1px solid #332e44; border-radius: 6px;
    gridline-color: #2a2638; alternate-background-color: #1d1a26; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: #403857; }}
QHeaderView::section {{ background: #262233; color: {MUTED}; border: none; padding: 5px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid #332e44; border-radius: 8px; top: -1px; }}
QTabBar::tab {{ background: #1e1b27; padding: 7px 14px; border-top-left-radius: 7px; border-top-right-radius: 7px; }}
QTabBar::tab:selected {{ background: #2a2638; }}
QGroupBox {{ border: 1px solid #332e44; border-radius: 8px; margin-top: 14px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {MUTED}; }}
QStatusBar {{ background: #1a1722; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; }}
QSplitter::handle {{ background: #15131c; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3b3550; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: #262233; border: 1px solid #3b3550; color: #ece9f5; }}
"""

_EMOTION_COLORS = {
    "#fcd34d": ("happy", "excited", "playful", "relieved"),
    "#f9a8d4": ("teasing", "affectionate", "shy", "embarrassed", "smug"),
    "#93c5fd": ("sad", "hurt", "crying", "tired", "bored"),
    "#fca5a5": ("angry", "annoyed"),
    "#c4b5fd": ("scared", "nervous", "worried", "pained"),
    "#5eead4": ("surprised", "confused"),
}


def emotion_color(emotion: str) -> str:
    for color, names in _EMOTION_COLORS.items():
        if emotion in names:
            return color
    return "#cbd5e1"


def name_color(name: str) -> str:
    """A stable, readable colour per character name."""
    if not name:
        return "#cbd5e1"
    from PySide6.QtGui import QColor

    hue = hashlib.md5(name.casefold().encode()).digest()[0] / 255.0
    return QColor.fromHslF(hue, 0.75, 0.72).name()
