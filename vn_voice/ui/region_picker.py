"""Full-screen overlay on the game monitor: drag a box around the dialogue (text + name plate)."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from . import ACCENT

HELP = (
    "Drag a box around the dialogue text <b>and</b> the speaker's name plate. "
    "Leave out animated parts if you can (like a blinking arrow). "
    "Press <b>Enter</b> or double-click to confirm, <b>Esc</b> to cancel."
)


class RegionPicker(QWidget):
    picked = Signal(list)  # [left, top, width, height] in screenshot pixels

    def __init__(self, image: QImage, screen, current: list[int] | None = None):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self._image = image
        self._origin: QPoint | None = None
        self._rect = QRect()
        if screen is not None:
            self.setScreen(screen)
            self.setGeometry(screen.geometry())

        bar = QWidget(self)
        bar.setObjectName("pickerBar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        # Scoped to the bar so the buttons keep the app's normal button styles.
        bar.setStyleSheet("QWidget#pickerBar { background: rgba(21,19,28,235); border-radius: 10px; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 10, 10, 10)
        text = QLabel(HELP)
        text.setWordWrap(True)
        text.setMinimumWidth(420)
        text.setStyleSheet("background: transparent; font-size: 11pt;")
        self._ok = QPushButton("Use this box")
        self._ok.setObjectName("primary")
        self._ok.setEnabled(False)
        cancel = QPushButton("Cancel")
        self._ok.clicked.connect(self._accept)
        cancel.clicked.connect(self.close)
        layout.addWidget(text, 1)
        layout.addWidget(self._ok)
        layout.addWidget(cancel)
        self._bar = bar

        # An existing box is shown once the overlay has its final size (see showEvent).
        self._initial = QRect(*current) if current and len(current) == 4 else None

    # -- coordinate mapping between the widget (logical px) and the screenshot (physical px)
    def _scale(self) -> tuple[float, float]:
        return self._image.width() / max(1, self.width()), self._image.height() / max(1, self.height())

    def _to_image(self, r: QRect) -> QRect:
        sx, sy = self._scale()
        return QRect(round(r.x() * sx), round(r.y() * sy), round(r.width() * sx), round(r.height() * sy))

    def _to_widget(self, r: QRect) -> QRect:
        sx, sy = self._scale()
        return QRect(round(r.x() / sx), round(r.y() / sy), round(r.width() / sx), round(r.height() / sy))

    # -- events
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial is not None:
            self._rect = self._to_widget(self._initial)
            self._initial = None
            self._ok.setEnabled(True)
        width = min(900, self.width() - 40)
        self._bar.setGeometry((self.width() - width) // 2, 20, width, 74)
        self.activateWindow()
        self.setFocus()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.drawImage(self.rect(), self._image)
        p.fillRect(self.rect(), QColor(0, 0, 0, 150))
        if self._rect.width() > 2 and self._rect.height() > 2:
            p.drawImage(self._rect, self._image, self._to_image(self._rect))
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawRect(self._rect.adjusted(0, 0, -1, -1))
            size = self._to_image(self._rect)
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self._rect.bottomLeft() + QPoint(4, 18), f"{size.width()} x {size.height()}")
        p.end()

    def mousePressEvent(self, event) -> None:
        # Only remember where the press happened; a box starts once the mouse actually drags,
        # so a stray click (or the clicks of a double-click) doesn't wipe the current box.
        if event.button() == Qt.LeftButton:
            self._origin = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if self._origin is None:
            return
        pos = event.position().toPoint()
        if (pos - self._origin).manhattanLength() < 4:
            return
        self._rect = QRect(self._origin, pos).normalized()
        self._ok.setEnabled(self._rect.width() > 10 and self._rect.height() > 10)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._origin = None

    def mouseDoubleClickEvent(self, _event) -> None:
        self._accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._accept()

    def _accept(self) -> None:
        if self._rect.width() <= 10 or self._rect.height() <= 10:
            return
        r = self._to_image(self._rect).intersected(QRect(0, 0, self._image.width(), self._image.height()))
        self.picked.emit([r.x(), r.y(), r.width(), r.height()])
        self.close()
