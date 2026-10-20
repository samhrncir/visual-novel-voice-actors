"""Entry point: python -m vn_voice"""
from __future__ import annotations

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler

from .config import DATA_DIR, load_profile, load_settings
from .screen import enable_dpi_awareness


def _setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(DATA_DIR / "vn_voice.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    ]
    if sys.stderr is not None:  # pythonw has no console
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers)
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "elevenlabs", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    _setup_logging()
    enable_dpi_awareness()  # before Qt, so screen grabs and the picker agree on pixels

    from PySide6.QtWidgets import QApplication, QMessageBox

    from .ui import STYLE
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VN Voice Actors")
    app.setStyleSheet(STYLE)

    def excepthook(exc_type, exc, tb):
        logging.getLogger("vn_voice").error("Unhandled error", exc_info=(exc_type, exc, tb))
        QMessageBox.critical(None, "VN Voice Actors", "".join(traceback.format_exception_only(exc_type, exc)))

    sys.excepthook = excepthook

    settings = load_settings()
    window = MainWindow(settings, load_profile(settings.active_profile))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
