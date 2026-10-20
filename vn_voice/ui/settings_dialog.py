"""Settings window: API keys, models, voice engine, reading behaviour and hotkeys."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Hotkeys, Settings, get_secret, set_secret
from ..hotkeys import parse_hotkey
from ..reader import MODELS as CLAUDE_MODELS
from ..tts import ENGINE_NAMES
from ..tts.elevenlabs_engine import MODELS as ELEVEN_MODELS

LOCAL_MODES = {
    "custom": "Built-in voices: 9 voices, strongest emotion control (installed)",
    "design": "Designed voices: unlimited, described in words (downloads another 4.5 GB)",
}
ENGINE_FIELDS = ("engine", "elevenlabs_model", "elevenlabs_stability", "local_mode", "local_device")


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    label.setOpenExternalLinks(True)
    return label


def _form_page() -> tuple[QWidget, QFormLayout]:
    """A tab page whose form stays at the top instead of spreading rows apart."""
    page = QWidget()
    outer = QVBoxLayout(page)
    form = QFormLayout()
    form.setVerticalSpacing(8)
    outer.addLayout(form)
    outer.addStretch(1)
    return page, form


def _combo(options: dict[str, str], current: str, editable: bool = False) -> QComboBox:
    box = QComboBox()
    box.setEditable(editable)
    for key, label in options.items():
        box.addItem(label, key)
    index = box.findData(current)
    if index >= 0:
        box.setCurrentIndex(index)
    elif editable:
        box.setEditText(current)
    return box


def _combo_value(box: QComboBox) -> str:
    if box.isEditable():
        text = box.currentText().strip()
        index = box.findText(text)
        return box.itemData(index) if index >= 0 else text
    return box.currentData()


class KeyField(QWidget):
    """Password-style API key box. Empty means 'keep the saved key'."""

    def __init__(self, secret_name: str, placeholder: str):
        super().__init__()
        self.secret_name = secret_name
        self.cleared = False
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.Password)
        saved = get_secret(secret_name)
        self.edit.setPlaceholderText("Saved (leave blank to keep)" if saved else placeholder)
        show = QCheckBox("Show")
        show.toggled.connect(lambda on: self.edit.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(show)

    def commit(self) -> bool:
        """Save a newly typed key. Returns True if the key changed."""
        value = self.edit.text().strip()
        if not value:
            return False
        set_secret(self.secret_name, value)
        return True


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        self._settings = settings
        self.result_settings: Settings | None = None
        self.engine_changed = False

        tabs = QTabWidget()
        tabs.addTab(self._claude_tab(), "Claude (reading)")
        tabs.addTab(self._voices_tab(), "Voices")
        tabs.addTab(self._reading_tab(), "Reading")
        tabs.addTab(self._hotkeys_tab(), "Hotkeys")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    # -- tabs
    def _claude_tab(self) -> QWidget:
        s = self._settings
        page, form = _form_page()
        self.anthropic_key = KeyField("anthropic", "sk-ant-...")
        form.addRow("Anthropic API key", self.anthropic_key)
        form.addRow("", _note('Get one at <a href="https://console.anthropic.com/settings/keys" '
                              'style="color:#b48cff">console.anthropic.com</a>. It is stored in Windows '
                              "Credential Manager, not in a file."))
        self.claude_model = _combo(CLAUDE_MODELS, s.claude_model, editable=True)
        form.addRow("Model", self.claude_model)
        self.claude_effort = _combo({"low": "Low (fastest)", "medium": "Medium", "high": "High (slowest)"},
                                    s.claude_effort)
        form.addRow("Effort", self.claude_effort)
        self.include_scene = QCheckBox("Also send a small picture of the whole scene")
        self.include_scene.setChecked(s.include_scene)
        form.addRow("Scene", self.include_scene)
        form.addRow("", _note("Lets Claude see facial expressions and who is on screen, for a better tone. "
                              "Roughly doubles the image cost per line."))
        self.context_lines = QSpinBox()
        self.context_lines.setRange(0, 20)
        self.context_lines.setValue(s.context_lines)
        form.addRow("Recent lines sent as context", self.context_lines)
        form.addRow("", _note("Rough cost per line: Opus 5.5 about $0.01, Sonnet 5.5 about $0.005, "
                              "Haiku 5.5 well under $0.001. The status bar shows the running total."))
        return page

    def _voices_tab(self) -> QWidget:
        s = self._settings
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QFormLayout()
        self.engine = _combo(ENGINE_NAMES, s.engine)
        top.addRow("Voice engine", self.engine)
        layout.addLayout(top)

        eleven = QGroupBox("ElevenLabs")
        form = QFormLayout(eleven)
        self.eleven_key = KeyField("elevenlabs", "Your ElevenLabs API key")
        form.addRow("API key", self.eleven_key)
        form.addRow("", _note('Create one at <a href="https://elevenlabs.io/app/settings/api-keys" '
                              'style="color:#b48cff">elevenlabs.io</a> with Text to Speech and Voices '
                              "(read) access."))
        self.eleven_model = _combo(ELEVEN_MODELS, s.elevenlabs_model, editable=True)
        form.addRow("Model", self.eleven_model)
        self.stability = QSlider(Qt.Horizontal)
        self.stability.setRange(0, 100)
        self.stability.setValue(round(s.elevenlabs_stability * 100))
        row = QHBoxLayout()
        row.addWidget(QLabel("Expressive"))
        row.addWidget(self.stability, 1)
        row.addWidget(QLabel("Steady"))
        form.addRow("Delivery", row)
        form.addRow("", _note("v4 and v3 use three steps (expressive, natural, steady); the slider snaps to the "
                              "nearest one. Expressive acts more but can be less predictable."))
        layout.addWidget(eleven)

        local = QGroupBox("Local GPU (Qwen3-TTS)")
        form = QFormLayout(local)
        self.local_mode = _combo(LOCAL_MODES, s.local_mode)
        form.addRow("Voices", self.local_mode)
        self.local_device = QLineEdit(s.local_device)
        form.addRow("Device", self.local_device)
        form.addRow("", _note("Runs on your graphics card: free and private, about 4.5 GB of video memory."))
        layout.addWidget(local)

        volume_row = QFormLayout()
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 150)
        self.volume.setValue(round(s.volume * 100))
        volume_row.addRow("Volume", self.volume)
        layout.addLayout(volume_row)
        layout.addStretch(1)
        return page

    def _reading_tab(self) -> QWidget:
        s = self._settings
        page, form = _form_page()
        self.read_narration = QCheckBox("Read narration aloud")
        self.read_narration.setChecked(s.read_narration)
        self.read_thoughts = QCheckBox("Read inner thoughts aloud")
        self.read_thoughts.setChecked(s.read_thoughts)
        self.interrupt = QCheckBox("Stop the current line when the game moves on (and skip lines clicked past)")
        self.interrupt.setChecked(s.interrupt_on_advance)
        self.start_on_launch = QCheckBox("Start following the game as soon as the app opens")
        self.start_on_launch.setChecked(s.start_on_launch)
        for box in (self.read_narration, self.read_thoughts, self.interrupt, self.start_on_launch):
            form.addRow(box)

        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.05, 5.0)
        self.sensitivity.setSingleStep(0.05)
        self.sensitivity.setSuffix(" % of the box")
        self.sensitivity.setValue(s.sensitivity * 100)
        form.addRow("New line when this much changes", self.sensitivity)
        self.settle = QSpinBox()
        self.settle.setRange(150, 3000)
        self.settle.setSingleStep(50)
        self.settle.setSuffix(" ms")
        self.settle.setValue(s.settle_ms)
        form.addRow("Wait for text to stop changing", self.settle)
        self.force_after = QDoubleSpinBox()
        self.force_after.setRange(0, 30)
        self.force_after.setSuffix(" s  (0 = never)")
        self.force_after.setValue(s.force_after_s)
        form.addRow("Read anyway if the box keeps moving", self.force_after)
        self.poll = QSpinBox()
        self.poll.setRange(80, 1000)
        self.poll.setSingleStep(20)
        self.poll.setSuffix(" ms")
        self.poll.setValue(s.poll_interval_ms)
        form.addRow("Check the screen every", self.poll)
        form.addRow(_note("If lines get missed, lower the change threshold. If the same line is read twice "
                          "or text is cut off mid-typing, raise the wait time."))
        return page

    def _hotkeys_tab(self) -> QWidget:
        h = self._settings.hotkeys
        page, form = _form_page()
        self.hk = {
            "read_now": QLineEdit(h.read_now),
            "toggle": QLineEdit(h.toggle),
            "replay": QLineEdit(h.replay),
            "stop": QLineEdit(h.stop),
        }
        form.addRow("Read the current line now", self.hk["read_now"])
        form.addRow("Start / pause following", self.hk["toggle"])
        form.addRow("Replay the last line", self.hk["replay"])
        form.addRow("Stop speaking", self.hk["stop"])
        form.addRow(_note("Work anywhere, even while the game has focus. Examples: F8, Ctrl+Alt+R, Shift+F2. "
                          "Leave a box empty to turn that hotkey off."))
        return page

    # -- save
    def _save(self) -> None:
        for edit in self.hk.values():
            combo = edit.text().strip()
            if combo:
                try:
                    parse_hotkey(combo)
                except ValueError as e:
                    QMessageBox.warning(self, "Hotkey", f"'{combo}' isn't a valid hotkey ({e}).")
                    return
        updated = self._settings.model_copy(update={
            "claude_model": _combo_value(self.claude_model),
            "claude_effort": self.claude_effort.currentData(),
            "include_scene": self.include_scene.isChecked(),
            "context_lines": self.context_lines.value(),
            "engine": self.engine.currentData(),
            "elevenlabs_model": _combo_value(self.eleven_model),
            "elevenlabs_stability": self.stability.value() / 100,
            "local_mode": self.local_mode.currentData(),
            "local_device": self.local_device.text().strip() or "cuda",
            "volume": self.volume.value() / 100,
            "read_narration": self.read_narration.isChecked(),
            "read_thoughts": self.read_thoughts.isChecked(),
            "interrupt_on_advance": self.interrupt.isChecked(),
            "start_on_launch": self.start_on_launch.isChecked(),
            "sensitivity": round(self.sensitivity.value() / 100, 5),
            "settle_ms": self.settle.value(),
            "force_after_s": self.force_after.value(),
            "poll_interval_ms": self.poll.value(),
            "hotkeys": Hotkeys(**{action: edit.text().strip() for action, edit in self.hk.items()}),
        })
        try:
            self.anthropic_key.commit()
            eleven_changed = self.eleven_key.commit()
        except Exception as e:
            QMessageBox.warning(self, "API keys", f"Couldn't save the API key: {e}")
            return
        # A new ElevenLabs key only matters if ElevenLabs is the engine in use (don't reload the GPU model).
        self.engine_changed = (eleven_changed and updated.engine == "elevenlabs") or any(
            getattr(updated, f) != getattr(self._settings, f) for f in ENGINE_FIELDS
        )
        self.result_settings = updated
        self.accept()
