"""Main window: controls, the line being spoken, the game screen, history and the cast list."""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..casting import NARRATOR_KEY
from ..config import AGES, Profile, Settings, get_secret, list_profiles, load_profile, save_profile, save_settings
from ..hotkeys import GlobalHotkeys
from ..pipeline import LineRecord, Pipeline
from ..screen import Grabber, Monitor, clamp_region, list_monitors
from ..tts import ENGINE_NAMES
from . import ACCENT, BAD, GOOD, MUTED, WARN, emotion_color, name_color
from .region_picker import RegionPicker
from .settings_dialog import SettingsDialog

log = logging.getLogger(__name__)

GENDERS = ["unknown", "female", "male"]
AGE_CHOICES = ["unknown", *AGES]
DEFAULT_PROFILE = "My Visual Novel"
COL_NAME, COL_GENDER, COL_AGE, COL_VOICE, COL_SPEED, COL_LINES, COL_DESC = range(7)


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()


def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)
    if title:
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        layout.addWidget(label)
    return frame, layout


def muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted")
    label.setWordWrap(True)
    return label


class MainWindow(QMainWindow):
    hotkey_pressed = Signal(str)

    def __init__(self, settings: Settings, profile: Profile):
        super().__init__()
        self.settings = settings
        self.pipeline = Pipeline(settings, profile)
        self._engine_state = ("loading", "Starting the voice engine…")
        self._monitors: list[Monitor] = []
        self._picker: RegionPicker | None = None
        self._hotkeys: GlobalHotkeys | None = None
        self._hotkey_hint = ""
        self._placed = False
        self._last_error = ""
        self.setWindowTitle("VN Voice Actors")
        self.resize(1440, 840)
        self._build()
        self._wire()
        self._register_hotkeys()
        self._load_monitors()
        self._refresh_profiles()
        self._sync_engine_combo()
        self._rebuild_cast()
        self._refresh_setup()
        self._refresh_stats()
        self.pipeline.start()
        if settings.start_on_launch:
            QTimer.singleShot(800, lambda: self.pipeline.set_auto(True))

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 8)
        root.setSpacing(10)

        bar = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start reading")
        self.start_btn.setObjectName("primary")
        self.start_btn.setCheckable(True)
        self.start_btn.setMinimumWidth(160)
        self.read_btn = QPushButton("Read now")
        self.replay_btn = QPushButton("Replay")
        self.stop_btn = QPushButton("Stop voice")
        for button in (self.start_btn, self.read_btn, self.replay_btn, self.stop_btn):
            bar.addWidget(button)
        bar.addStretch(1)
        bar.addWidget(muted("Game"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(190)
        bar.addWidget(self.profile_combo)
        self.new_game_btn = QPushButton("New game…")
        bar.addWidget(self.new_game_btn)
        bar.addSpacing(12)
        bar.addWidget(muted("Voices"))
        self.engine_combo = QComboBox()
        for engine_id, label in ENGINE_NAMES.items():
            self.engine_combo.addItem(label, engine_id)
        bar.addWidget(self.engine_combo)
        self.settings_btn = QPushButton("Settings")
        bar.addWidget(self.settings_btn)
        root.addLayout(bar)

        self.banner = QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        root.addWidget(self.banner)
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self.banner.hide)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ---- left column
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        self.setup_card, sl = card("Getting started")
        self.setup_label = QLabel()
        self.setup_label.setTextFormat(Qt.RichText)
        self.setup_label.setWordWrap(True)
        sl.addWidget(self.setup_label)
        lv.addWidget(self.setup_card)

        now, nl = card("Now speaking")
        head = QHBoxLayout()
        self.speaker_label = QLabel("—")
        self.speaker_label.setObjectName("speaker")
        self.emotion_chip = QLabel()
        self.emotion_chip.setObjectName("chip")
        self.emotion_chip.hide()
        self.speaking_label = QLabel()
        self.speaking_label.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
        head.addWidget(self.speaker_label)
        head.addSpacing(8)
        head.addWidget(self.emotion_chip)
        head.addStretch(1)
        head.addWidget(self.speaking_label)
        nl.addLayout(head)
        self.line_label = QLabel("Press Start, then play your visual novel.")
        self.line_label.setObjectName("line")
        self.line_label.setWordWrap(True)
        self.line_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.line_label.setMinimumHeight(60)
        nl.addWidget(self.line_label)
        self.delivery_label = QLabel()
        self.delivery_label.setObjectName("delivery")
        self.delivery_label.setWordWrap(True)
        nl.addWidget(self.delivery_label)
        self.meta_label = QLabel()
        self.meta_label.setObjectName("meta")
        nl.addWidget(self.meta_label)
        self.status_label = muted("Paused. Press Start (or the toggle hotkey) to follow the game.")
        nl.addWidget(self.status_label)
        lv.addWidget(now)

        screen_card, sc = card("Game screen")
        row = QHBoxLayout()
        self.monitor_combo = QComboBox()
        row.addWidget(self.monitor_combo, 1)
        self.select_btn = QPushButton("Select dialogue box…")
        row.addWidget(self.select_btn)
        sc.addLayout(row)
        self.preview_label = QLabel("No dialogue box selected yet.")
        self.preview_label.setObjectName("preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(100)
        self.preview_label.setMaximumHeight(150)
        # Don't let the picture dictate the panel width; it is rescaled on resize instead.
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._preview_pixmap: QPixmap | None = None
        sc.addWidget(self.preview_label)
        lv.addWidget(screen_card)

        hist, hl = card("History (double-click a line to hear it again)")
        self.history_list = QListWidget()
        self.history_list.setWordWrap(True)
        self.history_list.setAlternatingRowColors(True)
        hl.addWidget(self.history_list, 1)
        lv.addWidget(hist, 1)
        splitter.addWidget(left)

        # ---- right column: the cast
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        cast_card, cl = card("Cast")
        self.cast_table = QTableWidget(0, 7)
        self.cast_table.setHorizontalHeaderLabels(["Character", "Gender", "Age", "Voice", "Speed", "Lines",
                                                   "Described as"])
        header = self.cast_table.horizontalHeader()
        for col in (COL_NAME, COL_GENDER, COL_AGE, COL_SPEED, COL_LINES):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_VOICE, QHeaderView.Interactive)
        header.setSectionResizeMode(COL_DESC, QHeaderView.Stretch)
        self.cast_table.setColumnWidth(COL_VOICE, 215)
        self.cast_table.verticalHeader().setVisible(False)
        self.cast_table.verticalHeader().setDefaultSectionSize(36)
        self.cast_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cast_table.setSelectionMode(QTableWidget.SingleSelection)
        self.cast_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cl.addWidget(self.cast_table, 1)
        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("▶  Preview voice")
        self.recast_btn = QPushButton("Pick another voice")
        self.remove_btn = QPushButton("Remove")
        for button in (self.preview_btn, self.recast_btn, self.remove_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        cl.addLayout(buttons)
        cl.addWidget(muted("New characters get a voice automatically the first time they speak, matched to "
                           "their gender, age and personality. Change anything here; it's saved per game."))
        rv.addWidget(cast_card, 1)
        splitter.addWidget(right)
        splitter.setSizes([560, 880])

        self.engine_status = QLabel()
        self.engine_status.setTextFormat(Qt.RichText)
        self.stats_label = QLabel()
        self.statusBar().addWidget(self.engine_status, 1)
        self.statusBar().addPermanentWidget(self.stats_label)

    def _wire(self) -> None:
        p = self.pipeline
        p.status.connect(self._on_status)
        p.error.connect(self._on_error)
        p.line_ready.connect(self._on_line)
        p.cast_changed.connect(self._rebuild_cast)
        p.preview.connect(self._show_preview)
        p.speaking.connect(lambda on: self.speaking_label.setText("● Speaking" if on else ""))
        p.engine_state.connect(self._on_engine_state)
        p.stats_changed.connect(self._refresh_stats)
        p.auto_changed.connect(self._on_auto)
        self.start_btn.clicked.connect(lambda checked: self.pipeline.set_auto(checked))
        self.read_btn.clicked.connect(lambda: self.pipeline.read_now())
        self.replay_btn.clicked.connect(lambda: self.pipeline.replay_last())
        self.stop_btn.clicked.connect(lambda: self.pipeline.stop_audio())
        self.profile_combo.activated.connect(self._on_profile_chosen)
        self.new_game_btn.clicked.connect(self._new_game)
        self.engine_combo.activated.connect(self._on_engine_chosen)
        self.settings_btn.clicked.connect(self._open_settings)
        self.monitor_combo.activated.connect(self._on_monitor_chosen)
        self.select_btn.clicked.connect(self._select_region)
        self.history_list.itemDoubleClicked.connect(lambda item: self.pipeline.replay(item.data(Qt.UserRole)))
        self.preview_btn.clicked.connect(self._preview_selected)
        self.cast_table.cellDoubleClicked.connect(lambda _r, _c: self._preview_selected())
        self.recast_btn.clicked.connect(self._recast_selected)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.hotkey_pressed.connect(self._on_hotkey)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(5000)

    # ------------------------------------------------------------------ window placement
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._placed:
            return
        self._placed = True
        game = self._qscreen_for(self._current_monitor())
        if game is not None and self.screen() == game:
            other = next((s for s in QGuiApplication.screens() if s != game), None)
            if other is not None:
                geo = other.availableGeometry()
                self.move(geo.center() - self.rect().center())

    # ------------------------------------------------------------------ hotkeys
    def _register_hotkeys(self) -> None:
        if self._hotkeys is not None:
            self._hotkeys.stop()
        h = self.settings.hotkeys
        bindings = {"read_now": h.read_now, "toggle": h.toggle, "replay": h.replay, "stop": h.stop}
        self._hotkeys = GlobalHotkeys(bindings, self.hotkey_pressed.emit)
        self._hotkeys.start()
        if self._hotkeys.failed:
            self._on_error("Some hotkeys couldn't be set: " + "; ".join(self._hotkeys.failed))
        hints = [f"{h.toggle} start/pause", f"{h.read_now} read now", f"{h.replay} replay", f"{h.stop} stop"]
        self._hotkey_hint = " · ".join(hint for hint in hints if not hint.startswith(" "))

    def _on_hotkey(self, action: str) -> None:
        p = self.pipeline
        if action == "toggle":
            p.set_auto(not p.auto)
        elif action == "read_now":
            p.read_now()
        elif action == "replay":
            p.replay_last()
        elif action == "stop":
            p.stop_audio()

    # ------------------------------------------------------------------ monitors and the dialogue box
    def _qscreen_for(self, monitor: Monitor | None):
        if monitor is None:
            return None
        screens = QGuiApplication.screens()
        for screen in screens:  # Qt keeps each screen's native top-left position on Windows
            geo = screen.geometry()
            if geo.x() == monitor.left and geo.y() == monitor.top:
                return screen
        ordered = sorted(screens, key=lambda s: (s.geometry().x(), s.geometry().y()))
        index = next((i for i, m in enumerate(self._monitors) if m.name == monitor.name), -1)
        return ordered[index] if 0 <= index < len(ordered) else None

    def _current_monitor(self) -> Monitor | None:
        name = self.pipeline.profile.monitor
        return next((m for m in self._monitors if m.name == name), None)

    def _load_monitors(self) -> None:
        self._monitors = list_monitors()
        self.monitor_combo.clear()
        for i, monitor in enumerate(self._monitors):
            screen = self._qscreen_for(monitor)
            model = f" · {screen.name()}" if screen is not None and screen.name() else ""
            self.monitor_combo.addItem(monitor.label(i) + model, monitor.name)
        profile = self.pipeline.profile
        if self._current_monitor() is None and self._monitors:
            # Guess: the game is on a screen other than the main one.
            guess = next((m for m in self._monitors if not m.primary), self._monitors[0])
            profile.monitor = guess.name
            self.pipeline.geometry_changed()
        index = self.monitor_combo.findData(profile.monitor)
        if index >= 0:
            self.monitor_combo.setCurrentIndex(index)
        self._snapshot_region()

    def _on_monitor_chosen(self, index: int) -> None:
        name = self.monitor_combo.itemData(index)
        if name != self.pipeline.profile.monitor:
            self.pipeline.profile.monitor = name
            self.pipeline.geometry_changed()
            self._snapshot_region()
            self._refresh_setup()

    def _grab_monitor(self, monitor: Monitor) -> np.ndarray | None:
        try:
            grabber = Grabber()
            try:
                return grabber.grab_monitor(monitor)
            finally:
                grabber.close()
        except Exception as e:
            self._on_error(f"Couldn't capture that screen: {e}")
            return None

    def _snapshot_region(self) -> None:
        monitor, region = self._current_monitor(), self.pipeline.profile.region
        if monitor is None or not region:
            self._preview_pixmap = None
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("No dialogue box selected yet.")
            return
        box = clamp_region(region, monitor)
        shot = self._grab_monitor(monitor)
        if shot is not None and box is not None:
            left, top, width, height = box
            self._show_preview(shot[top:top + height, left:left + width])

    def _select_region(self) -> None:
        monitor = self._current_monitor()
        if monitor is None:
            self._on_error("Pick the screen the game is on first.")
            return
        shot = self._grab_monitor(monitor)
        if shot is None:
            return
        self._last_shot = shot
        self._picker = RegionPicker(to_qimage(shot), self._qscreen_for(monitor), self.pipeline.profile.region)
        self._picker.picked.connect(self._on_region_picked)
        self._picker.show()

    def _on_region_picked(self, rect: list) -> None:
        self.pipeline.profile.region = [int(v) for v in rect]
        self.pipeline.geometry_changed()
        left, top, width, height = rect
        self._show_preview(self._last_shot[top:top + height, left:left + width])
        self._on_status("Dialogue box saved. Press Start when you're ready.")
        self._refresh_setup()

    def _show_preview(self, rgb: np.ndarray) -> None:
        if rgb is None or rgb.size == 0:
            return
        self._preview_pixmap = QPixmap.fromImage(to_qimage(rgb))
        self._fit_preview()

    def _fit_preview(self) -> None:
        if self._preview_pixmap is None:
            return
        self.preview_label.setPixmap(self._preview_pixmap.scaled(
            max(10, self.preview_label.width() - 8), self.preview_label.maximumHeight() - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_preview()

    # ------------------------------------------------------------------ games (profiles)
    def _refresh_profiles(self) -> None:
        names = list_profiles()
        active = self.pipeline.profile.name
        if active not in names:
            names.append(active)
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(set(names), key=str.casefold))
        self.profile_combo.setCurrentText(active)

    def _on_profile_chosen(self, index: int) -> None:
        name = self.profile_combo.itemText(index)
        if name and name != self.pipeline.profile.name:
            self._switch_profile(load_profile(name))

    def _new_game(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New game", "Name of the visual novel (Claude uses it to recognise the characters):"
        )
        name = name.strip()
        if not ok or not name:
            return
        profile = load_profile(name)
        if not profile.monitor:
            profile.monitor = self.pipeline.profile.monitor
        save_profile(profile)
        self._switch_profile(profile)

    def _switch_profile(self, profile: Profile) -> None:
        self.pipeline.cast.save()
        self.settings = self.settings.model_copy(update={"active_profile": profile.name})
        save_settings(self.settings)
        self.pipeline.settings = self.settings
        self.pipeline.set_profile(profile)
        self.history_list.clear()
        self.speaker_label.setText("—")
        self.line_label.setText("New game selected.")
        self.delivery_label.clear()
        self.meta_label.clear()
        self.emotion_chip.hide()
        self._load_monitors()
        self._refresh_profiles()
        self._refresh_setup()

    # ------------------------------------------------------------------ settings and engine
    def _sync_engine_combo(self) -> None:
        index = self.engine_combo.findData(self.settings.engine)
        if index >= 0:
            self.engine_combo.setCurrentIndex(index)

    def _on_engine_chosen(self, index: int) -> None:
        engine_id = self.engine_combo.itemData(index)
        if engine_id != self.settings.engine:
            self._apply_settings(self.settings.model_copy(update={"engine": engine_id}), engine_changed=True)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.Accepted and dialog.result_settings is not None:
            self._apply_settings(dialog.result_settings, dialog.engine_changed)
        self._refresh_setup()

    def _apply_settings(self, settings: Settings, engine_changed: bool) -> None:
        hotkeys_changed = settings.hotkeys != self.settings.hotkeys
        self.settings = settings
        save_settings(settings)
        self.pipeline.apply_settings(settings, engine_changed)
        if hotkeys_changed:
            self._register_hotkeys()
        self._sync_engine_combo()
        self._refresh_setup()
        self._refresh_stats()

    def _on_engine_state(self, state: str, message: str) -> None:
        self._engine_state = (state, message)
        color = {"ready": GOOD, "loading": WARN}.get(state, BAD)
        self.engine_status.setText(f"<span style='color:{color}'>●</span>&nbsp; {message}")
        if state == "error":
            self._on_error(message)
        if state == "ready":
            self._rebuild_cast()
        self._refresh_setup()

    # ------------------------------------------------------------------ live updates
    def _on_auto(self, on: bool) -> None:
        self.start_btn.setChecked(on)
        self.start_btn.setText("⏸  Pause" if on else "▶  Start reading")

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_error(self, text: str) -> None:
        log.warning("UI error: %s", text)
        self.banner.setText(text)
        self.banner.show()
        self._banner_timer.start(12000)
        self.status_label.setText(text)

    def _on_line(self, record: LineRecord) -> None:
        line = record.line
        name = record.character or "Narration"
        self.speaker_label.setText(name)
        self.speaker_label.setStyleSheet(f"color: {name_color(record.character)};")
        self.emotion_chip.setText(f"{line.emotion}  {'●' * line.intensity}")
        self.emotion_chip.setStyleSheet(f"background: {emotion_color(line.emotion)};")
        self.emotion_chip.show()
        self.line_label.setText(line.text)
        delivery = line.delivery
        if line.volume != "normal":
            delivery = f"{delivery} · {line.volume}" if delivery else line.volume
        self.delivery_label.setText(delivery)
        meta = [f"Voice: {record.voice.name}" if record.voice else "No voice",
                f"read in {record.read_seconds:.1f}s", f"${record.cost:.4f}"]
        if record.note:
            meta.append(record.note)
        self.meta_label.setText(" · ".join(meta))

        item = QListWidgetItem(f"{record.at}   {name}  [{line.emotion}]   {line.text}")
        item.setData(Qt.UserRole, record)
        item.setToolTip(line.delivery or line.emotion)
        if not record.spoken:
            item.setForeground(QColor(MUTED))
        self.history_list.insertItem(0, item)
        while self.history_list.count() > 300:
            self.history_list.takeItem(self.history_list.count() - 1)
        self._update_line_counts()

    def _refresh_stats(self) -> None:
        p = self.pipeline
        noun = "line" if p.lines_read == 1 else "lines"
        parts = [f"Claude: {p.lines_read} {noun} read, about ${p.session_cost:.3f} this session"]
        if p.engine is not None:
            parts.append(p.engine.usage_text())
        if self._hotkey_hint:
            parts.append(self._hotkey_hint)
        self.stats_label.setText("    |    ".join(part for part in parts if part))

    def _refresh_setup(self) -> None:
        has_key = bool(get_secret("anthropic"))
        state, message = self._engine_state
        profile = self.pipeline.profile
        has_box = bool(profile.monitor and profile.region)

        def row(ok: bool | None, done: str, todo: str) -> str:
            mark = {True: f"<span style='color:{GOOD}'>✓</span>", None: f"<span style='color:{WARN}'>…</span>",
                    False: f"<span style='color:{BAD}'>✗</span>"}[ok]
            return f"<p style='margin:3px 0'>{mark}&nbsp;&nbsp;{done if ok else todo}</p>"

        engine_ok = True if state == "ready" else (None if state == "loading" else False)
        rows = [
            row(has_key, "Claude API key saved", "Add your <b>Anthropic API key</b> in Settings (Claude reads the screen)"),
            row(engine_ok, message, message),
            row(has_box, "Dialogue box selected",
                "Pick the game's screen below, then click <b>Select dialogue box…</b>"),
        ]
        if profile.name == DEFAULT_PROFILE and not profile.characters:
            rows.append(f"<p style='margin:3px 0; color:{MUTED}'>Tip: click <b>New game…</b> and type the game's "
                        "title so Claude can recognise its characters.</p>")
        self.setup_card.setVisible(not (has_key and state == "ready" and has_box))
        self.setup_label.setText("".join(rows))

    # ------------------------------------------------------------------ cast table
    def _selected_key(self) -> str | None:
        row = self.cast_table.currentRow()
        item = self.cast_table.item(row, COL_NAME) if row >= 0 else None
        return item.data(Qt.UserRole) if item is not None else None

    def _rebuild_cast(self) -> None:
        focus = QApplication.focusWidget()
        if isinstance(focus, QLineEdit) and self.cast_table.isAncestorOf(focus):
            QTimer.singleShot(1500, self._rebuild_cast)  # don't wipe a description mid-edit
            return
        cast = self.pipeline.cast
        engine = self.pipeline.engine
        voices = engine.voices() if engine is not None else []
        keys = cast.keys()
        selected = self._selected_key()
        self.cast_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            character = cast.get(key)
            if character is None:
                continue
            is_narrator = key == NARRATOR_KEY
            name = QTableWidgetItem("Narrator" if is_narrator else character.name)
            name.setData(Qt.UserRole, key)
            name.setForeground(QColor(MUTED if is_narrator else name_color(character.name)))
            name.setToolTip(character.description or "")
            self.cast_table.setItem(row, COL_NAME, name)

            gender = QComboBox()
            gender.addItems(GENDERS)
            gender.setCurrentText(character.gender)
            gender.currentTextChanged.connect(lambda v, k=key: self.pipeline.cast.update(k, gender=v))
            self.cast_table.setCellWidget(row, COL_GENDER, gender)

            age = QComboBox()
            for choice in AGE_CHOICES:
                age.addItem(choice.replace("_", " "), choice)
            age.setCurrentIndex(max(0, age.findData(character.age)))
            age.currentIndexChanged.connect(
                lambda _i, k=key, box=age: self.pipeline.cast.update(k, age=box.currentData()))
            self.cast_table.setCellWidget(row, COL_AGE, age)

            voice_box = QComboBox()
            current = character.voices.get(engine.id, "") if engine is not None else ""
            if not current:
                voice_box.addItem("Chosen when they first speak", "")
            for voice in voices:
                voice_box.addItem(voice.label(), voice.id)
                voice_box.setItemData(voice_box.count() - 1, voice.description, Qt.ToolTipRole)
            if not voices:
                voice_box.addItem("Voices still loading…", "")
                voice_box.setEnabled(False)
            index = voice_box.findData(current)
            voice_box.setCurrentIndex(max(0, index))
            voice_box.activated.connect(lambda _i, k=key, box=voice_box: self._voice_chosen(k, box))
            self.cast_table.setCellWidget(row, COL_VOICE, voice_box)

            speed = QComboBox()
            speed.setToolTip("Speaking speed")
            choices = sorted({0.8, 0.9, 1.0, 1.1, 1.2, round(character.speed, 2)})
            for value in choices:
                speed.addItem(f"{value:.1f}x" if value * 10 == int(value * 10) else f"{value:.2f}x", value)
            speed.setCurrentIndex(choices.index(round(character.speed, 2)))
            speed.activated.connect(
                lambda _i, k=key, box=speed: self.pipeline.cast.update(k, speed=float(box.currentData())))
            self.cast_table.setCellWidget(row, COL_SPEED, speed)

            lines = QTableWidgetItem(str(character.lines))
            lines.setTextAlignment(Qt.AlignCenter)
            self.cast_table.setItem(row, COL_LINES, lines)

            description = QLineEdit(character.description)
            description.setPlaceholderText("How they sound (used for casting)")
            description.editingFinished.connect(
                lambda k=key, box=description: self.pipeline.cast.update(k, description=box.text().strip()))
            self.cast_table.setCellWidget(row, COL_DESC, description)

            if key == selected:
                self.cast_table.selectRow(row)

    def _voice_chosen(self, key: str, box: QComboBox) -> None:
        engine = self.pipeline.engine
        voice_id = box.currentData()
        if engine is None or not voice_id:
            return
        self.pipeline.cast.set_voice(key, engine.id, voice_id)
        self.pipeline.preview_voice(key)

    def _update_line_counts(self) -> None:
        for row in range(self.cast_table.rowCount()):
            item = self.cast_table.item(row, COL_NAME)
            character = self.pipeline.cast.get(item.data(Qt.UserRole)) if item is not None else None
            lines = self.cast_table.item(row, COL_LINES)
            if character is not None and lines is not None:
                lines.setText(str(character.lines))

    def _preview_selected(self) -> None:
        key = self._selected_key()
        if key is None:
            self._on_status("Select a character in the cast list first.")
            return
        self.pipeline.preview_voice(key)

    def _recast_selected(self) -> None:
        key = self._selected_key()
        engine = self.pipeline.engine
        if key is None or engine is None:
            return
        if self.pipeline.cast.recast(key, engine) is not None:
            self._rebuild_cast()
            self.pipeline.preview_voice(key)

    def _remove_selected(self) -> None:
        key = self._selected_key()
        if not key:  # None or the narrator
            return
        character = self.pipeline.cast.get(key)
        answer = QMessageBox.question(self, "Remove character",
                                      f"Remove {character.name} from the cast? They'll be re-cast if they speak again.")
        if answer == QMessageBox.Yes:
            self.pipeline.cast.remove(key)
            self._rebuild_cast()

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event) -> None:
        if self._picker is not None:
            try:
                self._picker.close()
            except RuntimeError:  # already closed and deleted
                pass
        if self._hotkeys is not None:
            self._hotkeys.stop()
        self.pipeline.cast.save()
        save_settings(self.settings)
        self.pipeline.shutdown()
        event.accept()
