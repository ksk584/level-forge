"""Interactive viewer using PyQt5 (already installed in this env).

Play yourself:
    python -m levelforge play  --level path/to/level.json
    Controls: Arrow keys / WASD move, R reset, Esc quit.

Watch the trained agent replay a level:
    python -m levelforge watch --level path/to/level.json --checkpoint checkpoint_gen_N.pt [--delay ms]
    Controls: space pause | R restart | +/- speed | Esc quit. Default 350 ms per move;
    the window stays open when the episode ends showing the outcome.
"""

from __future__ import annotations

import sys

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level

CELL = 34


def _make_board_cls(QtCore, QtGui, QtWidgets, level):
    from .environment import SokobanEnv

    class Board(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.env = SokobanEnv(level, max_steps=10**9)
            self.setFixedSize(level.width * CELL + 2, level.height * CELL + 2 + 28)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.status = ""

        def statustext(self):
            lv = self.env.level
            if self.env.done:
                outcome = "SOLVED!" if self.env.failure is None else f"FAILED ({self.env.failure})"
                return f"{outcome}  |  steps={self.env.steps}  |  box cost logged"
            return f"energy={lv.energy}  |  boxes on goals={lv.boxes_on_goals()}/{self.env.box_total}  |  step={self.env.steps}"

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            lv = self.env.level
            goals = set(lv.goals)
            box_map = {(x, y): k for x, y, k in lv.boxes}
            for y in range(lv.height):
                for x in range(lv.width):
                    rect = QtCore.QRect(x * CELL, y * CELL, CELL - 1, CELL - 1)
                    if lv.walls[y, x]:
                        painter.fillRect(rect, QtGui.QColor("#111827"))
                        continue
                    painter.fillRect(rect, QtGui.QColor("#f9fafb"))
                    if (x, y) in goals:
                        painter.setBrush(QtGui.QColor("#16a34a"))
                        painter.drawEllipse(x * CELL + CELL // 3, y * CELL + CELL // 3, CELL // 3, CELL // 3)
                    if (x, y) in box_map:
                        kind = box_map[(x, y)]
                        color = {BOX_NORMAL: "#c08457", BOX_HEAVY: "#6b7280", BOX_FRAGILE: "#93c5fd"}[kind]
                        painter.setBrush(QtGui.QColor(color))
                        painter.drawRoundedRect(x * CELL + 3, y * CELL + 3, CELL - 7, CELL - 7, 6, 6)
            ax, ay = lv.agent
            painter.setBrush(QtGui.QColor("#2563eb"))
            painter.drawEllipse(ax * CELL + 4, ay * CELL + 4, CELL - 9, CELL - 9)
            # status strip
            painter.fillRect(QtCore.QRect(0, lv.height * CELL, self.width(), 28), QtGui.QColor("#0f172a"))
            painter.setPen(QtGui.QColor("#e2e8f0"))
            painter.drawText(8, lv.height * CELL + 19, self.status or self.statustext())
            painter.end()

        def keyPressEvent(self, event):
            keys = {
                QtCore.Qt.Key_Up: 0, QtCore.Qt.Key_W: 0,
                QtCore.Qt.Key_Down: 1, QtCore.Qt.Key_S: 1,
                QtCore.Qt.Key_Left: 2, QtCore.Qt.Key_A: 2,
                QtCore.Qt.Key_Right: 3, QtCore.Qt.Key_D: 3,
            }
            if event.key() == QtCore.Qt.Key_R:
                self.env.reset()
                self.update()
                return
            if event.key() == QtCore.Qt.Key_Escape:
                QtWidgets.qApp.quit()
                return
            if event.key() in keys and not self.env.done:
                self.env.step(keys[event.key()])
                self.update()

    return Board


def _window(QtWidgets, board, title):
    window = QtWidgets.QMainWindow()
    window.setCentralWidget(board)
    window.setWindowTitle(title)
    window.show()


def run(level: Level):
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as e:  # pragma: no cover
        print("PyQt5 is not installed; use `python -m levelforge render --level ... --out level.png` instead.")
        raise SystemExit(2) from e

    app = QtWidgets.QApplication(sys.argv)
    Board = _make_board_cls(QtCore, QtGui, QtWidgets, level)
    board = Board()
    _window(QtWidgets, board, "LevelForge — arrows/WASD move, R reset, Esc quit")
    sys.exit(app.exec_())


def run_agent(level: Level, checkpoint: str, delay_ms: int = 350):
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as e:  # pragma: no cover
        print("PyQt5 is not installed; use `python -m levelforge render --level ... --out level.png` instead.")
        raise SystemExit(2) from e

    from .dqn import DQNAgent, flatten_obs

    try:
        agent = DQNAgent.load(checkpoint, device="cpu")
    except Exception as e:
        print(f"[levelforge] could not load checkpoint: {checkpoint}\n           {e}")
        raise SystemExit(2) from e

    app = QtWidgets.QApplication(sys.argv)
    Board = _make_board_cls(QtCore, QtGui, QtWidgets, level)

    class ReplayBoard(Board):
        def __init__(self):
            super().__init__()
            self.env.max_steps = 500  # wandering agents must not spin forever
            self.obs, _ = self.env.reset()
            self.paused = False
            self.delay_ms = int(delay_ms)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(max(20, self.delay_ms))
            self._step_once()

        def _set_delay(self, ms: int):
            self.delay_ms = max(30, min(5000, ms))
            if self.timer.isActive():
                self.timer.start(self.delay_ms)
            self.status = f"{self.statustext()}  |  speed={self.delay_ms}ms/step (+/-)"
            self.update()

        def _step_once(self):
            if self.env.done:
                outcome = "SOLVED" if self.env.failure is None else f"FAILED ({self.env.failure})"
                self.status = f"{outcome} after {self.env.steps} steps | R restart | Esc quit"
                self.timer.stop()
                self.update()
                print(f"[levelforge] replay finished: {self.status}")
                return
            vec = flatten_obs(self.obs, self.env.level.energy)
            action = agent.select_action(vec, greedy=True)
            self.obs, _r, _done, info = self.env.step(action)
            self.status = f"{self.statustext()}  |  speed={self.delay_ms}ms/step (+/-)"
            self.update()

        def tick(self):
            if self.paused:
                return
            self._step_once()

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_R:
                self.obs, _ = self.env.reset()
                self.status = ""
                self.timer.start(self.delay_ms)
                self.update()
                return
            if event.key() == QtCore.Qt.Key_Space:
                self.paused = not self.paused
                self.status = ("paused — space to resume" if self.paused else self.statustext())
                self.update()
                return
            if event.key() in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal, QtCore.Qt.Key_PageUp):
                self._set_delay(int(self.delay_ms / 1.5))  # faster
                return
            if event.key() in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore, QtCore.Qt.Key_PageDown):
                self._set_delay(int(self.delay_ms * 1.5))  # slower
                return
            super().keyPressEvent(event)

    board = ReplayBoard()
    _window(QtWidgets, board, "LevelForge — DQN replay | space pause | R replay | +/- speed | Esc quit")
    sys.exit(app.exec_())
