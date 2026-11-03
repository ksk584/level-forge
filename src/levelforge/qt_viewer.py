"""Optional interactive viewer using PyQt5 (already installed in this env).

Launch with:  python -m levelforge play --level path/to/level.json
Controls:     Arrow keys / WASD move, R reset, Esc quit.
"""

from __future__ import annotations

import sys

from .grid import BOX_FRAGILE, BOX_HEAVY, BOX_NORMAL, Level

CELL = 34


def run(level: Level):
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as e:  # pragma: no cover
        print("PyQt5 is not installed; use `python -m levelforge render --level ... --out level.png` instead.")
        raise SystemExit(2) from e

    from .environment import SokobanEnv

    class Board(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.env = SokobanEnv(level, max_steps=10**9)
            self.setFixedSize(level.width * CELL + 2, level.height * CELL + 2)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)

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
            if event.key() in keys:
                self.env.step(keys[event.key()])
                self.update()
                lv = self.env.level
                self.window().setWindowTitle(f"LevelForge | energy={lv.energy} | boxes on goals={lv.boxes_on_goals()}/{self.env.box_total} | step={self.env.steps}")

    app = QtWidgets.QApplication(sys.argv)
    board = Board()
    window = QtWidgets.QMainWindow()
    window.setCentralWidget(board)
    window.setWindowTitle("LevelForge — arrows/WASD move, R reset, Esc quit")
    window.show()
    sys.exit(app.exec_())
