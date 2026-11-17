"""Main-thread OpenCV camera/status view, with terminal single-key fallback."""

import os
import select
import sys

import numpy as np


class TerminalKeys:
    """Read commands immediately without Enter; always restore terminal flags."""

    def __init__(self):
        self.previous = None

    def __enter__(self):
        if sys.stdin.isatty():
            import termios
            import tty

            self.previous = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def poll(self):
        if self.previous is None:
            return []
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            value = os.read(sys.stdin.fileno(), 1).decode('ascii', errors='ignore')
            if value:
                keys.append('q' if value in ('\x1b', '\x03') else value.lower())
        return keys

    def __exit__(self, *_):
        if self.previous is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.previous)


class StatusView:
    """RGB camera tiles and collection health; all GUI calls on main thread."""

    def __init__(self, camera_names, max_width=1280):
        self.camera_names = list(camera_names)
        self.max_width = max_width
        self.enabled = False
        self.cv2 = None
        self.window_name = 'Pika + Piper | LeRobot Collect'

    def start(self):
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            return self
        try:
            import cv2

            self.cv2 = cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)
            self.enabled = True
        except Exception as exc:
            print('StatusView unavailable: %s' % exc, file=sys.stderr)
        return self

    def render_once(self, images, status):
        if not self.enabled:
            return None
        cv2 = self.cv2
        tiles = []
        for name in self.camera_names:
            image = images.get(name)
            if image is None:
                image = np.zeros((480, 640, 3), np.uint8)
            h, w = image.shape[:2]
            tile = cv2.resize(image[:, :, ::-1], (int(w * 300 / h), 300))
            cv2.putText(tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 40), 2)
            tiles.append(tile)
        strip = np.concatenate(tiles, axis=1)
        if strip.shape[1] > self.max_width:
            strip = cv2.resize(
                strip, (self.max_width, int(strip.shape[0] * self.max_width / strip.shape[1]))
            )
        panel = np.zeros((340, strip.shape[1], 3), np.uint8)

        def put(text, y, color=(225, 225, 225), scale=0.55):
            # Built-in OpenCV fonts render ASCII. Unicode task text is retained
            # unchanged in dataset metadata even when the view shows '?'.
            text = str(text).encode('ascii', errors='replace').decode('ascii')
            while (
                scale > 0.3
                and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                > panel.shape[1] - 20
            ):
                scale -= 0.03
            cv2.putText(
                panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
            )

        green, red, yellow = (60, 230, 60), (50, 60, 240), (0, 225, 235)
        put('TASK: ' + status.get('task', ''), 24, green)
        state = 'REC' if status.get('recording') else 'PAUSED'
        put(
            '%s | teleop=%s | BACK=%d' % (state, status.get('mode', '?'), bool(status.get('back'))),
            54,
            red if status.get('recording') else yellow,
            0.7,
        )
        put(
            'episode #%d | buffered %d | %.1f fps | %.1f s'
            % (
                status.get('episode_index', 0),
                status.get('buffered_frames', 0),
                status.get('fps', 0),
                status.get('elapsed', 0),
            ),
            84,
        )
        put(
            'SAVED: %d episodes / %d frames'
            % (status.get('saved_episodes', 0), status.get('saved_frames', 0)),
            113,
            green,
        )
        put('SUBTASK: ' + status.get('skill_text', ''), 151, yellow, 0.85)
        put(status.get('message', ''), 185, yellow)
        put('Controller: ' + status.get('reason', ''), 214)
        put('c start/resume | SPACE pause | s save | r discard', 254)
        put('1..9 subtask (paused) | e back-event | t teleop (paused)', 282)
        put('q/ESC exit (unsaved buffer discarded)', 310)
        cv2.imshow(self.window_name, np.concatenate([strip, panel], axis=0))
        key = cv2.waitKey(1) & 0xFF
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return 'q'
        except Exception:
            return 'q'
        return 'q' if key == 27 else chr(key).lower() if key < 128 else None

    def stop(self):
        if self.enabled:
            self.cv2.destroyWindow(self.window_name)
            self.cv2.waitKey(1)
            self.enabled = False
