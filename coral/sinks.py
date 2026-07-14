"""Where annotated frames go. A sink, not a mode -- the loop produces one frame and hands
it to every enabled sink, so --display both is not a special case.

MjpegSink is the one that MUST work: it is a socket, it needs no display server, and it
works while the camera is mounted overhead and the board is headless on Wi-Fi.

HdmiSink is best-effort ON PURPOSE. Mendel runs a Wayland compositor and OpenCV's imshow
is built against GTK/X11, so the window may simply refuse to open. Losing the window must
never lose the run.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"


class MjpegSink:
    """Serves the latest published frame as multipart/x-mixed-replace on :port."""

    def __init__(self, port: int = 8080):
        self._frame = None
        # Monotonic counter bumped on every publish(). Handlers remember the seq of the
        # frame they last sent and block until a NEWER one shows up -- that is what turns
        # "wait for a frame" and "wait for a DIFFERENT frame" into the same wait.
        self._frame_seq = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closing = False
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.end_headers()
                last_seq = 0
                try:
                    while True:
                        result = sink._wait_for_new_frame(last_seq)
                        if result is None:
                            break   # sink is closing; disconnect quietly
                        jpeg, last_seq = result
                        self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except OSError:
                    # the browser tab closed; that is not an error. OSError is the common
                    # base for every socket-teardown flavor we see here (BrokenPipeError,
                    # ConnectionResetError, ConnectionAbortedError, ...).
                    pass

            def log_message(self, *args):
                pass            # do not spam the console with one line per frame

        self._server = ThreadingHTTPServer(("", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _wait_for_new_frame(self, last_seq: int):
        """Block until a frame newer than `last_seq` is published, or the sink closes.

        Returns (jpeg_bytes, new_seq), or None if the sink is closing. The bounded
        `wait(timeout=...)` exists only so a handler notices shutdown promptly even if
        it somehow missed the close() notify -- it is not the polling mechanism; the
        Condition wait is, so this blocks instead of spinning a core.
        """
        with self._condition:
            while not self._closing and self._frame_seq <= last_seq:
                self._condition.wait(timeout=0.5)
            if self._closing:
                return None
            return self._frame, self._frame_seq

    def publish(self, frame) -> None:
        # Encode once, here, rather than once per connected client. On a Cortex-A35 this
        # encode is a real cost -- it is one of the six stages the runner times.
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode the annotated frame")
        with self._condition:
            self._frame = buf.tobytes()
            self._frame_seq += 1
            self._condition.notify_all()

    def close(self) -> None:
        # Wake every handler blocked in _wait_for_new_frame before tearing the server
        # down, so none of them can be left waiting on the condition forever.
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()


class HdmiSink:
    """cv2.imshow, if the board will give us a window. Disables itself if not."""

    WINDOW = "EdgeObjectDetector - coarse-17 on Edge TPU"

    def __init__(self):
        self.available = True
        self._quit = False

    def publish(self, frame) -> None:
        if not self.available:
            return
        try:
            cv2.imshow(self.WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._quit = True
        except cv2.error as exc:
            # Say it ONCE, then stop trying. Not silent -- but not fatal either.
            print(f"WARNING: HDMI display unavailable ({exc}); continuing with the stream "
                  f"only. This is expected on Mendel's Wayland compositor.")
            self.available = False

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        if self.available:
            cv2.destroyAllWindows()


def build_sinks(display: str, port: int) -> list:
    """display in {stream, hdmi, both} -> the sinks to feed."""
    if display == "stream":
        return [MjpegSink(port)]
    if display == "hdmi":
        return [HdmiSink()]
    if display == "both":
        return [MjpegSink(port), HdmiSink()]
    raise ValueError(f"unknown display {display!r}; expected stream, hdmi or both")
