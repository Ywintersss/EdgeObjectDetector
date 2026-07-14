"""Where annotated frames go. A sink, not a mode -- the loop produces one frame and hands
it to every enabled sink, so --display both is not a special case.

MjpegSink is the one that MUST work: it is a socket, it needs no display server, and it
works while the camera is mounted overhead and the board is headless on Wi-Fi.

HdmiSink is best-effort ON PURPOSE. Mendel runs a Wayland compositor and OpenCV's imshow
is built against GTK/X11, so the window may simply refuse to open. Losing the window must
never lose the run.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"


class MjpegSink:
    """Serves the latest published frame as multipart/x-mixed-replace on :port."""

    def __init__(self, port: int = 8080):
        self._frame = None
        self._lock = threading.Lock()
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.end_headers()
                try:
                    while True:
                        jpeg = sink._latest_jpeg()
                        if jpeg is None:
                            continue
                        self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass        # the browser tab closed; that is not an error

            def log_message(self, *args):
                pass            # do not spam the console with one line per frame

        self._server = ThreadingHTTPServer(("", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://0.0.0.0:{self.port}/"

    def _latest_jpeg(self):
        with self._lock:
            return self._frame

    def publish(self, frame) -> None:
        # Encode once, here, rather than once per connected client. On a Cortex-A35 this
        # encode is a real cost -- it is one of the five stages the runner times.
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed to encode the annotated frame")
        with self._lock:
            self._frame = buf.tobytes()

    def close(self) -> None:
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
