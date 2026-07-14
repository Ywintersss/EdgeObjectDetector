import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coral"))
import sinks as S  # noqa: E402


@pytest.fixture
def frame():
    return np.full((120, 160, 3), 64, dtype=np.uint8)


def test_mjpeg_sink_serves_a_published_frame_as_a_jpeg(frame):
    sink = S.MjpegSink(port=0)          # port 0 -> OS picks a free one, so tests never clash
    try:
        sink.publish(frame)
        # Connect over the LOOPBACK address, not the 0.0.0.0 the sink advertises: 0.0.0.0
        # is a bind address, and connecting to it is not portable.
        with urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/", timeout=5) as response:
            assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
            chunk = response.read(4096)
        # JPEG SOI marker -- proves a real encoded image went out, not an empty stream.
        assert b"\xff\xd8" in chunk
    finally:
        sink.close()


def test_mjpeg_sink_url_names_the_bound_port(frame):
    sink = S.MjpegSink(port=0)
    try:
        assert sink.url.startswith("http://")
        assert str(sink.port) in sink.url
    finally:
        sink.close()


def test_hdmi_sink_degrades_instead_of_crashing_when_no_display(monkeypatch, frame):
    # Mendel runs Wayland; OpenCV's imshow is built against GTK/X11 and may refuse to open
    # a window. That must cost us the WINDOW, never the RUN -- the MJPEG stream is still
    # carrying the session.
    import cv2

    def explode(*args, **kwargs):
        raise cv2.error("no display")

    monkeypatch.setattr(cv2, "imshow", explode)
    sink = S.HdmiSink()
    sink.publish(frame)             # must not raise
    assert sink.available is False
    sink.publish(frame)             # still must not raise once disabled
    sink.close()


def test_build_sinks_both_returns_a_stream_and_a_display():
    built = S.build_sinks("both", port=0)
    try:
        kinds = {type(s).__name__ for s in built}
        assert kinds == {"MjpegSink", "HdmiSink"}
    finally:
        for s in built:
            s.close()


def test_build_sinks_rejects_an_unknown_display():
    with pytest.raises(ValueError, match="unknown display"):
        S.build_sinks("hologram", port=0)
