import sys
import time
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
            # Ask for less than one frame's total wire size (part headers + JPEG bytes),
            # not more: io.BufferedReader.read(n) blocks until it fills n bytes or hits
            # EOF, and now that the handler correctly sends exactly one frame and then
            # blocks for the next, there is no further data to pad a larger read with.
            chunk = response.read(512)
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


def test_client_connected_before_any_frame_does_not_spin_the_cpu():
    # A client connecting before the first publish() must block quietly on the
    # condition, not busy-loop on `if jpeg is None: continue`. process_time() is
    # process CPU time (all threads), so a spinning handler shows up as cpu ~= wall;
    # a properly blocked handler shows up as cpu ~= 0 regardless of wall time.
    sink = S.MjpegSink(port=0)
    try:
        conn = urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/", timeout=5)
        try:
            start_cpu = time.process_time()
            start_wall = time.perf_counter()
            time.sleep(0.4)
            cpu_used = time.process_time() - start_cpu
            wall = time.perf_counter() - start_wall
            assert cpu_used < 0.2 * wall, (
                f"handler burned {cpu_used:.3f}s CPU over a {wall:.3f}s wait -- "
                "it is spinning instead of blocking")
        finally:
            conn.close()
    finally:
        sink.close()


def test_handler_blocks_between_frames_instead_of_resending(frame):
    # Once a frame has been sent and the client is caught up, the handler must wait
    # for a NEW frame (a newer sequence number) instead of re-sending the identical
    # JPEG bytes as fast as the socket accepts writes.
    sink = S.MjpegSink(port=0)
    try:
        sink.publish(frame)
        conn = urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/", timeout=5)
        try:
            conn.read(512)    # drain the one frame that is already available (less than
                              # its total wire size -- see the note in the JPEG test above)
            # No further publish() calls -- if the handler is spinning, this window
            # will show cpu ~= wall; if it is blocked waiting for a new frame, ~= 0.
            start_cpu = time.process_time()
            start_wall = time.perf_counter()
            time.sleep(0.4)
            cpu_used = time.process_time() - start_cpu
            wall = time.perf_counter() - start_wall
            assert cpu_used < 0.2 * wall, (
                f"handler burned {cpu_used:.3f}s CPU over a {wall:.3f}s wait -- "
                "it is re-sending the same frame instead of blocking for a new one")
        finally:
            conn.close()
    finally:
        sink.close()
