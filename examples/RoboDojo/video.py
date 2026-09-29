"""Bounded PyAV frame reads for long-running multiworker RoboDojo training."""
from contextlib import ExitStack
import gc
import logging
import math
import time

import av
import numpy as np

logger = logging.getLogger(__name__)


def _decode_rgb_frame(path, timestamp):
    with ExitStack() as resources:
        container = resources.enter_context(av.open(str(path)))
        stream = container.streams.video[0]
        codec = stream.codec_context
        # libdav1d otherwise starts 192 threads per open on the training host.
        codec.thread_count = 1
        codec.thread_type = "SLICE"
        # PyAV 12 retains stream/container cycles even after container.close().
        # Closing the codec explicitly releases its native buffers and workers
        # immediately, without waiting for Python's cyclic garbage collector.
        if hasattr(codec, "close"):
            resources.callback(codec.close, strict=False)
        time_base = float(stream.time_base)
        rate = stream.average_rate or stream.guessed_rate
        tolerance = 0.5 / float(rate) + 0.002 if rate else 0.022
        container.seek(int(timestamp / time_base), stream=stream, backward=True, any_frame=False)
        frames = container.decode(video=0)
        resources.callback(frames.close)
        closest, difference = None, float("inf")
        for frame in frames:
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * time_base)
            current = abs(frame_time - timestamp)
            if current < difference:
                closest, difference = frame, current
            if frame_time > timestamp and current > difference:
                break
            if frame_time > timestamp + 1.0:
                break
        if closest is None or difference > tolerance:
            raise ValueError(f"No aligned video frame: path={path}, timestamp={timestamp}, "
                             f"nearest_error={difference}, tolerance={tolerance}")
        # Own the pixel buffer independently of the decoder we are closing.
        return np.array(closest.to_ndarray(format="rgb24"), copy=True)


def read_rgb_frame(path, timestamp):
    """Read the same nearest RGB frame; retry allocation failures only."""
    timestamp = float(timestamp)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"Invalid video timestamp: {timestamp}")
    for attempt in range(3):
        try:
            return _decode_rgb_frame(path, timestamp)
        except av.error.MemoryError:
            if attempt == 2:
                logger.exception("Video allocation failed: path=%s timestamp=%s", path, timestamp)
                raise
            logger.warning("Retrying video allocation: path=%s timestamp=%s attempt=%s",
                           path, timestamp, attempt + 1)
            gc.collect()
            time.sleep(0.1 * (attempt + 1))
