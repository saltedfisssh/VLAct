"""Real-codec frame alignment, deterministic cleanup, and bounded retries."""
import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from examples.RoboDojo.video import read_rgb_frame


class VideoReaderTest(unittest.TestCase):
    def test_real_video_alignment_and_cleanup_without_gc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "colors.mkv"
            expected = []
            with av.open(str(path), mode="w") as output:
                stream = output.add_stream("ffv1", rate=25)
                stream.width, stream.height, stream.pix_fmt = 32, 32, "bgr0"
                for i in range(8):
                    rgb = np.full((32, 32, 3), [i * 20, 60, 255 - i * 20], dtype=np.uint8)
                    expected.append(rgb)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            gc.collect()
            threads = len(list(Path("/proc/self/task").iterdir()))
            descriptors = len(list(Path("/proc/self/fd").iterdir()))
            was_enabled = gc.isenabled()
            gc.disable()
            try:
                for i in range(64):
                    index = (i * 3) % 8
                    actual = read_rgb_frame(path, index / 25 + 0.001)
                    np.testing.assert_array_equal(actual, expected[index])
                self.assertLessEqual(len(list(Path("/proc/self/task").iterdir())), threads + 1)
                self.assertLessEqual(len(list(Path("/proc/self/fd").iterdir())), descriptors + 1)
            finally:
                if was_enabled:
                    gc.enable()
                gc.collect()
            with self.assertRaisesRegex(ValueError, "No aligned"):
                read_rgb_frame(path, 100.0)

    def test_allocation_retry_preserves_request_and_is_bounded(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        failure = av.error.MemoryError(12, "Cannot allocate memory")
        with patch("examples.RoboDojo.video.time.sleep"), patch(
            "examples.RoboDojo.video._decode_rgb_frame", side_effect=[failure, frame]
        ) as decode:
            self.assertIs(read_rgb_frame("same.mp4", 1.25), frame)
            self.assertEqual(decode.call_args_list[0], decode.call_args_list[1])
        with patch("examples.RoboDojo.video.time.sleep"), patch(
            "examples.RoboDojo.video._decode_rgb_frame", side_effect=failure
        ) as decode:
            with self.assertRaises(av.error.MemoryError):
                read_rgb_frame("same.mp4", 1.25)
            self.assertEqual(decode.call_count, 3)
        with patch("examples.RoboDojo.video._decode_rgb_frame", side_effect=ValueError("corrupt")) as decode:
            with self.assertRaisesRegex(ValueError, "corrupt"):
                read_rgb_frame("same.mp4", 1.25)
            self.assertEqual(decode.call_count, 1)


if __name__ == "__main__":
    unittest.main()
