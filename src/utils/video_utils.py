# Video Utilities for reading, writing, and resizing frames
from dataclasses import dataclass
import os
import cv2
import numpy as np


# Read-only container holding video resolution, fps, and duration
@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float

    # Returns formatted video info as a readable string
    def __str__(self):
        return (
            f"{self.width}x{self.height} @ {self.fps:.1f}fps | "
            f"{self.total_frames} frames | {self.duration_seconds:.1f}s"
        )


# Reads video files or camera streams frame by frame
class VideoReader:

    # Stores video source path or camera index
    def __init__(self, source):
        self._source = source
        self._cap = None
        self._metadata = None

    # Opens video capture when entering with block
    def __enter__(self):
        self._open()
        return self

    # Releases video capture when exiting with block
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    # Provides read-only access to extracted video metadata
    @property
    def metadata(self):
        if self._metadata is None:
            raise RuntimeError("Reader not open — use with VideoReader(...) as reader:")
        return self._metadata

    # Generates frame index and frame pairs sequentially
    def __iter__(self):
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Reader not open — use with VideoReader(...) as reader:")

        frame_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame_idx, frame
            frame_idx += 1

    # Closes and cleans up the OpenCV video capture object
    def release(self):
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None

    # Verifies source exists and loads video properties
    def _open(self):
        if isinstance(self._source, str) and not os.path.isfile(self._source):
            raise FileNotFoundError(f"Video not found: {self._source}")

        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self._source}")

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = max(int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        duration = total / fps if fps > 0 and total > 0 else 0.0

        self._metadata = VideoMetadata(
            width=w,
            height=h,
            fps=fps,
            total_frames=total,
            duration_seconds=round(duration, 2),
        )


# List of video codecs to try in order of preference
_CODEC_FALLBACKS = ["avc1", "h264", "x264", "mp4v"]


# Saves video frames into an output mp4 video file
class VideoWriter:

    # Initializes output path, frame rate, and target dimensions
    def __init__(self, output_path, metadata, width=None, height=None):
        self._output_path = output_path
        self._fps = metadata.fps
        self._width = width or metadata.width
        self._height = height or metadata.height
        self._writer = None

    # Creates and opens output video file when entering with block
    def __enter__(self):
        self._open()
        return self

    # Flushes and closes output file when exiting with block
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    # Writes a single frame to output video, resizing if needed
    def write(self, frame):
        if self._writer is None:
            raise RuntimeError("Writer not open — use with VideoWriter(...) as writer:")

        h, w = frame.shape[:2]
        if w != self._width or h != self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        self._writer.write(frame)

    # Closes and releases the OpenCV video writer object
    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # Finds a working codec and initializes the OpenCV video writer
    def _open(self):
        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)

        for codec in _CODEC_FALLBACKS:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                self._output_path, fourcc, self._fps,
                (self._width, self._height),
            )
            if writer.isOpened():
                self._writer = writer
                return
            writer.release()

        raise IOError(
            f"Cannot create video '{self._output_path}' — "
            f"no working codec found from {_CODEC_FALLBACKS}"
        )


# Resizes an image while preserving original aspect ratio by default
def resize_frame(frame, width=None, height=None, keep_aspect_ratio=True):
    if width is None and height is None:
        raise ValueError("Provide at least 'width' or 'height'.")

    h, w = frame.shape[:2]

    if keep_aspect_ratio:
        if width is not None and height is not None:
            new_w, new_h = width, height
        elif width is not None:
            ratio = width / w
            new_w, new_h = width, int(h * ratio)
        else:
            ratio = height / h
            new_w, new_h = int(w * ratio), height
    else:
        new_w = width or w
        new_h = height or h

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
