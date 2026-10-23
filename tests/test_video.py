"""Raw decoding and the existing RGB reader use the same decoder settings."""

import av
import numpy as np
import pytest

from reveil.video import decoded_frames, read_frames


@pytest.mark.parametrize("threads", [None, 3])
@pytest.mark.parametrize("size", [None, (32, 24)])
def test_decoded_frames_and_rgb_reader_match_original_decode(make_video, threads, size):
    path = make_video(frames=8, width=64, height=48)
    scaled = {} if size is None else {"width": size[0], "height": size[1], "interpolation": "AREA"}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if threads is not None:
            stream.codec_context.thread_count = threads
        expected = [frame.to_ndarray(format="rgb24", **scaled) for frame in container.decode(stream)]
    raw = list(decoded_frames(path, threads=threads))
    assert len(raw) == len(expected) == 8
    assert all(isinstance(frame, av.VideoFrame) for frame in raw)
    for frame, rgb, original in zip(raw, read_frames(path, size=size, threads=threads), expected):
        assert np.array_equal(frame.to_ndarray(format="rgb24", **scaled), original)
        assert np.array_equal(rgb, original)
