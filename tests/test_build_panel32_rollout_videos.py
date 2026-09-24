import pytest

from grids.eval.build_panel32_rollout_videos import (
    BuildError,
    sekai_original_excerpt_start,
)


def test_sekai_excerpt_start_accepts_legacy_note():
    context = {
        "selection_note": "only 1264.300000--1270.300000 s downloaded.",
        "source_uri": "https://www.youtube.com/watch?v=abc#t=1.0",
    }
    assert sekai_original_excerpt_start(context) == 1264.3


def test_sekai_excerpt_start_accepts_locked_url_fragment():
    context = {
        "selection_note": "only the 15-second review range was downloaded.",
        "source_uri": "https://www.youtube.com/watch?v=abc#t=2802.900000",
    }
    assert sekai_original_excerpt_start(context) == 2802.9


def test_sekai_excerpt_start_fails_closed_without_timestamp():
    with pytest.raises(BuildError, match="selection_note or source_uri"):
        sekai_original_excerpt_start(
            {"selection_note": "review range", "source_uri": "https://x/y"}
        )
