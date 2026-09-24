from interactive.engine_api import PHYSICAL_NULL_STEER, PHYSICAL_NULL_THROTTLE
from interactive.external_models import ours_panel32


def test_action_geometry_and_physical_noop():
    assert ours_panel32.ACTIONS["F"] == (0.5, 0.0)
    assert ours_panel32.ACTIONS["R"] == (0.0, 0.5)
    assert ours_panel32.ACTIONS["L"] == (0.0, -0.5)
    assert ours_panel32.ACTIONS["N"] == (
        PHYSICAL_NULL_THROTTLE,
        PHYSICAL_NULL_STEER,
    )
    assert ours_panel32.ACTIONS["N"] != (0.0, 0.0)


def test_prompt_is_exact_shared_neutral_prompt():
    assert ours_panel32.PROMPT == "A first-person view of an outdoor environment."


def test_context_seed_is_stable_action_independent_and_context_specific():
    first = ours_panel32.context_seed("ego4d-bristol-001")
    assert first == ours_panel32.context_seed("ego4d-bristol-001")
    assert first != ours_panel32.context_seed("sekai-london-001")
