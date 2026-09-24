from code_release.baselines.dreamx_runner import context_seed


def test_context_seed_is_stable_and_action_independent():
    assert context_seed("frodobots-u31") == context_seed("frodobots-u31")
    assert context_seed("frodobots-u31") != context_seed("ego4d-bristol-01")
    assert 0 <= context_seed("sekai-0bhwadzjekq") < 2**31 - 1
