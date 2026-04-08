from __future__ import annotations

from autonomous_returns_v0 import (
    AutonomousReturnsEnv,
    DispositionAction,
    HeuristicPolicy,
    ReturnsAction,
)


def test_heuristic_does_not_flag_perfect_items_as_fraud():
    """Perfect items (score ~8.9) should NOT trigger fraud flags even with high score."""
    policy = HeuristicPolicy()
    env = AutonomousReturnsEnv("hard")
    obs = env.reset(seed=42)
    
    found_perfect = False
    while not obs.done and obs.current_item is not None:
        item = obs.current_item
        score = float(item.noisy_condition_score)
        hist_flags = int(item.customer_profile.historical_fraud_flags)
        recent_abuse = item.customer_profile.recent_abuse_signals
        
        if score >= 8.0 and hist_flags == 0 and not recent_abuse:
            action = policy(obs)
            assert action.action != DispositionAction.FLAG_FRAUD, (
                f"Perfect item (score={score}) incorrectly flagged as fraud"
            )
            found_perfect = True
            break
        obs = env.step(ReturnsAction(action=DispositionAction.WAIT))
    
    assert found_perfect, "No high-scoring clean item found"


def test_heuristic_inspect_thresholds():
    """INSPECT should trigger on ambiguous mid-range items, not extreme scores."""
    policy = HeuristicPolicy()
    
    class MockCustomerProfile:
        total_returns = 1
        historical_fraud_flags = 1
        recent_abuse_signals = False
    
    class MockItem:
        noisy_condition_score = 6.5
        price = 400.0
        stated_return_reason = "not as expected"
        customer_profile = MockCustomerProfile()
        damage_flags = []
        packaging_condition = "opened"
        inspection_note = None
    
    class MockObs:
        current_item = MockItem()
        remaining_items = 5
        pending_resolution_count = 0
        running_ledger = 0.0
        last_three_resolutions = []
    
    action = policy(MockObs())
    assert action.action == DispositionAction.INSPECT, (
        f"Ambiguous item with fraud history should INSPECT, got {action.action}"
    )


def test_heuristic_fraud_flag_requires_multiple_signals():
    """Fraud flagging should require >= 3 weak signals."""
    policy = HeuristicPolicy()
    
    class MockCustomerProfile:
        total_returns = 2
        historical_fraud_flags = 0
        recent_abuse_signals = False
    
    class MockItem:
        noisy_condition_score = 6.0
        price = 100.0
        stated_return_reason = "no longer needed"
        customer_profile = MockCustomerProfile()
        damage_flags = []
        packaging_condition = "good"
        inspection_note = None
    
    class MockObs:
        current_item = MockItem()
        remaining_items = 5
        pending_resolution_count = 0
        running_ledger = 0.0
        last_three_resolutions = []
    
    action = policy(MockObs())
    assert action.action != DispositionAction.FLAG_FRAUD


def test_policy_config_has_no_fraud_score_threshold():
    """PolicyConfig should not have fraud_score_threshold (it was inverted)."""
    from autonomous_returns_v0 import PolicyConfig
    config = PolicyConfig()
    assert not hasattr(config, 'fraud_score_threshold'), (
        "fraud_score_threshold should be removed - it causes false positives"
    )


def test_policy_config_inspect_thresholds():
    """PolicyConfig should have explicit INSPECT thresholds."""
    from autonomous_returns_v0 import PolicyConfig
    config = PolicyConfig()
    assert hasattr(config, 'inspect_score_low')
    assert hasattr(config, 'inspect_score_high')
    assert hasattr(config, 'inspect_min_price')
    assert config.inspect_score_low == 5.0
    assert config.inspect_score_high == 7.5
    assert config.inspect_min_price == 250.0
