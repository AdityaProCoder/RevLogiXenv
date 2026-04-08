"""Tests for BaselineAgent response parsing robustness."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from autonomous_returns_v0 import DispositionAction, ReturnsAction
from autonomous_returns_v0.baseline import BaselineAgent, BaselineRunConfig


class MockObservation:
    """Minimal observation for testing parse_action."""
    def __init__(self, has_item: bool = True):
        self.current_item = MagicMock() if has_item else None


@pytest.fixture
def agent():
    """Create agent with google provider (no API key needed at init)."""
    with patch.dict(os.environ, {"VERTEX_PROJECT_ID": "test-project"}):
        return BaselineAgent(BaselineRunConfig(provider="google", model="gemini-2.0-flash"))


class TestParseActionRobustness:
    """Test that parse_action handles edge cases gracefully."""

    def test_parse_action_exact_match(self, agent):
        """Exact lowercase match returns correct action."""
        obs = MockObservation(has_item=True)
        result = agent.parse_action("resell_full", obs)
        assert result.action == DispositionAction.RESELL_FULL

    def test_parse_action_case_insensitive(self, agent):
        """Case insensitive match works."""
        obs = MockObservation(has_item=True)
        result = agent.parse_action("RESELL_FULL", obs)
        assert result.action == DispositionAction.RESELL_FULL

    def test_parse_action_contains_match(self, agent):
        """Partial match in response returns correct action."""
        obs = MockObservation(has_item=True)
        result = agent.parse_action("I recommend resell_full for this item", obs)
        assert result.action == DispositionAction.RESELL_FULL

    def test_parse_action_empty_string_fallback(self, agent):
        """Empty string returns safe default RESELL_DISCOUNT_30."""
        obs = MockObservation(has_item=True)
        result = agent.parse_action("", obs)
        assert result.action == DispositionAction.RESELL_DISCOUNT_30

    def test_parse_action_none_returns_fallback(self, agent):
        """None input returns safe default."""
        obs = MockObservation(has_item=True)
        result = agent.parse_action(None, obs)
        assert result.action == DispositionAction.RESELL_DISCOUNT_30

    def test_parse_action_garbage_text_fallback(self, agent):
        """Random garbage text returns safe default."""
        obs = MockObservation(has_item=True)
        result = agent.parse_action("asdfghjkl qwerty", obs)
        assert result.action == DispositionAction.RESELL_DISCOUNT_30

    def test_parse_action_no_current_item_returns_wait(self, agent):
        """When no current item, always returns WAIT."""
        obs = MockObservation(has_item=False)
        result = agent.parse_action("anything", obs)
        assert result.action == DispositionAction.WAIT