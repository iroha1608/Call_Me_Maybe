"""
    Unit tests for the JSONStateTracker class, which is responsible for
    tracking the state of a JSON document during incremental parsing.
"""
import pytest
from src.constraints.state import JSONStateTracker, FSMState


class TestJSONTracker:
    """Unit tests for the JSONStateTracker class."""

    @pytest.fixture
    def tracker(self) -> JSONStateTracker:
        """
            Fixture to create a new instance of JSONStateTracker for each test.
            Returns:
                A new instance of JSONStateTracker.
        """
        return JSONStateTracker()

    def test_determine_current_state_incremental(
        self, tracker: JSONStateTracker
    ) -> None:
        """
            Test the determine_current_state method
            with incremental JSON input.
            This test simulates the incremental parsing of a JSON document by
            providing partial JSON strings and verifying that the state tracker
            correctly identifies the current state, whether it's in a string,
            and the last key encountered.
            Args:
                tracker:
                    An instance of JSONStateTracker provided by the fixture.
        """
        ctx1 = tracker.determine_current_state('{\n  "name"')
        assert ctx1.state == FSMState.COLON
        assert ctx1.in_string is False
        assert ctx1.last_key == "name"

        ctx2 = tracker.determine_current_state('{\n  "name": ')
        assert ctx2.state == FSMState.VALUE
        assert ctx2.is_value_context is True

    def test_get_allowed_characters_no_array(
        self, tracker: JSONStateTracker
    ) -> None:
        """
            Test the get_allowed_characters method to ensure that array
            characters are not allowed in value contexts.
            This test verifies that when the state tracker is in
            a value context, the allowed characters do not include array
            delimiters ("[" and "]"). This is important to ensure
            that the state tracker correctly identifies
            the context and restricts the allowed characters accordingly.
            Args:
                tracker:
                    An instance of JSONStateTracker provided by the fixture.
        """
        allowed_depth_1 = (
            tracker.get_allowed_characters(FSMState.VALUE, depth=1))
        allowed_depth_2 = (
            tracker.get_allowed_characters(FSMState.VALUE, depth=2))

        assert "[" not in allowed_depth_1
        assert "]" not in allowed_depth_1
        assert "[" not in allowed_depth_2
        assert "]" not in allowed_depth_2

    def test_empty_string_reset(self, tracker: JSONStateTracker) -> None:
        """
            Test that providing an empty string to determine_current_state
            resets the state tracker to the initial state.
            This test simulates the scenario where the input string is cleared,
            which should reset the state tracker to the initial state (BEGIN)
            with a depth of 0. This ensures that the state tracker can handle
            resets correctly and does not retain any previous state information
            when the input is cleared.
            Args:
                tracker:
                    An instance of JSONStateTracker provided by the fixture.
        """
        tracker.determine_current_state('{"test": ')
        ctx = tracker.determine_current_state("")
        assert ctx.state == FSMState.BEGIN
        assert ctx.depth == 0
