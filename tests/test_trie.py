"""
    Unit tests for the TokenTrie class in src.constraints.trie.
    # type: ignore[arg-type]
"""
import pytest
from src.constraints.trie import TokenTrie


class TestTokenTrie:
    """Unit tests for the TokenTrie class in src.constraints.trie."""
    @pytest.fixture
    def trie(self) -> TokenTrie:
        return TokenTrie()

    def test_insert_and_retrieve_normal(self, trie: TokenTrie) -> None:
        """
            Test normal insertion and retrieval of tokens.
            This test verifies that tokens can be inserted into the trie
            and retrieved correctly based on their clean and stripped strings.
            It checks that the token IDs are stored and retrieved as expected.
            Args:
                trie: An instance of TokenTrie provided by the fixture.
        """
        trie.insert(clean_str="hello", stripped_str="hello", token_id=100)
        trie.insert(clean_str="hell", stripped_str="hell", token_id=101)

        res = trie.get_token_with_prefix("he", use_stripped=True)
        assert res == {100, 101}

    def test_whitespace_token_handling(self, trie: TokenTrie) -> None:
        """
            Test that whitespace tokens are correctly identified and stored.
            This test checks that when tokens representing whitespace
            (e.g., "\n" and "   ") are inserted into the trie, their token IDs
            are correctly stored. It verifies that these token IDs are included
            in the set of whitespace token IDs maintained by the trie.
            Args:
                trie: An instance of TokenTrie provided by the fixture.
        """
        trie.insert(clean_str="\n", stripped_str="", token_id=198)
        trie.insert(clean_str="   ", stripped_str="", token_id=220)

        assert 198 in trie.whitespace_token_ids
        assert 220 in trie.whitespace_token_ids

    def test_invalid_type_insertion(self, trie: TokenTrie) -> None:
        """
            Test that inserting a token with invalid types raises a TypeError.
            This test verifies that if the insert method is called with None
            for the clean_str, stripped_str, or token_id parameters,
            a TypeError is raised. This ensures that the trie enforces type
            constraints and does not allow invalid data to be inserted,
            which could lead to unexpected behavior or errors during token
            retrieval.
            Args:
                trie: An instance of TokenTrie provided by the fixture.
        """
        with pytest.raises(TypeError):
            trie.insert(clean_str=None, stripped_str=None, token_id=999)
