"""
Trie implementation for efficient token matching in the Call Me Maybe project.
"""


class TrieNode:
    """
        A node in the Trie data structure used for storing token strings.
        Each node contains a dictionary of child nodes and a set of token IDs
        that correspond to the tokens
        that can be formed from the path to this node.
    """
    # __slots__を定義->__dict__での属性管理がされない->メモリの節約
    __slots__ = ["children", "token_ids"]

    def __init__(self) -> None:
        self.children: dict[str, 'TrieNode'] = {}
        # このノードで終わる、あるいは通過するtoken_idの集合
        self.token_ids: set[int] = set()


class TokenTrie:
    """
        Trie structure to store and match token strings for both clean
        and stripped forms.
    """
    def __init__(self) -> None:
        """
            Initializes the TokenTrie with separate root nodes for clean and
            stripped strings, a hashmap for first character indexing, and a
            set for whitespace token IDs.
        """
        # clean_strのroot node
        self.clean_root = TrieNode()
        # stripped_strのroot node
        self.stripped_root = TrieNode()
        # 構文の高速評価のため、先頭文字によるハッシュマップを使用。
        self.first_char_index: dict[str, set[int]] = {}
        # すべての空白tokenリスト
        self.whitespace_token_ids: set[int] = set()

    def insert(
        self, clean_str: str, stripped_str: str, token_id: int
    ) -> None:
        """
            Inserts a token into the Trie based on its clean and stripped
            string forms.
            Args:
                clean_str (str): The clean string representation of the token.
                stripped_str (str):
                    The stripped string representation of the token.
                token_id (int): The unique identifier for the token.
            Raises:
                ValueError:
                    If either clean_str or stripped_str is not a string,
                    or if both are empty strings.
        """
        if not isinstance(clean_str, str) or not isinstance(stripped_str, str):
            raise (
                f"Invalid token string type for ID {token_id}. Must be string."
            )
        if not clean_str and not stripped_str:
            return

        if clean_str and not stripped_str:
            self.whitespace_token_ids.add(token_id)

        elif stripped_str:
            first_char = stripped_str[0]
            if first_char not in self.first_char_index:
                self.first_char_index[first_char] = set()
            self.first_char_index[first_char].add(token_id)

        # clean_str用のツリー構築(文字列内部での完全一致用)
        if clean_str:
            node = self.clean_root
            for char in clean_str:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.token_ids.add(token_id)

        # stripped_str用のツリー構築(key、valueの開始時の前方一致用)
        if stripped_str:
            node = self.stripped_root
            for char in stripped_str:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.token_ids.add(token_id)

    def get_token_with_prefix(
        self, prefix: str, use_stripped: bool = False
    ) -> set[int]:
        """
            Retrieves the set of token IDs that match a given prefix string.
            Args:
                prefix (str): The prefix string to match against the Trie.
                use_stripped (bool):
                    Whether to use the stripped Trie for matching.
            Returns:
                set[int]: A set of token IDs that match the given prefix.
        """
        if not prefix or not isinstance(prefix, str):
            return set()

        node = self.stripped_root if use_stripped else self.clean_root
        for char in prefix:
            if char not in node.children:
                return set()
            node = node.children[char]
        return node.token_ids
