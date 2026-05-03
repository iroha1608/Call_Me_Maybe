
class TrieNode:
    """
        Trie木を構成する単一ノード。双方向Trie木。
        空間計算量(メモリ) -> 時間計算量をO(1)~O(L)に最適化。
        接頭辞としてこのノードを通過する全token_idを保持、
        実行時の深さ優先探索(DFS)のオーバーヘッドを排除する。
    """
    __slots__ = ["children", "token_ids"]

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.token_ids: list[int] = []


class TokenTrie:
    """
        自己回帰生成における語彙走査を圧縮するデータ構造。
    """
    def __init__(self) -> None:
        # clean_strのroot node
        self.clean_root = TrieNode()
        # stripped_strのroot node
        self.stripped_root = TrieNode()
        # 構文の高速評価のため、先頭文字によるハッシュマップを使用。
        self.first_char_index: dict[str, list[int]] = {}
        # すべての空白tokenリスト
        self.whitespace_token_ids: list[int] = []

    def insert(
        self, clean_str: str, stripped_str: str, token_id: int
    ) -> None:
        """
            ConstraintFilterの初期化時にループで実行。ツリーの構築。
        """
        # clean_str = 全語彙を.replace("Ġ", " ")。
        # stripped_str = 前語彙を.lstrip()。
        # 空白をstrip()した文字だけ無い文字=スペースのtoken。
        if clean_str and not stripped_str:
            self.whitespace_token_ids.append(token_id)
        elif stripped_str:
            # 文字列のprefixの文字が辞書に無い時。
            first_char = stripped_str[0]
            if first_char not in self.first_char_index:
                # prefixの文字から始まるtoken_idのリストを作成。
                self.first_char_index[first_char] = []
            self.first_char_index[first_char].append(token_id)

        # clean_str用のツリー構築(文字列内部での完全一致用)
        if clean_str:
            node = self.clean_root
            for char in clean_str:
                # もし文字がroot nodeの中にない時
                if char not in node.children:
                    # prefixTrieNodeを複製
                    node.children[char] = TrieNode()
                node = node.children[char]
                # 
                node.token_ids.append(token_id)

        # stripped_str用のツリー構築(key、valueの開始時の前方一致用)
        if stripped_str:
            node = self.stripped_root
            for char in stripped_str:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.token_ids.append(token_id)

    def get_token_with_prefix(
        self, prefix: str, use_stripped: bool = False
    ) -> list[int]:
        """今使ってません"""
        if not prefix:
            return []

        node = self.stripped_root if use_stripped else self.clean_root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.token_ids

