import sys
import math
import re
from enum import Enum
from typing import Any

from src.tokenizer import Tokenizer
from src.models import FunctionDefinition
from src.trie import TrieNode, TokenTrie


class FSMState(str, Enum):
    """Json解析用オートマトンの状態を表す。"""
    EXPECT_BEGIN_OBJECT = "EXPECT_BEGIN_OBJECT"
    EXPECT_KEY = "EXPECT_KEY"
    EXPECT_COLON = "EXPECT_COLON"
    EXPECT_VALUE = "EXPECT_VALUE"
    EXPECT_COMMA_OR_END_OBJECT = "EXPECT_COMMA_OR_END_OBJECT"
    DONE = "DONE"


def _get_attr(obj: Any, key: str, default: Any = "") -> Any:
    """dict object避け"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ConstraintFilter:
    """
    プッシュダウンオートマトン(PDA)
    制約付きデコーディングを行う。
    LLMが出力した確率分布(Logits)を操作、
    Json構文、スキーマに違反するtokenをフィルタリングするクラス。
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        available_functions: list[FunctionDefinition]
    ) -> None:
        self._tokenizer = tokenizer
        self._available_functions = available_functions

        # ループ中、辞書展開のオーバーヘッド回避で辞書内の項目をキャッシュ
        self._vocab_items: list[tuple[int, str]] = (
            list(tokenizer._id_to_token.items()))
        # 許可するroot key
        self._allowed_root_keys: list[str] = ["name", "parameters"]
        # 使用可能関数から名前の一覧を取得
        self._valid_fn_names: list[str] = [
            _get_attr(fn, "name", "") for fn in self._available_functions
        ]
        self._target_fn_name: str = ""

        # Trie木の初期化と全語彙の登録(空間計算量 O(V * 平均長))、一度のみ実行
        self._trie = TokenTrie()
        # 語彙からid, strをループ
        for t_id, t_str in self._vocab_items:
            clean_str = t_str.replace("Ġ", " ")
            stripped_str = clean_str.lstrip()
            self._trie.insert(clean_str, stripped_str, t_id)

        self._cached_text = ""
        self._cached_loop_state: tuple[
            int, bool, bool, str, bool, str, str, frozenset[Any]
        ] = (
            0, False, False, "", False, "", "", frozenset()
        )
