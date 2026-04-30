import sys
import math
import re
from enum import Enum
from typing import Any

from src.tokenizer import Tokenizer
from src.models import FunctionDefinition


class FSMState(str, Enum):
    """Json解析用オートマトンの状態を表す。"""
    EXPECT_BEGIN_OBJECT = "EXPECT_BEGIN_OBJECT"
    EXPECT_KEY = "EXPECT_KEY"
    EXPECT_COLON = "EXPECT_COLON"
    EXPECT_VALUE = "EXPECT_VALUE"
    EXPECT_COMMA_OR_END_OBJECT = "EXPECT_COMMA_OR_END_OBJECT"
    DONE = "DONE"
    ERROR = "ERROR"


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
        自己回帰生成における語彙走査を O(V) -> O(L)に圧縮するデータ構造。
        Lは計算する文字列の長さ。
    """
    def __init__(self) -> None:
        self.clean_root = TrieNode()
        self.stripped_root = TrieNode()
        # 構文の高速評価のため、先頭文字による0(1)ハッシュマップを使用。
        self.first_char_index: dict[str, list[int]] = {}
        self.whitespace_token_ids: list[int] = []

    def insert(
        self, clean_str: str, stripped_str: str, token_id: int
    ) -> None:
        """
            ConstraintFilterの初期化時に一度だけ実行。ツリーの構築。
        """
        if not stripped_str and clean_str:
            self.whitespace_token_ids.append(token_id)
        elif stripped_str:
            first_char = stripped_str[0]
            if first_char not in self.first_char_index:
                self.first_char_index[first_char] = []
            self.first_char_index[first_char].append(token_id)

        # clean_str用のツリー構築(文字列内部での完全一致用)
        if clean_str:
            node = self.clean_root
            for char in clean_str:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
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
        if not prefix:
            return []

        node = self.stripped_root if use_stripped else self.clean_root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.token_ids


def _get_attr(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ConstraintFilter:
    """
    プッシュダウンオートマトン(PDA)、前方一致を使用し、
    制約付きデコーディングを行う。
    言語モデルが出力した確率分布(Logits)を操作、
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
        self._vocab_items = list(tokenizer._id_to_token.items())
        self._allowed_root_keys = ["name", "parameters"]
        self._valid_fn_names = [
            _get_attr(fn, "name", "") for fn in self._available_functions
        ]
        self._target_fn_name = ""

        # Trie木の初期化と全語彙の登録(空間計算量 O(V * 平均長))、一度のみ実行
        self._trie = TokenTrie()
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

    def set_user_prompt(self, user_prompt: str) -> None:
        """
            推論ループ内でプロンプトごとに呼び出し、
            標的関数の再計算とFSMの差分キャッシュをリセットする。
        """
        # FSM内での動的Logit Bias(Semantic Logit Steering)
        # 確率分布を直接操作するためのスコアリング事前計算
        self._target_fn_name = self._calculate_target_function(user_prompt)
        self._cached_text = ""
        self._cached_loop_state = (
            0, False, False, "", False, "", "", frozenset()
        )
        # self._prompt_token_ids = set(self._tokenizer.encode(user_prompt))

    def _calculate_target_function(self, user_prompt: str) -> str:
        """
            ユーザープロンプトと関数定義間のJaccard係数を計算し、
            意味論的に最も関連性の高い関数名をO(1)実行時に利用可能にする
        """
        if not user_prompt:
            return ""

        prompt_words = set(re.findall(r'[a-zA-Z0-9_]+', user_prompt.lower()))
        best_fn = ""
        max_score = -1.0

        for fn in self._available_functions:
            name = _get_attr(fn, "name", "")
            desc = _get_attr(fn, "description", "")
            name_words = set(name.lower().replace("_", " ").split())
            desc_words = set(re.findall(r'[a-zA-Z0-9_]+', desc.lower()))
            target_words = desc_words | name_words

            # Jaccard係数 = (積集合の要素数) / (和集合の要素数)
            intersection = prompt_words & target_words
            union = prompt_words | target_words
            score = len(intersection) / len(union) if union else 0.0

            if score > max_score:
                max_score = score
                best_fn = name

        return best_fn

    def _run_fsm_loop(
        self,
        start_state: tuple[
            int, bool, bool, str, bool, str, str, frozenset[Any]],
        text_chunk: str
    ) -> tuple[
            int, bool, bool, str, bool, str, str, frozenset[Any]
    ]:
        (depth, in_string, escape, last_structural_char, is_value_context,
         current_string, last_key, seen_root_keys) = start_state

        seen_keys_set = set(seen_root_keys)

        # 文字列から1文字ずつループ
        for char in text_chunk:
            # もし文字列内なら
            if in_string:
                if escape:
                    escape = False
                    current_string += char
                elif char == "\\":
                    escape = True
                    current_string += char
                elif char == '"':
                    in_string = False
                    last_structural_char = '"'
                    # print(f"4-2. DEBUG: char='{char}'だよ")
                    if not is_value_context and depth == 1:
                        # print("4-3. DEBUG: last_keyを定義したよ")
                        last_key = current_string
                        seen_keys_set.add(last_key)
                        # print(f"4-4. DEBUG: last_key='{last_key}'だよ")
                else:
                    current_string += char
            else:
                if char == '"':
                    in_string = True
                    current_string = ""
                elif char == "{":
                    depth += 1
                    last_structural_char = char
                    is_value_context = False
                elif char == "}":
                    if depth > 0:
                        depth -= 1
                    last_structural_char = char
                    is_value_context = False
                elif char in [","]:
                    last_structural_char = char
                    is_value_context = False
                elif char in [":"]:
                    last_structural_char = char
                    is_value_context = True

        return (
            depth, in_string, escape, last_structural_char, is_value_context,
            current_string, last_key, frozenset(seen_keys_set)
        )

    def _determine_current_state(
        self, current_text: str
    ) -> tuple[str, int, str, bool, bool, str, frozenset[str]]:
        """
        文字列を線形走査し、現在の文脈を決定論的に抽出する。
        生成済みテキスト長(N)に対して1回のループ処理で状態を判定。
        O(N): 線形時間計算量

        Returns:
            state: 次に期待される構文状態
            depth: int 括弧のスタック(ネストの深さ測定用)
            current_string: 現在入力中の文字列の中身
            in_string: 文字列内部にいるかのフラグ
            is_value_context: ":"の後 -> True、",", "{}"の後 -> False
        """
        if not current_text:
            self._cached_text = ""
            self._cached_loop_state = (
                0, False, False, "", False, "", "", frozenset()
            )
            return (
                FSMState.EXPECT_BEGIN_OBJECT, 0, "",
                False, False, "", frozenset()
            )

        if current_text.startswith(self._cached_text):
            new_chunk = current_text[len(self._cached_text):]
            loop_state = self._run_fsm_loop(self._cached_loop_state, new_chunk)
        else:
            initial_state: tuple[
                int, bool, bool, str, bool, str, str, frozenset[Any]
            ] = (
                0, False, False, "", False, "", "", frozenset()
            )
            loop_state = self._run_fsm_loop(initial_state, current_text)

        self._cached_text = current_text
        self._cached_loop_state = loop_state

        (depth, in_string, escape, last_structural_char, is_value_context,
         current_string, last_key, seen_root_keys) = loop_state

        state: str = FSMState.EXPECT_COMMA_OR_END_OBJECT
        if depth == 0 and not in_string:
            if last_structural_char == "}":
                state = FSMState.DONE
            else:
                state = FSMState.EXPECT_BEGIN_OBJECT
        elif in_string:
            if is_value_context:
                state = FSMState.EXPECT_VALUE
            else:
                state = FSMState.EXPECT_KEY
        else:
            if last_structural_char in ("{", ","):
                state = FSMState.EXPECT_KEY
            elif last_structural_char == ":":
                state = FSMState.EXPECT_VALUE
            elif last_structural_char == '"':
                if is_value_context:
                    state = FSMState.EXPECT_COMMA_OR_END_OBJECT
                else:
                    state = FSMState.EXPECT_COLON

        return (
            state, depth, current_string,
            in_string, is_value_context, last_key, seen_root_keys
        )

    def _get_allowed_characters(self, state: str, depth: int) -> set[str]:
        """基本的なJson構文に基づいて許可する文字セットを取得"""
        if state == FSMState.EXPECT_BEGIN_OBJECT:
            return {"{"}
        elif state == FSMState.EXPECT_KEY:
            return {'"'}
        elif state == FSMState.EXPECT_COLON:
            return {':'}
        elif state == FSMState.EXPECT_VALUE:
            # valueを深くネストし続けないように制限
            if depth >= 2:
                return {
                    '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    ".", "-", "+", "e", "E", "t", "f", "n", ",", "}", "]"
                }
            return {
                '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                ".", "-", "+", "e", "E", "t", "f", "n",
                "{", "[", ",", "}", "]"
            }
        elif state == FSMState.EXPECT_COMMA_OR_END_OBJECT:
            return {",", "}", "]"}
        elif state == FSMState.DONE:
            return {"\n", " "}
        return set()

    def filter_logits(
        self, logits: list[float], current_text: str
    ) -> list[float]:
        """
        状態遷移(FSM)アルゴリズム
        プッシュダウンオートマトン(PDA)
        """
        try:
            # 1. current_textを解析し、現在のJsonノードを特定する
            (state, depth, current_string, in_string,
             is_value_context, last_key, seen_root_keys) = (
                self._determine_current_state(current_text)
            )

            # 全ての確率を-infで初期化
            filtered_logits = [-math.inf] * len(logits)
            # スタックの深さが1の場合のみルートキーのスキーマ強制を適用
            is_root_level = (depth == 1)
            # 現在の状態に合わせて許容する文字を取得
            allowed_chars = self._get_allowed_characters(state, depth)

            # 同じ必須キーのループ防止
            if is_root_level and state == FSMState.EXPECT_COMMA_OR_END_OBJECT:
                # nameしか出てない時、終了を禁止
                if "name" in seen_root_keys and (
                        "parameters" not in seen_root_keys):
                    allowed_chars = {",", " ", "\n", "\t", "\r"}
                # name, parameterが出た時、終了を許可
                if "name" in seen_root_keys and (
                        "parameters" in seen_root_keys):
                    allowed_chars = {"}", " ", "\n", "\t", "\r"}

            allowed_structural = {
                c
                for c in allowed_chars
                if c.strip()
            }

            # ルートキーの順番の強制(name -> parameters)
            # 使用したら削除
            if "name" not in seen_root_keys:
                available_root_keys = ["name"]
            elif "parameters" not in seen_root_keys:
                available_root_keys = ["parameters"]
            else:
                available_root_keys = []

            # 使用したキーはリストから削除
            valid_token_ids = set()

            # 空白tokenの補充
            if " " in allowed_chars or "\n" in allowed_chars:
                valid_token_ids.update(self._trie.whitespace_token_ids)

            # 選択された関数名の動的取得とスキーマ解析
            selected_fn = self._target_fn_name
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', current_text)
            if name_match:
                selected_fn = name_match.group(1)

            allowed_param_keys = []
            if selected_fn:
                for fn in self._available_functions:
                    if _get_attr(fn, "name") == selected_fn:
                        params = _get_attr(fn, "parameters", {})
                        if isinstance(params, dict):
                            allowed_param_keys = list(params.keys())
                        break

            # 関数名が未確定の場合、
            # デッドロック回避で全関数のキーを許可
            if not allowed_param_keys:
                all_keys: set[str] = set()
                for fn in self._available_functions:
                    params = _get_attr(fn, "parameters", {})
                    if isinstance(params, dict):
                        all_keys.update(params.keys())
                allowed_param_keys = list(all_keys)

            # Trie木を用いたO(1) ~ O(L) の語彙フィルタリング
            if not in_string:
                # 構文文字の待機時
                # TrieのハッシュマップからO(1)で許可token群取得
                for char in allowed_structural:
                    if char in self._trie.first_char_index:
                        valid_token_ids.update(
                            self._trie.first_char_index[char]
                        )

                # DONEに到達時
                # 終了tokenの確率を強制的に引き上げ
                if state == FSMState.DONE:
                    # 既存許可リストをクリア
                    valid_token_ids = set()
                    # Engineが停止条件として認めるEOS tokenのみ許可する
                    eos_id = getattr(self._tokenizer, "eos_tokne_id", None)
                    if eos_id is None and (
                            hasattr(self._tokenizer, "_tokenizer")):
                        eos_id = getattr(
                            self._tokenizer._tokenizer, "eos_tokne_id", None)
                    if isinstance(eos_id, list):
                        eos_id = eos_id[0]
                    if eos_id is not None and eos_id < len(logits):
                        valid_token_ids.add(eos_id)
                    else:
                        valid_token_ids.add(151643)

                if is_root_level and (
                        state == FSMState.EXPECT_COMMA_OR_END_OBJECT):
                    if len(seen_root_keys) >= len(self._allowed_root_keys):
                        filtered_valid_ids = set()
                        for t_id in valid_token_ids:
                            t_str = (self._tokenizer._id_to_token.get(
                                    t_id, "").replace("Ġ", " ")
                            )
                            if "," not in t_str and '"' not in t_str:
                                filtered_valid_ids.add(t_id)
                        valid_token_ids = filtered_valid_ids

                # Semantic制約(ルートキーの開始)
                if (depth == 1 or depth == 2) and state == FSMState.EXPECT_KEY:
                    filtered_valid_ids = set()
                    if depth == 1:
                        keys_to_check = available_root_keys
                    else:
                        keys_to_check = allowed_param_keys

                    for t_id in valid_token_ids:
                        t_str = (
                            self._tokenizer._id_to_token.get(
                                t_id, "").replace("Ġ", " ")
                        )
                        if '"' in t_str:
                            content = t_str[t_str.find('"') + 1:]
                            if not keys_to_check or (
                                    not content) or (
                                    any((k.startswith(content)) or (
                                        content == k + '"')
                                        for k in keys_to_check)):
                                filtered_valid_ids.add(t_id)
                            else:
                                # 引用符を含まない時、純粋な空白tokenのみ許可
                                if not t_str.strip(' \n\r\tĊ'):
                                    filtered_valid_ids.add(t_id)

                    valid_token_ids = filtered_valid_ids

                # EXPECT_VALUEの時
                # nameのvalue(関数名)
                if is_root_level and (
                        state == FSMState.EXPECT_VALUE) and (
                        last_key == "name"):
                    filtered_valid_ids = set()

                    for t_id in valid_token_ids:
                        t_str = (
                            self._tokenizer._id_to_token.get(
                                t_id, "").replace("Ġ", " ")
                            )

                        if '"' in t_str:
                            content = t_str[t_str.find('"') + 1:]
                            if not content:
                                filtered_valid_ids.add(t_id)
                            elif self._target_fn_name:
                                if self._target_fn_name.startswith(
                                        content) or (
                                        content == (
                                            self._target_fn_name + '"')):
                                    filtered_valid_ids.add(t_id)
                            # user_promptが渡されずスコアリングされていない時
                            # 全関数名を許可
                            else:
                                if any((fn.startswith(content)) or (
                                    content == fn + '"')
                                        for fn in self._valid_fn_names):
                                    filtered_valid_ids.add(t_id)
                        else:
                            # 引用符を含まない時、純粋な空白tokenのみ許可
                            if not t_str.strip(' \n\r\tĊ'):
                                filtered_valid_ids.add(t_id)

                    valid_token_ids = filtered_valid_ids
            else:
                # 文字列内部の処理
                if (depth == 1 or depth == 2) and not is_value_context:
                    if depth == 1:
                        keys_to_check = available_root_keys
                    else:
                        keys_to_check = allowed_param_keys
                    for t_id, t_str in self._vocab_items:
                        clean_str = t_str.replace("Ġ", " ")
                        new_str = current_string + clean_str
                        if not keys_to_check or (
                                any((k.startswith(new_str)) or (
                                    new_str == k + '"')
                                    for k in keys_to_check)):
                            valid_token_ids.add(t_id)

                elif is_root_level and is_value_context and last_key == "name":
                    for t_id, t_str in self._vocab_items:
                        clean_str = t_str.replace("Ġ", " ")
                        new_str = current_string + clean_str
                        if self._target_fn_name:
                            if self._target_fn_name.startswith(new_str) or (
                                    new_str == self._target_fn_name + '"'):
                                valid_token_ids.add(t_id)
                        else:
                            if any((fn.startswith(new_str)) or (
                                new_str == fn + '"')
                                    for fn in self._valid_fn_names):
                                valid_token_ids.add(t_id)

                else:
                    valid_token_ids.update(
                        t_id for t_id, _ in self._vocab_items
                    )

            # Logitsの再構築とSemantic Logit Steering
            for t_id in valid_token_ids:
                filtered_logits[t_id] = logits[t_id]

                if is_root_level and is_value_context and (
                        last_key == "name" and self._target_fn_name):
                    filtered_logits[t_id] += 10.0

                # promptに含まれるtokenに下駄を履かせる
                # if depth >= 2 and in_string:
                    # if t_id in self._prompt_token_ids:
                    # filtered_logits[t_id] += 10.0

                if state == FSMState.DONE:
                    filtered_logits[t_id] += 100.0

            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # fail safe: パイプラインのクラッシュを防ぐために元のlogitsを返す
            return logits
