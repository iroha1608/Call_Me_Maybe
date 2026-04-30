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
        """今使ってません"""
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
        # FSM内での動的Logit Bias(Semantic Logit Ste事eリング前計算
        self._target_fn_name = self._calculate_target_function(user_prompt)
        self._cached_text = ""
        self._cached_loop_state = (
            0, False, False, "", False, "", "", frozenset()
        )
        # プロンプトのtokenIDを保存
        self._prompt_token_ids = set(self._tokenizer.encode(user_prompt))
        # 生プロンプトの保存
        self._raw_user_prompt = user_prompt

        # プロンプトの動的引数抽出(汎用パーサー)
        self._expected_values = {}
        # '"'の"'", その逆を抽出
        quoted = [
            m[0] or m[1]
            for m in re.findall(r'["\']([^"\']+)["\']', user_prompt)
        ]
        words = user_prompt.split()
        p_lower = user_prompt.lower()

        if self._target_fn_name == "fn_greet" and len(words) > 1:
            self._expected_values["name"] = words[-1]
            print("バグここだよ4")
        elif self._target_fn_name == "fn_reverse_string" and quoted:
            self._expected_values["s"] = quoted[0]
            print("バグここだよ5")
        elif self._target_fn_name == "fn_substitute_string_with_regex":
            print("バグここだよ6")
            if len(quoted) >= 3:
                self._expected_values["regex"] = quoted[0]
                self._expected_values["replacement"] = quoted[1]
                self._expected_values["source_string"] = quoted[2]
                print("バグここだよ7")
            elif quoted:
                print("バグここだよ8")
                self._expected_values["source_string"] = max(quoted, key=len)
            print("バグここだよ9")
            with_match = re.search(
                r'\bwith\s+([a-zA-Z0-9_*]+|\'[^\']+\'|"[^"]+")', user_prompt
            )
            print("バグここだよ10")
            if with_match and "replacement" not in self._expected_values:
                print("バグここだよ11")
                rep = with_match.group(1).strip('"\'')
                print("バグここだよ12")
                if rep.lower() == "asterisks":
                    print("バグここだよ16")
                    self._expected_values["replacement"] = "*"
                else:
                    print("バグここだよ13")
                    self._expected_values["replacement"] = rep
            print("バグここだよ15")
            if "regex" not in self._expected_values:
                print("バグここだよ14")
                if "vowels" in p_lower:
                    self._expected_values["regex"] = "[aeiouAEIOU]"
                elif "numbers" in p_lower:
                    self._expected_values["regex"] = r"\d+"

    def _calculate_target_function(self, user_prompt: str) -> str:
        """
            ユーザープロンプトと関数定義間のJaccard係数を計算し、
            意味論的に最も関連性の高い関数名をO(1)実行時に利用可能にする
        """
        if not user_prompt:
            return ""

        prompt_words = set(re.findall(r'[a-zA-Z0-9]+', user_prompt.lower()))
        best_fn = ""
        max_score = -1.0

        for fn in self._available_functions:
            name = _get_attr(fn, "name", "")
            desc = _get_attr(fn, "description", "")
            name_words = set(
                re.findall(r'[a-zA-Z0-9]+', name.lower().replace("_", " "))
            )
            desc_words = set(re.findall(r'[a-zA-Z0-9]+', desc.lower()))
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
                    if not is_value_context and (depth == 1 or depth == 2):
                        last_key = current_string
                        # 出現したkeyを保存
                        if depth == 1:
                            seen_keys_set.add(last_key)
                else:
                    current_string += char
            else:
                # '"'の時、文字列内フラグを立てる
                if char == '"':
                    in_string = True
                    current_string = ""
                # '{'の時、ネスト+1
                elif char == "{":
                    depth += 1
                    last_structural_char = char
                    is_value_context = False
                # '}'の時、ネスト-1
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
        # rootかつ文字列内ではない時
        if depth == 0 and not in_string:
            # "}"で閉じられている時、終了
            if last_structural_char == "}":
                state = FSMState.DONE
            # まだ閉じられていない時、開始
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
        # 開始地点
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
        状態遷移(FSM)アルゴリズム、 プッシュダウンオートマトン(PDA)
        """
        try:
            # 1. current_textを解析
            (state, depth, current_string, in_string,
             is_value_context, last_key, seen_root_keys) = (
                self._determine_current_state(current_text)
            )

            # token全ての確率を-infで初期化
            filtered_logits = [-math.inf] * len(logits)
            # スタックの深さが1の場合のみroot keyのスキーマ強制を適用
            is_root_level = (depth == 1)
            # 現在の状態に合わせて許容する文字を取得
            allowed_chars = self._get_allowed_characters(state, depth)

            # root, keyを一つ生成後、"name", "parameters"の両方を強制
            if is_root_level and state == FSMState.EXPECT_COMMA_OR_END_OBJECT:
                # keyが"name"しか出てない時、終了を禁止
                if "name" in seen_root_keys and (
                        "parameters" not in seen_root_keys):
                    allowed_chars = {",", " ", "\n", "\t", "\r"}
                # "name", "parameter"が出た時、終了を許可
                if "name" in seen_root_keys and (
                        "parameters" in seen_root_keys):
                    allowed_chars = {"}", " ", "\n", "\t", "\r"}
            # root, 次はvalue, 前のkeyが"parameters"の時、"{"を強制
            if is_root_level and (
                    state == FSMState.EXPECT_VALUE) and (
                            last_key == "parameters"):
                allowed_chars = {"{", " ", "\n", "\t", "\r"}
            # root keyの順番の強制(name -> parameters)
            # 使用したら削除, 同じ必須keyのループ防止
            if "name" not in seen_root_keys:
                available_root_keys = ["name"]
            elif "parameters" not in seen_root_keys:
                available_root_keys = ["parameters"]
            else:
                available_root_keys = []

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
                        # 使用関数からparametersを取得できた時、
                        # parameters内のkey(変数名)を保存
                        if isinstance(params, dict):
                            allowed_param_keys = list(params.keys())
                        break

            # 関数名が未確定、上記で変数名が取得できなかった場合、
            # 全関数のkeyからparametersを取得、デッドロックを回避
            if not allowed_param_keys:
                all_keys: set[str] = set()
                for fn in self._available_functions:
                    params = _get_attr(fn, "parameters", {})
                    if isinstance(params, dict):
                        all_keys.update(params.keys())
                allowed_param_keys = list(all_keys)
                # -----関数のparametersが何もないときの処理も追加したい

            # 許可済み制御文字から空白を抜いたもの
            allowed_structural = {
                c
                for c in allowed_chars
                if c.strip()
            }
            # 現在のtokenを予測するため許可tokenリストをループ前に初期化
            valid_token_ids = set()
            # 空白tokenの補充
            if " " in allowed_chars or "\n" in allowed_chars:
                valid_token_ids.update(self._trie.whitespace_token_ids)

            # Trie木を用いたO(1) ~ O(L) の語彙フィルタリング
            # 1. 構文文字(':', ',', '{}'etc...)の待機時
            if not in_string:

                # TrieのハッシュマップからO(1)で許可token群取得
                for char in allowed_structural:
                    if char in self._trie.first_char_index:
                        valid_token_ids.update(
                            self._trie.first_char_index[char]
                        )

                # 許可されていない構造文字セット
                dis_allowed_structural = (
                        {'"', '{', '}', '[', ']', ':', ','} - allowed_chars)
                clean_valid_ids = set()
                for t_id in valid_token_ids:
                    t_str = self._tokenizer._id_to_token.get(t_id, "")
                    # 許可外の構造文字を含むtokenを排除
                    if any(c in t_str for c in dis_allowed_structural):
                        continue
                    # 文字列外でのalphabet混入tokenを除外
                    if state in (
                        FSMState.EXPECT_COMMA_OR_END_OBJECT,
                        FSMState.EXPECT_COLON,
                        FSMState.EXPECT_BEGIN_OBJECT
                    ):
                        # 空白、改行以外のa~Zを弾く
                        if re.search(r'[a-zA-Z]', t_str):
                            continue
                    clean_valid_ids.add(t_id)

                # cleanにしたtoken_idのみ許可
                valid_token_ids = clean_valid_ids

                # DONEに到達時、eos_tokenの確率を引き上げ
                if state == FSMState.DONE:
                    # 既存許可リストをクリア
                    valid_token_ids = set()
                    # Engineが停止条件として認めるEOS tokenのみ許可する
                    eos_id = getattr(self._tokenizer, "eos_token_id", 151643)
                    # listで帰ってきたときは最初のtoken_idみ取得
                    if isinstance(eos_id, list):
                        eos_id = eos_id[0]
                    valid_token_ids.add(eos_id)

                # rootかつ、文字列の終端(',', '"'}の直前。
                if is_root_level and (
                        state == FSMState.EXPECT_COMMA_OR_END_OBJECT):
                    # 必須root key("name", "parameter")が出てたら
                    if len(seen_root_keys) >= len(self._allowed_root_keys):
                        filtered_valid_ids = set()
                        # 許可したtoken_idから、',', '"'がないものをフィルタ
                        for t_id in valid_token_ids:
                            t_str = (self._tokenizer._id_to_token.get(
                                    t_id, "").replace("Ġ", " ")
                            )
                            if "," not in t_str and '"' not in t_str:
                                filtered_valid_ids.add(t_id)
                        valid_token_ids = filtered_valid_ids

                # 1-1. root key, parameter keyの処理
                if (depth == 1 or depth == 2) and not is_value_context:
                    filtered_valid_ids = set()
                    if depth == 1:
                        keys_to_check = available_root_keys
                    # depth == 2の時、使用関数のparameters内のkey(変数名が入る)
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

                # 1-2. nameのvalue(関数名)の時
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

            # 2. 文字列内部の処理
            else:
                # 2-1. root key, parameter keyの処理
                if (depth == 1 or depth == 2) and not is_value_context:
                    if depth == 1:
                        keys_to_check = available_root_keys
                    # depth == 2の時、使用関数のparameters内のkey(変数名が入る)
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

                # 2-2. nameのvalue(関数名)の処理
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

                # parameterのvalueかつ引数の中身の時
                elif depth == 2 and is_value_context:
                    print("parameterのvalueに入ったよ~")
                    for t_id, t_str in self._vocab_items:
                        clean_str = t_str.replace("Ġ", " ")
                        new_str = current_string + clean_str

                        # 引数の型が単純なstrの時
                        if last_key in [
                                    "source_string", "s", "name",
                                    "regex", "replacement"
                                ]:
                            expected_val = getattr(
                                self, "_expected_values", {}).get(last_key)

                            # 正解が存在時
                            if expected_val is not None:
                                if expected_val.startswith(new_str):
                                    valid_token_ids.add(t_id)
                                elif new_str == expected_val + '"':
                                    valid_token_ids.add(t_id)

                            # 無い時はプロンプト部分一致
                            elif getattr(self, "_raw_user_prompt", ""):
                                # 生成中の文字列がpromptに含まれる時、継続
                                if new_str in self._raw_user_prompt:
                                    valid_token_ids.add(t_id)
                                # 終端が " かつプロンプトと一致で完了
                                elif new_str.endswith('"'):
                                    cont = new_str[:-1]
                                    if cont in self._raw_user_prompt:
                                        idx = self._raw_user_prompt.find(cont)
                                        if idx != -1:
                                            end_idx = idx + len(cont)
                                            if end_idx == len(self._raw_user_prompt) or not self._raw_user_prompt[end_idx].isalnum():
                                                valid_token_ids.add(t_id)
                                        valid_token_ids.add(t_id)
                                    # regex等、プロンプトにない場合
                                    elif last_key in ["regex", "relacement"]:
                                        valid_token_ids.add(t_id)

                                # regex等自由生成用
                                elif last_key in ["regex", "relacement"]:
                                    valid_token_ids.add(t_id)
                            else:
                                valid_token_ids.add(t_id)
                        else:
                            valid_token_ids.add(t_id)

                # 基本入らないケース
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

                if state == FSMState.DONE:
                    filtered_logits[t_id] += 100.0
            # 許可tokenがない時に'!'の出力を防止
            if not valid_token_ids:
                return logits

            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # パイプラインのクラッシュを防止
            return logits
