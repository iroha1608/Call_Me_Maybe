import sys
import math
from enum import Enum

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
            fn["name"] for fn in self._available_functions
        ]

    def _determine_current_state(
        self,
        current_text: str,
    ) -> tuple[FSMState, list[str], str, bool, bool, str, set[str]]:
        """
        文字列を線形走査し、現在の文脈を決定論的に抽出する。
        生成済みテキスト長(N)に対して1回のループ処理で状態を判定。
        O(N): 線形時間計算量

        Returns:
            state: 次に期待される構文状態
            stack: 括弧のスタック(ネストの深さ測定用)
            current_string: 現在入力中の文字列の中身
            in_string: 文字列内部にいるかのフラグ
            is_value_context: ":"の後 -> True、",", "{}"の後 -> False
        """
        if not current_text:
            return (
                FSMState.EXPECT_BEGIN_OBJECT, [], "",
                False, False, "", set()
            )

        stack: list[str] = []
        in_string = False
        escape = False
        last_structural_char = ""
        is_value_context = False
        current_string = ""
        # ひとつ前に出力されたキーを追跡
        last_key = ""
        # 既に出力されたルートキーを追跡
        seen_root_keys: set[str] = set()

        # 文字列から1文字ずつループ
        for char in current_text:
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
                    if not is_value_context and len(stack) == 1:
                        # print("4-3. DEBUG: last_keyを定義したよ")
                        last_key = current_string
                        seen_root_keys.add(last_key)
                        # print(f"4-4. DEBUG: last_key='{last_key}'だよ")
                else:
                    current_string += char
            else:
                if char == '"':
                    in_string = True
                    current_string = ""
                    # last_structural_char = ""
                elif char == "{":
                    stack.append("{")
                    last_structural_char = char
                    is_value_context = False
                elif char == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                    last_structural_char = char
                    is_value_context = False
                elif char in [","]:
                    last_structural_char = char
                    is_value_context = False
                elif char in [":"]:
                    last_structural_char = char
                    is_value_context = True

        state = FSMState.EXPECT_COMMA_OR_END_OBJECT
        if not stack and not in_string:
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
            state, stack, current_string,
            in_string, is_value_context, last_key, seen_root_keys
        )

    def _get_allowed_characters(self, state: str) -> set[str]:
        """基本的なJson構文に基づいて許可する文字セットを取得"""
        if state == FSMState.EXPECT_BEGIN_OBJECT:
            return {"{"}
        elif state == FSMState.EXPECT_KEY:
            return {'"'}
        elif state == FSMState.EXPECT_COLON:
            return {':'}
        elif state == FSMState.EXPECT_VALUE:
            return {'"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "+", "t", "f", "n", "{", "["}
        elif state == FSMState.EXPECT_COMMA_OR_END_OBJECT:
            return {",", "}", "]"}
        elif state == FSMState.DONE:
            return {"\n", " "}
        return set()

    def filter_logits(
        self, logits: list[float], current_text: str
    ) -> list[float]:
        # print(f"DEBUG: fn_names='{self._valid_fn_names}'")
        try:
            # 状態遷移(FSM)
            # 1. current_textを解析し、現在のJsonノードを特定する
            # (例:key, value, bracketを待つ)
            (state, stack, current_string,
             in_string, is_value_context, last_key, seen_root_keys) = (
                self._determine_current_state(current_text)
            )

            # Valueの文字列内では自由に推論を許可する
            # if in_string and is_value_context:
            #     return logits

            # 元のデータが破壊されないように浅いコピー
            filtered_logits = list(logits)
            # ホワイトリスト方式
            # 全てのtokenを-infで初期化、許可されたものだけ元の確率で上書き
            # filtered_logits = [-math.inf] * len(logits)
            # 現在の状態に合わせて許容する文字を取得
            allowed_chars = self._get_allowed_characters(state)

            # スタックの深さが1の場合のみルートキーのスキーマ強制を適用
            depth = len(stack)
            is_root_level = (depth == 1)

            # 同じ必須キーのループ防止
            # 全てのキーを出力したら","を禁止する。
            if is_root_level and state == FSMState.EXPECT_COMMA_OR_END_OBJECT:
                if len(seen_root_keys) >= len(self._allowed_root_keys):
                    allowed_chars = {"}", " ", "\n", "\t", "\r"}
            # 使用したキーはリストから削除
            available_root_keys = [
                key
                for key in self._allowed_root_keys
                if key not in seen_root_keys
            ]

            # OOV Leakのマスキング
            valid_tids = {t_id for t_id, _ in self._vocab_items}
            for i in range(len(filtered_logits)):
                if i not in valid_tids:
                    filtered_logits[i] = -math.inf

            # クリーンな文字列を取得(Qwen固有のtoken形式に対応)
            # 標準的なBPEのスペースを処理
            def _clean_token(t_str: str) -> str:
                return t_str.replace("Ġ", " ")

            for token_id, token_str in self._vocab_items:
                clean_str = _clean_token(token_str)

                # 空のtokenは無限ループの原因のためマスキング
                if not clean_str:
                    filtered_logits[token_id] = -math.inf
                    continue

                # 特別なtoken(<|endoftext|>)等は<で始まる場合が多い
                # DONEならEOS tokenが生成を終了
                if clean_str.startswith("<"):
                    if state != FSMState.DONE:
                        filtered_logits[token_id] = -math.inf
                    continue
                # ホワイトリスト方式
                # if clean_str.startswith("<"):
                    # if  state == FSMState.DONE:
                    # filtered_logits[token_id] = logits[token_id]
                    # continue

                stripped = clean_str.lstrip()
                if not stripped:
                    if " " not in allowed_chars:
                        filtered_logits[token_id] = -math.inf
                    continue

                # 意味論の強制 接頭辞によるフィルター
                if is_root_level and not is_value_context:

                    # キー文字列の内部の時
                    if in_string:
                        new_str = current_string + clean_str
                        is_valid_schema = any(
                            (k.startswith(new_str))
                            or (new_str.startswith(k + '"'))
                            for k in available_root_keys
                        )
                        if not is_valid_schema:
                            filtered_logits[token_id] = -math.inf
                        # 接頭辞の評価後は構文チェックスキップ
                        continue

                    # キーの開始地点、"を生成するタイミングの時
                    elif state == FSMState.EXPECT_KEY:
                        if stripped.startswith('"'):
                            content = stripped[1:]
                            if content:
                                is_valid_schema = any(
                                    (k.startswith(content))
                                    or (content.startswith(k + '"'))
                                    for k
                                    in available_root_keys
                                )
                                if not is_valid_schema:
                                    filtered_logits[token_id] = -math.inf
                                continue
                                # ホワイトリスト方式
                                # if is_valid_schema:
                                #     filtered_logits[token_id] = (
                                #         logits[token_id]
                                #     )
                                # continue

                if is_root_level and is_value_context and last_key == "name":
                    # print(f"DEBUG4: last_key=nameだよ")
                    if in_string:
                        new_str = current_string + clean_str
                        is_valid_schema = any(
                            (fn.startswith(new_str))
                            or (new_str.startswith(fn + '"'))
                            for fn
                            in self._valid_fn_names
                        )
                        if not is_valid_schema:
                            filtered_logits[token_id] = -math.inf
                        continue
                    elif state == FSMState.EXPECT_VALUE:
                        if stripped.startswith('"'):
                            content = stripped[1:]
                            if content:
                                is_valid_schema = any(
                                    (fn.startswith(content))
                                    or (content.startswith(fn + '"'))
                                    for fn
                                    in self._valid_fn_names
                                )
                                if not is_valid_schema:
                                    filtered_logits[token_id] = -math.inf
                                continue

                # ルート以外の文字列(ネストされたキー)に対する自由生成を許可
                if in_string:
                    continue
                    # ホワイトリスト方式
                    # filtered_logits[token_id] = logits[token_id]
                    # continue

                # Syntaxの強制
                first_char = stripped[0]
                if first_char not in allowed_chars:
                    filtered_logits[token_id] = -math.inf
                # ホワイトリスト方式
                # if first_char in allowed_chars:
                    # filtered_logits[token_id] = logits[token_id]

            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # fail safe: パイプラインのクラッシュを防ぐために元のlogitsを返す
            return logits
