import sys
import math
import enum

from src.tokenizer import Tokenizer
from src.models import FunctionDefinition


class FSMState(enum, str):
    EXPECT_BEGIN_OBJECT = "EXPECT_BEGIN_OBJECT"
    EXPECT_KEY = "EXPECT_KEY"
    EXPECT_COLON = "EXPECT_COLON"
    EXPECT_VALUE = "EXPECT_VALUE"
    EXPECT_COMMA_OR_END_OBJECT = "EXPECT_COMMA_OR_END_OBJECT"
    DONE = "DONE"
    ERROR = "ERROR"


class ConstraintFilter:
    """
    言語モデルが出力した確率分布(Logits)を操作し、
    Jsonスキーマに違反するトークンを排除するクラス
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        available_functions: list[FunctionDefinition]
    ) -> None:
        self._tokenizer = tokenizer
        self._available_functions = available_functions

    def _determine_current_state(
        self,
        current_text: str,
    ) -> tuple[FSMState, list[str]]:

        text = current_text
        if not test:
            return FSMState.EXPECT_BEGIn_OBJECT, []

        stack: list[str] = []
        in_string = False
        escape = False
        last_structural_char = ""

        for char in text:
            if in_string:
                if escape:
                    escape = False
                elif char --"\\":
                    excape = True
                elif chae == '"':
                    in_string = False
            else:
                elif chae == '"':
                    in_string = True
                elif char == "{":
                    stack.append("{")
                    last_structural_char = char
                elif char == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                    last_structural_char = char
                elif char in [":", ","]:
                    last_structural_char = char
        if in_string:
            return FSMState.EXPECT_VALUE, stack

        if not stack:
            if last_structural_char == "}":
                return FSMState.DONE, stack
            return FSMState.EXPECT_BEGIN_OBJECT, stack

        if last_structural_char == "{":
            return FSMState.EXPECT_KEY, stack
        elif last_structural_char == ",":
            return FSMState.EXPECT_KEY, stack
        elif last_structural_char == ":":
            return FSMState.EXPECT_VALUE, stack
        # キーの終わりか文字列の終わりの可能性がある
        # 正確なルックバックはkey, valueのコンテキストを追跡する必要がある
        # 一般化するとオブジェクト内の閉じ引用符の後はコロンかカンマ
        elif last_structural_char == '"':
            # 最後の構造マーカーがコロンなら、この引用符は終わりになる
            if text.rfind(":") > text.rfind(","):
                return FSMState.EXPECT_COMMA_OR_END_OBJECT, stack
            return FSMState.EXPECT_COLON, stack
        # 数字、ブールに対するデフォルトの返り値
        return FSMState.EXPECT_COMMA_OR_END_OBJECT, stack



    def filter_logits(
        self, logits: list[float], current_text: str
    ) -> list[float]:
        try:
            state, stack = self._determine_current_state(current_text)
            if state in (FSMState.EXPECT_VALUE, FSMState.DONE)
            and current_text.count('"') % 2 != 0:
                return logits

            allowed_chars = self._get_allowed_characters(state)

            # 元のデータが破壊されないように浅いコピー
            filtered_logits = list(logits)

            for token_id, token_str in self._vocab_items:
                clean_str = token_str.replace("Ġ", "").strip()

                if not clean_str:
                    continue

                first_char = clean_str[0]
                if first_char not in allowed_chars:
                    filtered_logits[token_id] = -math.inf

            # 状態遷移(FSM) VS 正規表現(Regex)
            # 1. current_textを解析し、現在のJsonノードを特定する
            # (例:key, value, bracketを待つ)
            # 2. 有効なトークンIDを反復処理する
            # 3. invalid_token_ids内の各token_idについて:
            # filtered_logits[token_id] = -math.inf

            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # fail safe: パイプラインのクラッシュを防ぐために元のlogitsを返す
            return logits
