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
    プッシュダウンオートマトン(PDA)を使用し、制約付きデコーディングを行う。
    言語モデルが出力した確率分布(Logits)を操作、
    Json構文に違反するトークンをフィルタリングするクラス。
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

    def _determine_current_state(
        self,
        current_text: str,
    ) -> tuple[str, list[str]]:
        """
        生成済みテキスト長(N)に対して1回のループ処理で状態を判定。
        O(N): 線形時間計算量
        """

        text = current_text
        if not text:
            return FSMState.EXPECT_BEGIN_OBJECT, []

        stack: list[str] = []
        in_string = False
        escape = False
        last_structural_char = ""

        for char in text:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
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
        # キーの終わりか文字列の終わりの可能性がある。
        # 正確なチェックはkey, valueのコンテキストを追跡する必要があるが、
        # 代替オブジェクト内の閉じ引用符の後はコロンかカンマ
        elif last_structural_char == '"':
            # 最後がコロンなら、この引用符は終わりになる
            if text.rfind(":") > text.rfind(","):
                return FSMState.EXPECT_COMMA_OR_END_OBJECT, stack
            return FSMState.EXPECT_COLON, stack
        # 数字, boolに対するデフォルトの返り値
        return FSMState.EXPECT_COMMA_OR_END_OBJECT, stack

    def _get_allowed_characters(self, state: str) -> set[str]:
        """ FSMの遷移状態を許容される次の文字にマッピングする。"""
        # space, 改行, tab, 復帰文字
        white_space = {" ", "\n", "\t", "\r"}

        if state == FSMState.EXPECT_BEGIN_OBJECT:
            return {"{"} | white_space
        elif state == FSMState.EXPECT_KEY:
            return {'"', "}"} | white_space
        elif state == FSMState.EXPECT_COLON:
            return {':'} | white_space
        elif state == FSMState.EXPECT_VALUE:
            return {'"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "-", "+", "t", "f", "n", "{", "["} | white_space
        elif state == FSMState.EXPECT_COMMA_OR_END_OBJECT:
            return {",", "}", "]"} | white_space
        elif state == FSMState.DONE:
            return {"\n", " "}
        return set()

    def filter_logits(
        self, logits: list[float], current_text: str
    ) -> list[float]:
        try:
            # 状態遷移(FSM)
            # 1. current_textを解析し、現在のJsonノードを特定する
            # (例:key, value, bracketを待つ)
            state, stack = self._determine_current_state(current_text)

            # 文字列内では、Escapeされていない構造的な改行を除いて
            # ほぼなんでも許可する
            if state in (
                    FSMState.EXPECT_VALUE, FSMState.DONE) and (
                    current_text.count('"') % 2 != 0):
                return logits

            # 現在の状態に合わせて文字を許容
            allowed_chars = self._get_allowed_characters(state)
            # 元のデータが破壊されないように浅いコピー
            filtered_logits = list(logits)

            # クリーンな文字列を取得(Qwen固有のtoken形式に対応)
            def _clean_token(t_str: str) -> str:
                # 標準的なBPEのスペースを処理
                t_str = t_str.replace("Ġ", " ").replace(" ", " ")
                # 構造文字のみを検索する場合はスペース削除
                # 空白文字が厳密に許可されている場合は残す
                return t_str.lstrip() if t_str.lstrip() else " "

            for token_id, token_str in self._vocab_items:
                clean_str = _clean_token(token_str)

                # 特別なtoken(<|endoftext|>)等は<で始まる場合が多い
                # DONEならEOS tokenが生成を終了
                if clean_str.startswith("<") and state == FSMState.DONE:
                    continue

                # 完全に空のtokenは安全か中立
                if not clean_str:
                    continue

                first_char = clean_str[0]
                if first_char not in allowed_chars:
                    filtered_logits[token_id] = -math.inf

            # 3. invalid_token_ids内の各token_idについて:
            # filtered_logits[token_id] = -math.inf

            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # fail safe: パイプラインのクラッシュを防ぐために元のlogitsを返す
            return logits
