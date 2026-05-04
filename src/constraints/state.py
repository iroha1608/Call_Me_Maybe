from enum import Enum
from dataclasses import dataclass, field


class FSMState(str, Enum):
    """Json解析用オートマトンの状態を表す。"""
    BEGIN = "BEGIN"
    KEY = "KEY"
    COLON = "COLON"
    VALUE = "VALUE"
    COMMA_OR_END = "COMMA_OR_END"
    DONE = "DONE"


@dataclass(frozen=True)
class LoopState:
    """文字単位のパースループを回すための内部状態"""
    # 現在の文字列
    current_string: str = ""
    # 0=開始か終了、1=root、2=引数の中身
    depth:  int = 0
    # '"'のあと、文字列内
    in_string: bool = False
    # valueの入力中
    is_value_context: bool = False
    # '\'のあと、エスケープ後
    escape: bool = False
    # 最後に出てきた構造文字
    last_structural_char: str = ""
    # 最後に出たkey
    last_key: str = ""
    # これまでに出たroot key
    seen_root_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass
class ParsedContext:
    """最終的にdetermine_current_stateが返す情報"""
    # 現在の状態
    current_string: str
    state: FSMState
    depth:  int
    in_string: bool
    is_value_context: bool
    last_key: str
    seen_root_keys: frozenset[str]


class JSONStateTracker:
    """
        JSONの生成状態を追跡、管理するクラス。
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """新しいプロンプトが来た時に初期化"""
        self._cached_text = ""
        self._cached_loop_state = LoopState()
        # 開始地点

    def determine_current_state(self, current_text: str) -> ParsedContext:
        """
            文字列を線形走査し現在の文脈を抽出する。
            生成済みテキスト長(N)に対して1回のループ処理で状態を判定。
        """
        if not current_text:
            self.reset()
            return ParsedContext(
                current_string="",
                state=FSMState.BEGIN,
                depth=0,
                in_string=False,
                is_value_context=False,
                last_key="",
                seen_root_keys=frozenset()
            )

        # 前回の解析結果をキャッシュ、増えた文だけパースする
        if current_text.startswith(self._cached_text):
            new_chunk = current_text[len(self._cached_text):]
            loop_state = self._run_fsm_loop(self._cached_loop_state, new_chunk)
        else:
            loop_state = self._run_fsm_loop(LoopState(), current_text)
        # 現在の文字列でキャッシュを更新、次回続きから解析開始
        self._cached_text = current_text
        self._cached_loop_state = loop_state
        # LoopStateのプロパティへのアクセスを変数名で可能にする
        ls = loop_state
        state = FSMState.COMMA_OR_END

        # rootかつ文字列内ではない時
        if ls.depth == 0 and not ls.in_string:
            # "}"で閉じられている時、終了
            if ls.last_structural_char == "}":
                state = FSMState.DONE
            # まだ閉じられていない時、開始
            else:
                state = FSMState.BEGIN
        # 文字列内の時keyかvalueかチェック
        elif ls.in_string:
            if ls.is_value_context:
                state = FSMState.VALUE
            else:
                state = FSMState.KEY
        else:
            # 最後の構文文字が"{", ","の時、次はkey
            if ls.last_structural_char in ("{", ","):
                state = FSMState.KEY
            # ":"の時、次はvalue
            elif ls.last_structural_char == ":":
                state = FSMState.VALUE
            # '"'の時
            elif ls.last_structural_char == '"':
                # もしvalue内なら、次は","か"}"
                if ls.is_value_context:
                    state = FSMState.COMMA_OR_END
                # もしkey内なら、次は":"
                else:
                    state = FSMState.COLON

        return ParsedContext(
            current_string=ls.current_string,
            state=state,
            depth=ls.depth,
            in_string=ls.in_string,
            is_value_context=ls.is_value_context,
            last_key=ls.last_key,
            seen_root_keys=ls.seen_root_keys
        )

    def _run_fsm_loop(self, state: LoopState, chunk: str) -> LoopState:
        """文字列チャンクを1文字ずつパース、新しい内部情報を返す。"""
        # ループ内で更新するのでローカルで保存。
        depth = state.depth
        in_string = state.in_string
        escape = state.escape
        last_structural_char = state.last_structural_char
        is_value_context = state.is_value_context
        current_string = state.current_string
        last_key = state.last_key
        seen_keys_set = set(state.seen_root_keys)

        # 文字列から1文字ずつループ
        for char in chunk:
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

        return LoopState(
            current_string=current_string,
            depth=depth,
            in_string=in_string,
            is_value_context=is_value_context,
            escape=escape,
            last_structural_char=last_structural_char,
            last_key=last_key,
            seen_root_keys=frozenset(seen_keys_set)
        )

    def get_allowed_characters(self, state: FSMState, depth: int) -> set[str]:
        """基本的なJson構文に基づいて許可する文字セットを取得"""
        if state == FSMState.BEGIN:
            return {"{"}
        elif state == FSMState.KEY:
            return {'"'}
        elif state == FSMState.COLON:
            return {':'}
        elif state == FSMState.VALUE:
            # valueを深くネストし続けないように制限
            if depth >= 2:
                return {
                    '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    ".", "-", "e", "E", "t", "f", "n", ",", "}", "]"
                }
            return {
                '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                ".", "-", "e", "E", "t", "f", "n",
                "{", "[", ",", "}", "]"
            }
        elif state == FSMState.COMMA_OR_END:
            return {",", "}", "]"}
        elif state == FSMState.DONE:
            return {"\n", " "}
        return set()
