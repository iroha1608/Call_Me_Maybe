"""
FSMState is an enumeration that represents the states of
a finite state machine (FSM) used for parsing JSON.
LoopState is a dataclass that holds the current state of the parsing loop,
including the current string being parsed, the depth of nesting,
whether we are currently inside a string, and other relevant information.
ParsedContext is a dataclass that represents the context of the parsed JSON
at any given point, including the current string, the FSM state, depth,
and other details. JSONStateTracker is a class that manages the state of JSON
parsing, allowing us to determine the current state based on the input text
and providing allowed characters for each state.
"""
from enum import Enum
from dataclasses import dataclass, field


class FSMState(str, Enum):
    """
        FSMState is an enumeration that represents the states of
        a finite state machine (FSM) used for parsing JSON.
    """
    BEGIN = "BEGIN"
    KEY = "KEY"
    COLON = "COLON"
    VALUE = "VALUE"
    COMMA_OR_END = "COMMA_OR_END"
    DONE = "DONE"


@dataclass(frozen=True)
class LoopState:
    """
        LoopState is a dataclass that holds the current state of
        the parsing loop, including the current string being parsed,
        the depth of nesting, whether we are currently inside a string,
        and other relevant information.
    """
    current_string: str = ""
    depth:  int = 0
    in_string: bool = False
    is_value_context: bool = False
    escape: bool = False
    last_structural_char: str = ""
    last_key: str = ""
    seen_root_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass
class ParsedContext:
    """
        ParsedContext is a dataclass that represents the context of
        the parsed JSON at any given point, including the current string,
        the FSM state, depth, and other details.
    """
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
        JSONStateTracker is a class that manages the state of JSON parsing,
        allowing us to determine the current state based on the input text
        and providing allowed characters for each state.
    """
    def __init__(self) -> None:
        """Initialize the JSONStateTracker by resetting its state. """
        self.reset()

    def reset(self) -> None:
        """Reset the state of the JSONStateTracker to its initial values. """
        self._cached_text = ""
        self._cached_loop_state = LoopState()
        # 開始地点

    def determine_current_state(self, current_text: str) -> ParsedContext:
        """
            Determine the current state of
            JSON parsing based on the input text.
            This method uses a finite state machine (FSM) approach to
            analyze the structure of the JSON being parsed,
            including the depth of nesting, whether we are inside a string,
            and the context of keys and values.
            Args:
                current_text (str): The current text being parsed as JSON.
            Returns:
                ParsedContext: An object containing the current string,
                FSM state, depth, and other relevant details
                about the parsing context.
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
        """
            Run the finite state machine (FSM) loop to analyze the input chunk
            of text and update the parsing state accordingly. This method
            processes each character in the input chunk, updating the depth
            of nesting, whether we are inside a string, and other relevant
            information based on the structure of the JSON being parsed.
            Args:
                state (LoopState):
                    The current state of the FSM before processing the chunk.
                chunk (str): The new chunk of text to be processed.
            Returns:
                LoopState:
                    The updated state of the FSM after processing the chunk.
        """
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
        """
            Get the set of allowed characters based on the current
            FSM state and depth of nesting.
            This method defines the valid characters that can be generated
            by the LLM at each state of the JSON parsing process,
            helping to constrain the output to valid JSON structures.
            Args:
                state (FSMState): The current state of the FSM.
                depth (int):
                    The current depth of nesting in the JSON structure.
            Returns:
                set[str]: A set of characters that are allowed to be generated
                by the LLM based on the current FSM state and depth of nesting.
        """
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
                    ".", "-", "e", "E", "t", "f", "n",
                    ",", "}", "\n"
                }
            return {
                '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                ".", "-", "e", "E", "t", "f", "n",
                "{", ",", "}", "\n"
            }
        elif state == FSMState.COMMA_OR_END:
            return {",", "}", "]"}
        elif state == FSMState.DONE:
            return {"\n", " "}
        return set()
