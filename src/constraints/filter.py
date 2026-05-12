import sys
import math
import re

from src.tokenizer import Tokenizer
from src.models import ParameterDefinition, FunctionDefinition

from src.constraints.state import JSONStateTracker, FSMState, ParsedContext
from src.constraints.trie import TokenTrie


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
            list(tokenizer._id_to_token.items())
        )
        # 使用可能関数から名前の一覧を取得
        self._valid_fn_names: list[str] = [
            fn.name for fn in self._available_functions
        ]

        # Trie木の初期化と全語彙の登録(空間計算量 O(V * 平均長))、一度のみ実行
        self._trie = TokenTrie()
        # 語彙からid, strをループ
        for t_id, t_str in self._vocab_items:
            clean_str = t_str.replace("Ġ", " ").replace("Ċ", "\n")
            stripped_str = clean_str.lstrip()
            self._trie.insert(clean_str, stripped_str, t_id)

        self.state_tracker = JSONStateTracker()

    def set_user_prompt(self, user_prompt: str) -> None:
        """
            1個のプロンプトにつき1回のみ呼び出し。
            標的関数の再計算とFSMの差分キャッシュをリセットする。
        """
        # 最適な関数の取得
        self._target_fn_name: str = (
            self._calculate_target_function(user_prompt))
        # 状態トラッカーの初期化、キャッシュのリセット
        self.state_tracker.reset()
        # 生プロンプトの保存
        self._raw_user_prompt = user_prompt
        # プロンプトのtokenIDを保存(未使用)
        self._prompt_token_ids = set(self._tokenizer.encode(user_prompt))

        # JSON構文の初期状態処理のセットアップ
        initial_structure = '{\n  "name": "'
        self._forced_queue: list[int] = (
            self._tokenizer.encode(initial_structure))

        # 各key情報の初期化、取得
        self._expected_root_keys: set[str] = {"name", "parameters"}
        self._seen_root_keys: set[str] = set()
        self._available_root_keys: list[str] = []
        self._expected_param_keys: list[str] = (
            self._get_expected_param_keys())
        self._seen_param_keys: set[str] = set()
        self._param_index: int = 0

    def _calculate_target_function(self, user_prompt: str) -> str:
        """
            プロンプトセット時に使用。
            ユーザープロンプトと関数定義間のJaccard係数を計算し、
            意味論的に最も関連性の高い関数名を利用可能にする。
        """
        if not user_prompt:
            return ""

        prompt_words = set(re.findall(r'[a-zA-Z0-9]+', user_prompt.lower()))
        best_fn = ""
        max_score = -1.0

        for fn in self._available_functions:
            name = fn.name
            desc = fn.description
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

    def _get_expected_param_keys(self) -> list[str]:
        """
            プロンプトセット時、確定時に使用。
            選択された関数のparameters一覧を取得
        """
        # 使用関数からparametersを取得できた時、
        # parameters内のkey(変数名)を保存
        if self._target_fn_name:
            for fn in self._available_functions:
                if fn.name == self._target_fn_name:
                    params = fn.parameters
                    if isinstance(params, dict):
                        return list(params.keys())
        # 関数名が未確定、上記でkey(変数名)が取得できなかった時、
        # 全関数のkeyからparametersを取得、デッドロック回避
        all_keys: set[str] = set()
        for fn in self._available_functions:
            params = fn.parameters
            if isinstance(params, dict):
                all_keys.update(params.keys())

        return list(all_keys)

    def _get_current_function_schema(self) -> dict[str, ParameterDefinition]:
        """
            選択された関数(_target_fn_name)の"parameters"を動的に取得。
            Returnの例: {"source_string": { "type": "string"},
                         "regex":{ "type": "string"}}
        """
        # 最適な関数がない場合
        if not self._target_fn_name:
            return {}
        # 使用可能関数から1つづつループ
        for fn in self._available_functions:
            # ターゲット関数と同じ関数があった時
            if fn.name == self._target_fn_name:
                params = fn.parameters
                if isinstance(params, dict):
                    return params
        return {}

    def _get_available_root_keys(self) -> list[str]:
        """
            root keyの順番を強制("name"->"parameters")。
            現在強制キューで管理。
        """
        if "name" not in self._seen_root_keys:
            return ["name"]
        elif "parameters" not in self._seen_root_keys:
            return ["parameters"]
        else:
            return []

    def _apply_schema_constraints(
        self,
        ctx: ParsedContext,
        allowed_chars: set[str]
    ) -> set[str]:
        """
            進行状況に応じて、許可する文字を上書き。
            現在複数引数の時に継続許可、禁止で使用。
        """
        is_root_level = (ctx.depth == 1)

        # root, keyを一つ生成後、"name", "parameters"の両方を強制
        if ctx.state in (FSMState.COMMA_OR_END, FSMState.VALUE):
            allowed_chars = allowed_chars - {"]"}
            if is_root_level:
                # keyが"name"しか出てない時、継続許可、終了禁止
                if "name" in self._seen_root_keys and (
                        "parameters" not in self._seen_root_keys):
                    allowed_chars = allowed_chars - {"}"}
                    if ctx.state == FSMState.COMMA_OR_END:
                        return {",", " ", "\n", "\t", "\r"}
                # "name", "parameter"が出た時、継続禁止、終了許可
                elif "name" in self._seen_root_keys and (
                        "parameters" in self._seen_root_keys):
                    allowed_chars = allowed_chars - {","}
                    if ctx.state == FSMState.COMMA_OR_END:
                        return {"}", " ", "\n", "\t", "\r"}
            elif ctx.depth == 2:
                # 引数のkeyが残ってる時、継続許可、終了禁止
                if len(self._seen_param_keys) < len(self._expected_param_keys):
                    allowed_chars = allowed_chars - {"}"}
                    if ctx.state == FSMState.COMMA_OR_END:
                        return {",", " ", "\n", "\t", "\r"}
                # すべてのkeyが揃った時、継続禁止、終了許可
                else:
                    allowed_chars = allowed_chars - {","}
                    if ctx.state == FSMState.COMMA_OR_END:
                        return {"}", " ", "\n", "\t", "\r"}

        # root, 次はvalue, 前のkeyが"parameters"の時->"{"を強制
        if is_root_level and ctx.state == FSMState.VALUE and (
                ctx.last_key == "parameters"):
            return {"{", " ", "\n", "\t", "\r"}

        # 引数のない関数の時->keyの入力禁止、終了許可
        if not is_root_level and ctx.state == FSMState.KEY and ctx.depth == 2:
            if len(self._expected_param_keys) == 0:
                return {"}", " ", "\n", "\t", "\r"}

        return allowed_chars

    def _filter_structural_tokens(
        self, ctx: ParsedContext,
        valid_token_ids: set[int], allowed_chars: set[str], current_text: str
    ) -> set[int]:
        """
            文字列外の時、不要なtoken排除、型制約の適用。
            必要な構文文字のtoken_idのみ許可する。
            現在メインは数字の出力。
        """
        # 許可されていない構造文字セット
        dis_allowed_structural = (
            {'"', '{', '}', '[', ']', ':', ','} - allowed_chars
        )
        clean_valid_ids = set()

        # valueかつ、引数の中身の時、スキーマの型を取得
        param_type = None
        current_val_str = ""
        is_root_level = (ctx.depth == 1)

        # parameters 内部(引数の中身)の時
        if not is_root_level and ctx.state == FSMState.VALUE and (
                ctx.depth == 2):
            current_schema = self._get_current_function_schema()
            param_def = current_schema.get(ctx.last_key.strip())
            param_type = param_def.type.lower() if param_def else "string"
            val_match = re.search(r':([^:]*)$', current_text)
            if val_match:
                current_val_str = val_match.group(1).strip(' \n\r\tĊ')

        for t_id in valid_token_ids:
            t_str = self._tokenizer._id_to_token.get(t_id, "")
            # 1. 許可外の構造文字を含むtokenを排除
            if any(c in t_str for c in dis_allowed_structural):
                continue
            # 2. 文字列外でのalphabet混入tokenを除外
            if ctx.state in (
                FSMState.COMMA_OR_END,
                FSMState.COLON,
                FSMState.BEGIN
            ):
                # 空白、改行以外のa~Zを弾く
                if re.search(r'[a-zA-Z]', t_str):
                    continue
            if param_type:
                clean_str = t_str.replace("Ġ", " ").strip(' \n\r\tĊ')

                # "parameters": {"type": "string"}の時
                if param_type in ("string", "str"):
                    if '"' not in t_str and clean_str:
                        continue

                # "parameters": {"type": "number"}の時
                elif param_type in ("number", "num", "integer", "int"):

                    allowed_num = allowed_chars - {"t", "f", "n", '"'}
                    if '"' in t_str and clean_str:
                        continue

                    # "parameters": {"type": "int"}の時
                    if param_type in ("integer", "int"):
                        allowed_num = allowed_num - {'.'}

                    raw_prompt = getattr(self, "_raw_user_prompt", "")
                    force_terminate = False

                    if current_val_str:
                        # 生成中の数字がプロンプト内に存在するか
                        is_in_prompt = current_val_str in raw_prompt
                        # 小数点がある時、小数点以下の長さを計測
                        fraction_len = (
                            len(current_val_str.split(".")[1])
                            if "." in current_val_str
                            else 0)
                        # 出力中の数字がプロンプトに存在しない
                        # かつ許容する長さを超えた時
                        if len(current_val_str) > 15 or fraction_len > 5:
                            if not is_in_prompt:
                                force_terminate = True
                            # 生成中の数字とプロンプト中の数字が合致した時
                            prompt_numbers = (
                                re.findall(r'\d+(?:\.\d+)?', raw_prompt))
                            for p_num in prompt_numbers:
                                if current_val_str == p_num:
                                    force_terminate = True
                    # 強制終了
                    if force_terminate:
                        allowed_num = allowed_num - set("0123456789.eE+")

                    if not current_val_str:
                        allowed_num = allowed_num - {".", "e", 'E'}
                    if current_val_str and (
                            not current_val_str.endswith(('e', 'E'))):
                        allowed_num = allowed_num - {"-"}
                    if '.' in current_val_str or (
                            'e' in current_val_str.lower()):
                        allowed_num = allowed_num - {"."}
                    if 'e' in current_val_str.lower() or (
                            current_val_str and (
                                not current_val_str[-1].isdigit())):
                        allowed_num = allowed_num - {"e", "E"}
                    if current_val_str in ("0", "-0"):
                        allowed_num = allowed_num - set("0123456789")
                    if clean_str:
                        is_valid_token = True
                        temp_allowed = set(allowed_num)
                        for c in clean_str:
                            if c not in temp_allowed:
                                is_valid_token = False
                                break
                            if c == ".":
                                temp_allowed = temp_allowed - {"."}
                            if c in ("e", "E"):
                                temp_allowed = temp_allowed - {"e", "E", "."}
                            if c in ("-"):
                                temp_allowed = temp_allowed - {"-"}
                        if not is_valid_token:
                            continue

                # "parameters": {"type": "boolean"}の時
                elif param_type in ("boolean", "bool"):
                    if "," in clean_str or "}" in clean_str:
                        clean_valid_ids.add(t_id)
                        continue
                    if '"' in t_str:
                        continue
                    if clean_str:
                        new_val = current_val_str + clean_str
                        if not (
                            any(target.startswith(new_val) or new_val == target
                                for target in ("true", "false", "null"))):
                            continue

                # "parameters": {"type": "null"}の時
                elif param_type in ("null", "Null", "NULL"):
                    if "," in clean_str or "}" in clean_str:
                        clean_valid_ids.add(t_id)
                        continue
                    if '"' in t_str:
                        continue
                    if clean_str:
                        new_val = current_val_str + clean_str
                        if not "null".startswith(new_val):
                            continue

            clean_valid_ids.add(t_id)

        return clean_valid_ids

    def _filter_in_string_tokens(self, ctx: ParsedContext) -> set[int]:
        """
            文字列内の時、不要なtoken排除。
            関数名、引数の中身がstringの時のメイン処理。
        """
        valid_token_ids = set()
        target_strings: list[str] = []
        is_fn_name_context = False
        param_type = None

        # 2-1. 2-3. root key(name, parameters), parameter keyの処理
        if (ctx.depth == 1 or ctx.depth == 2) and not ctx.is_value_context:
            # 2-3-1. root key または最適関数の"parameters"内のkey(引数名)
            target_strings = (
                self._available_root_keys
                if ctx.depth == 1
                else list(
                    set(self._expected_param_keys) - self._seen_param_keys)
            )
            if not target_strings:
                return set()

        # 2-2. nameのvalue(関数名)の時
        elif ctx.depth == 1 and (
                ctx.is_value_context and ctx.last_key == "name"):
            is_fn_name_context = True
            target_strings = (
                [self._target_fn_name]
                if self._target_fn_name
                else self._valid_fn_names
            )

        # 2-4. parameterのvalueかつ引数の中身の時
        elif ctx.depth == 2 and ctx.is_value_context:
            # 現在の最適関数のschemaを取得
            current_schema = self._get_current_function_schema()
            # 入力中のkeyの情報を取得 デフォルトでstr型に設定
            param_def = current_schema.get(ctx.last_key.strip())
            param_type = param_def.type.lower() if param_def else "string"

        # 2-5. 語彙からid, strをループ
        for t_id, t_str in self._vocab_items:
            clean_str = t_str.replace("Ġ", " ")
            new_str = ctx.current_string + clean_str

            # 2-5-1. keyまたは関数名の入力中
            if target_strings or is_fn_name_context:
                if (
                    any((k.startswith(new_str)) or (new_str == k + '"')
                        for k in target_strings)):
                    valid_token_ids.add(t_id)

            # 2-5-2. "parameters": {"type": "string"}の時
            elif param_type in ("string", "str", "String", "Str"):
                # 文字列内の改行等JSONが壊れるtokenをブロック
                if "\n" in t_str or "\r" in t_str:
                    continue

                if '"' in clean_str:
                    # "より後ろの文字をチェック
                    after_quote = clean_str.split('"', 1)[1].strip()
                    if after_quote:
                        is_last_param = (
                            len(self._seen_param_keys) == len(
                                self._expected_param_keys))
                        # 残りの引数が無い時、継続禁止
                        if is_last_param and ',' in after_quote:
                            continue
                        # 残り引数がある時、終了禁止
                        if not is_last_param and '}' in after_quote:
                            continue

                    if '\\"' in clean_str or ctx.current_string.endswith('\\'):
                        valid_token_ids.add(t_id)
                        continue
                    current_val = ctx.current_string
                    unescaped_val = (
                        current_val.replace('\\"', '"').replace('\\\\', '\\'))
                    raw_prompt = getattr(self, "_raw_user_prompt", "")

                    if (len(unescaped_val) > 150
                            and unescaped_val not in raw_prompt):
                        if clean_str.strip() == '"':
                            valid_token_ids.add(t_id)
                        continue

                    valid_token_ids.add(t_id)
                    continue
                valid_token_ids.add(t_id)

            # 想定外のコンテキスト 基本入らないケース
            else:
                pass
        return valid_token_ids

    def filter_logits(
        self, logits: list[float], current_text: str
    ) -> list[float]:
        """
            制約付きデコーディングのメインロジック。
            状態遷移(FSM)アルゴリズム、 プッシュダウンオートマトン(PDA)。
        """
        try:
            # --------------- 強制キューによるバイアス ---------------
            if self._forced_queue:
                target_id = self._forced_queue.pop(0)
                filtered_logits = [-math.inf] * len(logits)
                filtered_logits[target_id] = 100.0
                return filtered_logits

            # current_textを解析
            ctx = self.state_tracker.determine_current_state(current_text)
            self._seen_root_keys = set(ctx.seen_root_keys)
            clean_text = current_text.strip()
            # parameters内、引数名の出現状況を適宜解析、更新
            param_match = re.search(
                r'"parameters"\s*:\s*\{(.*)', current_text, re.DOTALL)
            if param_match:
                self._seen_param_keys = set(
                    re.findall(r'"([^"]+)"\s*:', param_match.group(1)))

            # -------------------- 必須構文の処理 --------------------
            # 1. 関数名の出力が完了した時 -> 引数までキューに補充
            if (ctx.depth == 1
                    and not ctx.in_string
                    and ctx.state == FSMState.COMMA_OR_END
                    and ctx.last_key == "name"
                    and clean_text.endswith('"')):
                # 実際にモデルが選択した関数名に合わせてtarget, param keyを更新
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', current_text)
                if name_match:
                    self._target_fn_name = name_match.group(1)
                    self._expected_param_keys = (
                        self._get_expected_param_keys())
                if self._expected_param_keys:
                    first_arg = self._expected_param_keys[0]
                    inject_str = f',\n  "parameters": {{\n    "{first_arg}": '
                else:
                    inject_str = ',\n  "parameters": {{}}\n}}: '

                self._forced_queue = self._tokenizer.encode(inject_str)
                # 再帰してqueueを消費
                return self.filter_logits(logits, current_text)

            # 2. 引数の値を出力後まだkeyが残ってる時 -> 次の引数を補充
            if (ctx.depth == 2
                    and not ctx.in_string
                    and clean_text.endswith(',')):
                if len(self._seen_param_keys) < len(self._expected_param_keys):
                    self._param_index += 1
                    next_arg = self._expected_param_keys[self._param_index]
                    inject_str = f'    "{next_arg}": '
                    self._forced_queue = self._tokenizer.encode(inject_str)
                    # 再帰してqueueを消費
                    return self.filter_logits(logits, current_text)

            # Ex. 空白tokenが暴走した時
            if (ctx.state == FSMState.COMMA_OR_END
                    and current_text.endswith('   ')):
                if len(self._seen_param_keys) < len(self._expected_param_keys):
                    self._param_index += 1
                    next_arg = self._expected_param_keys[self._param_index]
                    inject_str = f',\n    "{next_arg}": '
                    self._forced_queue = self._tokenizer.encode(inject_str)
                    return self.filter_logits(logits, current_text)
                else:
                    self._forced_queue = self._tokenizer.encode("\n}")
                    return self.filter_logits(logits, current_text)

            # 3. 引数の出力が完了、parametersが閉じられた時 -> "}"を強制
            if (ctx.depth == 1
                    and not ctx.in_string
                    and clean_text.endswith('}')):
                self._forced_queue = self._tokenizer.encode('\n}')
                return self.filter_logits(logits, current_text)

            # -------------------- 必須keyの処理 --------------------
            # token全ての確率を-infで初期化
            filtered_logits = [-math.inf] * len(logits)
            # root keyの順番の強制
            self._available_root_keys = self._get_available_root_keys()
            # 現在の状態に合わせて許容する文字を取得
            allowed_chars = (
                self.state_tracker.get_allowed_characters(ctx.state, ctx.depth)
            )
            # 状況に応じてallowed_charsを上書き(主に複数引数の処理)
            allowed_chars = self._apply_schema_constraints(ctx, allowed_chars)
            # 許可済み制御文字から空白、改行を抜いたもの
            allowed_structural = {c for c in allowed_chars if c.strip()}
            # 現在のtokenを予測するため許可tokenリストをループ前に初期化
            valid_token_ids = set()
            # 空白tokenの補充
            if " " in allowed_chars or "\n" in allowed_chars:
                valid_token_ids.update(self._trie.whitespace_token_ids)

            # --------------- 構文文字フィルタリング  ---------------
            # 構文文字('{', '}', ':', ','...)の処理時
            if not ctx.in_string:
                # Trieのハッシュマップから許可token_id群を取得
                for char in allowed_structural:
                    if char in self._trie.first_char_index:
                        valid_token_ids.update(
                            self._trie.first_char_index[char]
                        )
                # 現在の文字列に適した構文文字のtoken_idのみ許可
                # valueがnumberかboolの時
                valid_token_ids = self._filter_structural_tokens(
                    ctx, valid_token_ids, allowed_chars, current_text
                )

            # ---------- 文字列内部(関数名、引数の中身)の処理 ----------
            else:
                valid_token_ids = self._filter_in_string_tokens(ctx)

            # ------------------------- 最終調整 -------------------------

            # 許可したtoken_idがない時、暴走防止でフォールバック
            if not valid_token_ids:
                print("許可tokenなし")
                return logits

            # Logitsの再構築
            for t_id in valid_token_ids:
                filtered_logits[t_id] = logits[t_id]

                # 関数名入力時のtoken_idを強制
                if (ctx.depth == 1
                        and ctx.is_value_context
                        and ctx.last_key == "name"
                        and self._target_fn_name):
                    filtered_logits[t_id] += 10.0

            return filtered_logits

        except Exception as e:
            print(f"[\33[31mConstrainFilter\33[31m] Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # パイプラインのクラッシュ防止フォールバック
            return logits
