import sys
import math
import re
from typing import Any

from src.tokenizer import Tokenizer
from src.models import FunctionDefinition

from src.constraints.state import JSONStateTracker, FSMState, ParsedContext
from src.constraints.trie import TokenTrie


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
            list(tokenizer._id_to_token.items())
        )
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

        self.state_tracker = JSONStateTracker()

    def set_user_prompt(self, user_prompt: str) -> None:
        """
            推論ループ内でプロンプトごとに呼び出し、
            標的関数の再計算とFSMの差分キャッシュをリセットする。
        """
        # 最適な関数の取得
        self._target_fn_name = self._calculate_target_function(user_prompt)

        # 状態トラッカーの初期化、キャッシュのリセット
        self.state_tracker.reset()

        # プロンプトのtokenIDを保存
        self._prompt_token_ids = set(self._tokenizer.encode(user_prompt))
        # 生プロンプトの保存
        self._raw_user_prompt = user_prompt
        # 引用符('"', "'")で囲まれたフレーズの保存
        phrases = re.findall(r'"([^"]+)"|\'([^\']+)\'', user_prompt)
        self._quoted_phrases = [m[0] or m[1] for m in phrases]

    def _calculate_target_function(self, user_prompt: str) -> str:
        """
            ユーザープロンプトと関数定義間のJaccard係数を計算し、
            意味論的に最も関連性の高い関数名を利用可能にする
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

    def _get_current_function_schema(self) -> dict[str, dict[str, str]]:
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
            if _get_attr(fn, "name") == self._target_fn_name:
                params = _get_attr(fn, "parameters", {})
                if isinstance(params, dict):
                    return params
        return {}

    def _get_expected_param_keys(self, selected_fn: str) -> list[str]:
        """選択された関数のparameters一覧を取得"""
        # 使用関数からparametersを取得できた時、
        # parameters内のkey(変数名)を保存
        if selected_fn:
            for fn in self._available_functions:
                if _get_attr(fn, "name") == selected_fn:
                    params = _get_attr(fn, "parameters", {})
                    if isinstance(params, dict):
                        return list(params.keys())
        # 関数名が未確定、上記でkey(変数名)が取得できなかった時、
        # 全関数のkeyからparametersを取得、デッドロック回避
        all_keys: set[str] = set()
        for fn in self._available_functions:
            params = _get_attr(fn, "parameters", {})
            if isinstance(params, dict):
                all_keys.update(params.keys())
        return list(all_keys)

    def _get_available_root_keys(
        self, seen_root_keys: frozenset[str]
    ) -> list[str]:
        """root keyの順番を強制("name"->"parameters")"""
        if "name" not in seen_root_keys:
            return ["name"]
        elif "parameters" not in seen_root_keys:
            return ["parameters"]
        else:
            return []

    def _apply_schema_constraints(
        self,
        ctx: ParsedContext,
        allowed_chars: set[str],
        expected_param_keys: list[str],
        seen_param_keys: set[str]
    ) -> set[str]:
        """進行状況に応じて、許可する文字を上書き"""
        is_root_level = (ctx.depth == 1)

        # root, keyを一つ生成後、"name", "parameters"の両方を強制
        if ctx.state in (FSMState.COMMA_OR_END, FSMState.VALUE):
            allowed_chars = allowed_chars - {"]"}
            if is_root_level:
                # keyが"name"しか出てない時、継続許可、終了禁止
                if "name" in ctx.seen_root_keys and (
                        "parameters" not in ctx.seen_root_keys):
                    allowed_chars = allowed_chars - {"}"}
                    if ctx.state == FSMState.COMMA_OR_END:
                        return {",", " ", "\n", "\t", "\r"}
                # "name", "parameter"が出た時、継続禁止、終了許可
                elif "name" in ctx.seen_root_keys and (
                        "parameters" in ctx.seen_root_keys):
                    allowed_chars = allowed_chars - {","}
                    if ctx.state == FSMState.COMMA_OR_END:
                        return {"}", " ", "\n", "\t", "\r"}
            elif ctx.depth == 2:
                # "parameters"のkeyが残ってる時、継続許可、終了禁止
                if len(seen_param_keys) < len(expected_param_keys):
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
            if len(expected_param_keys) == 0:
                return {"}", " ", "\n", "\t", "\r"}

        return allowed_chars

    def _filter_structural_tokens(
        self, ctx: ParsedContext,
        valid_token_ids: set[int], allowed_chars: set[str], current_text: str
    ) -> set[int]:
        """
            文字列外の時、不要なtoken排除、型制約の適用
            必要な構文文字のtoken_idのみ許可する
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
            clean_key = ctx.last_key.strip()
            param_type = current_schema.get(
                clean_key, {}).get("type", "string").lower()
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
                elif param_type in ("number", "num"):
                    allowed_num = allowed_chars - {"t", "f", "n", '"'}
                    if '"' in t_str and clean_str:
                        continue
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
                    # 無限ループ防止
                    if len(current_val_str) > 15 and (
                            not any(c in t_str for c in ".}")):
                        continue

                # "parameters": {"type": "boolean"}の時
                elif param_type in ("boolean", "bool"):
                    if '"' in t_str:
                        continue
                    if clean_str:
                        new_val = current_val_str + clean_str
                        if not (
                            any(target.startswith(new_val) or new_val == target
                                for target in ("true", "false", "null"))):
                            continue
            clean_valid_ids.add(t_id)

        return clean_valid_ids

    def _filter_expected_keys(
        self, ctx: ParsedContext,
        valid_token_ids: set[int], available_root_keys: list[str],
        expected_param_keys: list[str], seen_param_keys: set[str]
    ) -> set[int]:
        """keyの処理時、スキーマで期待されるkeyの前方一致のみ許可"""
        if ctx.state != FSMState.KEY or ctx.depth not in (1, 2):
            return valid_token_ids

        # root key(name, parameters)、またはparameters内のkey(変数名)が入る。
        keys_to_check = (
            available_root_keys
            if ctx.depth == 1
            else list(set(expected_param_keys) - seen_param_keys)
        )
        if not keys_to_check:
            return valid_token_ids

        filtered = set()
        for t_id in valid_token_ids:
            t_str = (
                self._tokenizer._id_to_token.get(t_id, "").replace("Ġ", " ")
            )
            if '"' in t_str:
                content = t_str[t_str.find('"') + 1:]
                # '"'の開始、または期待されるkey名の前方、完全一致のみ許可
                if not content or (
                        any((k.startswith(content)) or (content == k + '"')
                            for k in keys_to_check)):
                    filtered.add(t_id)
            else:
                # 引用符を含まない純粋な空白tokenのみ許可
                if not t_str.strip(' \n\r\tĊ'):
                    filtered.add(t_id)

        return filtered

    def _filter_target_fn_name(
        self, ctx: ParsedContext, valid_token_ids: set[int]
    ) -> set[int]:
        """ nameのvalue(関数名)の直前の時"""
        if not (
            ctx.depth == 1
            and ctx.state == FSMState.VALUE
            and ctx.last_key == "name"
        ):
            return valid_token_ids

        filtered = set()
        for t_id in valid_token_ids:
            t_str = (
                self._tokenizer._id_to_token.get(t_id, "").replace("Ġ", " ")
            )
            if '"' in t_str:
                content = t_str[t_str.find('"') + 1:]
                if not content:
                    filtered.add(t_id)
                elif self._target_fn_name:
                    # targetの前方、完全一致を許可
                    if (
                        self._target_fn_name.startswith(content)
                        or content == self._target_fn_name + '"'
                    ):
                        filtered.add(t_id)
                # targetが未定の時全関数名を許可
                else:
                    if (
                        any(fn.startswith(content) or content == fn + '"'
                            for fn in self._valid_fn_names)):
                        filtered.add(t_id)
            else:
                # 引用符を含まない時、純粋な空白tokenのみ許可
                if not t_str.strip(' \n\r\tĊ'):
                    filtered.add(t_id)

        return filtered

    def _filter_in_string_tokens(
        self, ctx: ParsedContext, expected_param_keys: list[str],
        seen_param_keys: set[str], available_root_keys: list[str]
    ) -> set[int]:

        valid_token_ids = set()
        target_strings = []
        is_fn_name_context = False
        param_type = None

        # 2-1. 2-3. root key(name, parameters), parameter keyの処理
        if (ctx.depth == 1 or ctx.depth == 2) and not ctx.is_value_context:
            # 2-3-1. root key または最適関数の"parameters"内のkey(引数名)
            target_strings = (
                available_root_keys
                if ctx.depth == 1
                else list(set(expected_param_keys) - seen_param_keys)
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
            param_type = (
                current_schema.get(
                    ctx.last_key, {}).get("type", "string").lower()
            )

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

            # "parameters": {"type": "string"}の時
            elif param_type in ("string", "str", "String", "Str"):
                if "\n" in t_str or "\r" in t_str:
                    continue
                valid_token_ids.add(t_id)

                # 生成中の文字列がプロンプト中の引用符と一致
                # is_exact_match = False
                # 生成中の文字列がプロンプトの引用符のprefix
                # is_prefix_of_phrase = False
                for phrase in getattr(self, "_quoted_phrases", []):
                    if ctx.current_string == phrase:
                        # is_exact_match = True
                        pass
                    elif (
                        phrase.startswith(ctx.current_string)
                        and (ctx.current_string != phrase)
                        and (ctx.current_string != "")
                    ):
                        pass
                        # is_prefix_of_phrase = True
                # Semantic Logit Steering(確率ブースト)
                if getattr(self, "_raw_user_prompt", ""):
                    # 生成中の文字列がプロンプトと合致
                    if new_str in self._raw_user_prompt:
                        self.p_aligned_t_ids.add(t_id)
                    elif new_str.endswith('"'):
                        cont = new_str[:-1]
                        if cont in self._raw_user_prompt:
                            idx = self._raw_user_prompt.find(cont)
                            if idx != -1:
                                end_idx = idx + len(cont)
                                if end_idx == len(self._raw_user_prompt) or (
                                        not (self._raw_user_prompt[end_idx])
                                        .isalnum()
                                        ):
                                    self.p_aligned_t_ids.add(t_id)

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
            # ソフト制約 "parameter"のvalueかつ引数の中身の時使用
            self.p_aligned_t_ids: set[int] = set()
            # current_textを解析
            ctx = self.state_tracker.determine_current_state(current_text)
            # token全ての確率を-infで初期化(ハルシネーション防止)
            filtered_logits = [-math.inf] * len(logits)
            # 現在の状態に合わせて許容する文字を取得
            allowed_chars = (
                self.state_tracker.get_allowed_characters(
                    ctx.state, ctx.depth
                )
            )

            # -------------------- 必須keyの処理 start --------------------
            # 選択された関数名の取得
            selected_fn = self._target_fn_name
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', current_text)
            if name_match:
                selected_fn = name_match.group(1)

            # 使用関数からparametersを取得、parameters内のkey(変数名)を保存
            expected_param_keys = self._get_expected_param_keys(selected_fn)
            # parametersの出現状況を解析、更新
            seen_param_keys = set()
            param_match = re.search(
                r'"parameters"\s*:\s*\{(.*)', current_text, re.DOTALL)
            if param_match:
                seen_param_keys = set(
                    re.findall(r'"([^"]+)"\s*:', param_match.group(1)))

            # root keyの順番の強制
            available_root_keys = (
                self._get_available_root_keys(ctx.seen_root_keys)
            )
            # 状況に応じてallowed_charsを上書き
            allowed_chars = self._apply_schema_constraints(
                    ctx, allowed_chars, expected_param_keys, seen_param_keys
            )
            # -------------------- 必須keyの処理 end --------------------

            # 許可済み制御文字から空白を抜いたもの
            allowed_structural = {c for c in allowed_chars if c.strip()}
            # 現在のtokenを予測するため許可tokenリストをループ前に初期化
            valid_token_ids = set()
            # 空白token(無害)の補充
            if " " in allowed_chars or "\n" in allowed_chars:
                valid_token_ids.update(self._trie.whitespace_token_ids)

            # ---------- Trie木を用いた 語彙フィルタリング ----------
            # 1. 構文文字(':', ',', '{}'etc...)の待機時
            if not ctx.in_string:
                # 1-1. Trieのハッシュマップから許可token_id群を取得
                for char in allowed_structural:
                    if char in self._trie.first_char_index:
                        valid_token_ids.update(
                            self._trie.first_char_index[char]
                        )
                # 1-2. cleanにした構文文字のtoken_idのみ許可
                valid_token_ids = self._filter_structural_tokens(
                    ctx, valid_token_ids, allowed_chars, current_text
                )
                # 1-3. keyのフィルタリング
                valid_token_ids = self._filter_expected_keys(
                    ctx, valid_token_ids, available_root_keys,
                    expected_param_keys, seen_param_keys
                )
                # 1-4. 関数名のフィルタリング
                valid_token_ids = self._filter_target_fn_name(
                    ctx, valid_token_ids
                )
                # 3. DONEに到達時、eos_tokenの確率を引き上げ
                if ctx.state == FSMState.DONE:
                    # 既存許可リストをクリア
                    valid_token_ids = set()
                    # Engineが停止条件として認めるEOS tokenのみ許可する
                    eos_id = getattr(self._tokenizer, "eos_token_id", None)
                    if eos_id is not None:
                        eos_id = (
                            eos_id[0] if isinstance(eos_id, list) else eos_id
                        )
                        valid_token_ids.add(eos_id)
                    else:
                        valid_token_ids.update(self._trie.whitespace_token_ids)

            # 2. 文字列内部の処理
            else:
                valid_token_ids = self._filter_in_string_tokens(
                    ctx, expected_param_keys, seen_param_keys,
                    available_root_keys
                )

            # 許可したtoken_idがない時、暴走防止
            if not valid_token_ids:
                print("許可tokenなし")
                return logits

            # Logitsの再構築とSemantic Logit Steering
            for t_id in valid_token_ids:
                filtered_logits[t_id] = logits[t_id]

                # name keyに対する関数のtokenに加算
                if (ctx.depth == 1
                        and ctx.is_value_context
                        and ctx.last_key == "name"
                        and self._target_fn_name):
                    filtered_logits[t_id] += 10.0

                # prompt一致のtokenに加算
                if self.p_aligned_t_ids and t_id in self.p_aligned_t_ids:
                    filtered_logits[t_id] += 15.0

                # 終了tokenに加算
                if ctx.state == FSMState.DONE:
                    filtered_logits[t_id] += 100.0

            return filtered_logits

        except Exception as e:
            print(f"ConstrainFilter: Error during logit filtering. "
                  f"{e}", file=sys.stderr)
            # パイプラインのクラッシュを防止
            return logits
