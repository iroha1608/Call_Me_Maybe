import sys
import json
import re
from typing import Any
from time import sleep

from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints.filter import ConstraintFilter


class EngineError(Exception):
    """
    独自例外を送出する設計
    tokenを使い果たしたら送出

    """
    pass


class GenerationEngine:
    """
    生成パイプラインと制約付きデコーディングのメインループ
    """
    def __init__(
        self,
        llm_client: LLMClient,
        tokenizer: Tokenizer,
        constraint_filter: ConstraintFilter,
        max_new_tokens: int = 256
    ) -> None:
        self._llm = llm_client
        self._tokenizer = tokenizer
        self._constraint_filter = constraint_filter
        self._max_new_tokens = max_new_tokens

    def _argmax(self, logits: list[float]) -> int:
        if not logits:
            raise ValueError("Empty logits list provided to argmax.")
        return logits.index(max(logits))

    def generate(self, prompt: str) -> dict[str, Any]:
        """
        prompt -> token化 -> InputIDs -> LLM -> Logits
        -> 制約付きフィルタリング -> Token選択 -> decode
        自己回帰性ループ
        """
        try:
            input_ids: list[int] = self._tokenizer.encode(prompt)
            token_ids: list[int] = []
            current_text: str = ""

            for _ in range(self._max_new_tokens):
                current_sequence: list[int] = input_ids + token_ids
                logits: list[float] = self._llm.get_logits(current_sequence)

                # --------------- 制約前のLogitsの表示 ---------------
                print("\x1b[2J\x1b[H\x1b[s", end="")
                print(" --------------- "
                      "\33[1;5;34mBefore Logits\33[0m ----------------")
                top_logits = (sorted(range(len(logits)),
                              key=lambda i: logits[i], reverse=True)[:5])
                for idx in top_logits:
                    t_str = self._tokenizer.decode([idx]).replace('\n', '\\n')
                    score = logits[idx]
                    print(f"|ID: {idx:6} | "
                          f"Score: {score:7.2f} | Token '{t_str}'")
                print(" ---------------------------------------------\n")

                # 無効なトークンを選択されないように処理
                filtered_logits = self._constraint_filter.filter_logits(
                    logits=logits, current_text=current_text
                )
                next_token_id = self._argmax(filtered_logits)
                token_ids.append(next_token_id)
                # decode
                current_text = self._tokenizer.decode(token_ids)

                # --------------- 制約後のLogitsの表示 ---------------
                top_filtered_logits = (sorted(range(len(logits)),
                                       key=lambda i: filtered_logits[i],
                                       reverse=True)[:5])
                print(" --------------- "
                      "\33[1;5;35mAfter Logits\33[0m ---------------")
                for idx in top_filtered_logits:
                    t_str = self._tokenizer.decode([idx]).replace('\n', '\\n')
                    score = filtered_logits[idx]
                    print(f"|\33[31mID\33[0m: {idx:6} | "
                          f"\33[32mScore\33[0m: {score:7.2f} | "
                          f"\33[33mToken\33[0m: '{t_str}'")
                print(" ---------------------------------------------\n")
                print(f"current_text: \33[1;3m{current_text}\33[0m")

                # current_text -> clean_text
                c_text = current_text.replace("Ċ", "\n").replace("Ġ", " ")
                clean_text = c_text.strip()
                # clean_text = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', cl_text)

                # max_tokensの上限前にJsonが形成された時点でループ脱出
                if clean_text.endswith("}"):
                    try:
                        parsed_json = json.loads(clean_text)
                        if isinstance(parsed_json, dict):
                            return parsed_json
                    except json.JSONDecodeError:
                        pass
                sleep(0.2)

            c_text = current_text.replace("Ċ", "\n").replace("Ġ", " ")
            clean_text = c_text.strip()
            # clean_text = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', cl_text)
            try:
                parsed_json = json.loads(clean_text)
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                pass

            # _max_tokensを超えたらエラー送出
            raise EngineError(
                "Maximum token reached without generating valid JSON."
            )

        except Exception as e:
            print(f"EngineError: Generation pipeline failed."
                  f"{e}", file=sys.stderr)
            return {"name": "unknown", "parameters": {
                }}
