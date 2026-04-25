import sys
import json
from typing import Any

from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints import ConstraintFilter


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
            input_ids = self._tokenizer.encode(prompt)
            generated_ids: list[int] = []
            current_text = ""

            for _ in range(self._max_new_tokens):
                current_sequence = input_ids + generated_ids
                logits = self._llm.get_logits(current_sequence)

                # 状態遷移(FSM) VS 正規表現(Regex)
                # 無効なトークンを選択されないように処理
                # 制約フィルタリング
                filtered_logits = self._constraint_filter.filter_logits(
                    logits=logits,
                    current_text=current_text
                )
                next_token_id = self._argmax(filtered_logits)
                generated_ids.append(next_token_id)

                # decode
                current_text = self._tokenizer.decode(generated_ids)

                # _max_tokensの上限になる前に有効なJsonオブジェクトが
                # 形成された時点でループ脱出 -> 計算資源を最適化
                if current_text.endswith("}"):
                    try:
                        parsed_json = json.loads(current_text)
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
            return {"error": str(e), "status": "failed"}
