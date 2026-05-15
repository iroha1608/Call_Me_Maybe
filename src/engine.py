"""Text generation engine with constraint filtering."""

import json
from typing import Any
from time import sleep

from src.llm_client import LLMClient
from src.tokenizer import Tokenizer
from src.constraints.filter import ConstraintFilter


class EngineError(Exception):
    """
        Custom exception for errors
        that occur during the text generation process.
    """
    pass


class GenerationEngine:
    """
       Generateed text is expected to be a JSON string
       that can be parsed into a dictionary.
       Args:
           llm_client (LLMClient):
               The language model client.
           tokenizer (Tokenizer):
               The tokenizer for encoding and decoding tokens.
           constraint_filter (ConstraintFilter):
               The filter for applying constraints to the generated tokens.
           max_new_tokens (int):
               The maximum number of new tokens to generate.
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
        """
            Get the index of the maximum value in the logits list.
            Args:
                logits (list[float]):
                    A list of logits corresponding to token probabilities.
            Returns:
                int: The index of the token with the highest logit score.
            Raises:
                EngineError: If the logits list is empty.
        """
        if not logits:
            raise EngineError("Empty logits list provided to argmax.")
        return logits.index(max(logits))

    def generate(self, prompt: str) -> dict[str, Any]:
        """
            Generate text based on the given prompt, applying constraints
            to the token generation process. The generated text is expected
            to be a JSON string that can be parsed into a dictionary.
            Args:
                prompt (str): The input prompt to guide the text generation.
            Returns:
                dict[str, Any]: The parsed JSON object resulting from the
                                generated text.
            Raises:
                EngineError: If the maximum number of tokens is reached without
                        generating valid JSON, or if the resulting text cannot
                        be parsed as JSON.
        """
        input_ids: list[int] = self._tokenizer.encode(prompt)
        token_ids: list[int] = []
        current_text: str = ""

        for _ in range(self._max_new_tokens):
            current_sequence: list[int] = input_ids + token_ids
            logits: list[float] = self._llm.get_logits(current_sequence)

            # ------------------- 制約前のLogitsの表示 -------------------
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
            print(" ---------------------------------------------- \n")

            # ------------------- Constraints Decoding -------------------
            filtered_logits = self._constraint_filter.filter_logits(
                logits=logits, current_text=current_text
            )
            next_token_id = self._argmax(filtered_logits)
            token_ids.append(next_token_id)
            current_text = self._tokenizer.decode(token_ids)

            # ------------------- 制約後のLogitsの表示 -------------------
            top_filtered_logits = (sorted(range(len(logits)),
                                   key=lambda i: filtered_logits[i],
                                   reverse=True)[:5])
            print(" ---------------- "
                  "\33[1;5;35mAfter Logits\33[0m ----------------")
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

            # max_tokensの上限前にJsonが形成された時点でループ脱出
            if clean_text.endswith("}"):
                try:
                    parsed_json = json.loads(clean_text)
                    if isinstance(parsed_json, dict):
                        return parsed_json
                except json.JSONDecodeError:
                    # まだ途中の可能性があるので継続
                    pass
            sleep(0.2)

        c_text = current_text.replace("Ċ", "\n").replace("Ġ", " ")
        clean_text = c_text.strip()
        try:
            parsed_json = json.loads(clean_text)
            if isinstance(parsed_json, dict):
                return parsed_json
        except json.JSONDecodeError:
            pass

        # _max_tokensを超え、有効なJSONじゃなければエラー
        raise EngineError(
            f"Maximum tokens ({self._max_new_tokens})reached without "
            "generating valid JSON. Resulting text: {clean_text}")
