from typing import Tuple

def split_reasoning_traces(text: str, reasoning_token: str = "</think>") -> Tuple[str|None, str]:
    if reasoning_token in text:
        reasoning, answer = text.split(reasoning_token, 1)
        return reasoning.strip(), answer.strip()
    else:
        return None, text.strip()