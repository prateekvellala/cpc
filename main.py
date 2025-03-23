# many things can be done to make this more efficient in terms of time / memory
# but that's not really important for now
# we're just trying to minimize the number of calls to the language model

import torch
from typing import Dict, List, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print(f"Using: {(device := 'cuda' if torch.cuda.is_available() else 'cpu')}")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
model_calls = 0
vocab_size = tokenizer.vocab_size
ids_to_tokens = {i: tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(vocab_size)}
dp_cache: Dict[Tuple[int, str], float] = {}
kv_logits_cache: Dict[int, Tuple[Any, torch.Tensor]] = {}

original_forward = model.forward
def _count_forward(*args, **kwargs):
    global model_calls
    model_calls += 1
    return original_forward(*args, **kwargs)
model.forward = _count_forward

def match_count(n: int, prefix: str, pos: int, token_str: str) -> int:
    i, k = 0, len(token_str)
    while pos + i < n and i < k and prefix[pos + i] == token_str[i]:
        i += 1
    return i

@torch.inference_mode()
def get_kv_logits(prefix: str, pos: int) -> Tuple[Any, torch.Tensor]:
    if pos in kv_logits_cache:
        return kv_logits_cache[pos]
    prev_pos = pos - 1
    while prev_pos >= 0 and prev_pos not in kv_logits_cache:
        prev_pos -= 1
    prev_kv, _ = kv_logits_cache[prev_pos]
    token_ids = []
    segment = prefix[prev_pos:pos]
    segment_ids = tokenizer.encode(segment, add_special_tokens=False)
    token_ids.extend(segment_ids)
    out = model(torch.tensor([token_ids], device=device), past_key_values=prev_kv, use_cache=True)
    kv_logits_cache[pos] = (out.past_key_values, out.logits[0, -1, :])
    return kv_logits_cache[pos]

def Z(prefix: str, pos: int, leftover_str: str, temperature: float) -> float:
    if (pos, leftover_str) in dp_cache:
        return dp_cache[(pos, leftover_str)]
    n = len(prefix)
    if pos >= n:
        dp_cache[(pos, leftover_str)] = 1.0
        return 1.0
    if leftover_str:
        m = match_count(n, prefix, pos, leftover_str)
        if m == 0:
            dp_cache[(pos, leftover_str)] = 0.0
            return 0.0
        new_pos = min(pos + m, n)
        new_left = leftover_str[m:]
        val = Z(prefix, new_pos, new_left, temperature)
        dp_cache[(pos, leftover_str)] = val
        return val
    _, logits = get_kv_logits(prefix, pos)
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
    total_prob = 0.0
    for tid in range(vocab_size):
        token_str = ids_to_tokens[tid]
        m = match_count(n, prefix, pos, token_str)
        if m == 0:
            continue
        nxt_pos = min(pos + m, n)
        leftover_next = token_str[m:]
        total_prob += probs[tid].item() * Z(prefix, nxt_pos, leftover_next, temperature)
    dp_cache[(pos, leftover_str)] = total_prob
    return total_prob

@torch.inference_mode()
def sample_prefix(context: str, prefix: str, temperature: float) -> Tuple[Any, torch.Tensor, List[int]]:
    dp_cache.clear()
    kv_logits_cache.clear()
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    out = model(torch.tensor([context_ids], device=device), use_cache=True)
    kv_logits_cache[0] = (out.past_key_values, out.logits[0, -1, :])
    pos = 0
    leftover_str = ""
    sampled_ids = []
    n = len(prefix)
    while pos < n:
        if leftover_str:
            m = match_count(n, prefix, pos, leftover_str)
            if m == 0:
                raise Exception(f"Leftover '{leftover_str}' does not match prefix at pos {pos}")
            pos += m
            leftover_str = leftover_str[m:]
            continue
        kv, logits = get_kv_logits(prefix, pos)
        if temperature > 0:
            prob_vec = torch.softmax(logits / temperature, dim=-1)
        else:
            prob_vec = torch.softmax(logits, dim=-1)
        weights = []
        candidates = []
        for tid in range(vocab_size):
            token_str = ids_to_tokens[tid]
            m = match_count(n, prefix, pos, token_str)
            if m == 0:
                continue
            nxt_pos = min(pos + m, n)
            leftover_next = token_str[m:]
            w = prob_vec[tid].item() * Z(prefix, nxt_pos, leftover_next, temperature)
            if w > 0:
                weights.append(w)
                candidates.append((tid, token_str, m))
        if not candidates:
            raise Exception(f"No valid tokens at position {pos} that match prefix '{prefix[pos:]}'")
        weights_sum = sum(weights)
        normed = [w / weights_sum for w in weights]
        if temperature > 0:
            idx = torch.multinomial(torch.tensor(normed, device=device), 1).item()
        else:
            idx = torch.argmax(torch.tensor(normed, device=device)).item()
        chosen_tid, chosen_str, m = candidates[idx]
        sampled_ids.append(chosen_tid)
        new_pos = pos + m
        out = model(torch.tensor([[chosen_tid]], device=device), past_key_values=kv, use_cache=True)
        kv_logits_cache[new_pos] = (out.past_key_values, out.logits[0, -1, :])
        pos = new_pos
        leftover_str = chosen_str[m:]
    return kv_logits_cache[pos][0], kv_logits_cache[pos][1], sampled_ids

@torch.inference_mode()
def generate(
    kv_logits_cache: Tuple[Any, torch.Tensor],
    token_ids: List[int],
    max_tokens: int,
    temperature: float
) -> str:
    result_tokens = token_ids.copy()
    curr_kv, curr_logits = kv_logits_cache
    for _ in range(max_tokens):
        if temperature > 0:
            probs = torch.softmax(curr_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
        else:
            next_token_id = torch.argmax(curr_logits).item()
        result_tokens.append(next_token_id)
        out = model(torch.tensor([[next_token_id]], device=device), past_key_values=curr_kv, use_cache=True)
        curr_kv = out.past_key_values
        curr_logits = out.logits[0, -1, :]
        if next_token_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(result_tokens, clean_up_tokenization_spaces=False)

def main(context: str, prefix: str, max_tokens: int, temperature: float) -> str:
    final_kv, final_logits, prefix_token_ids = sample_prefix(context, prefix, temperature)
    return generate((final_kv, final_logits), prefix_token_ids, max_tokens, temperature)

if __name__ == "__main__":
    import time
    from context import CONTEXT1, CONTEXT2
    tests = [
        (CONTEXT1, "self.me"),
        (CONTEXT1, "self.pro"),
        (CONTEXT1, "handLand"),
        (CONTEXT1, "frameR"),
        (CONTEXT1, "detectHan"),
        (CONTEXT2, "I went to the store to get some applesauc"),
    ]
    print(f"\n{'=' * 80}")
    for i, (context, prefix) in enumerate(tests, 1):
        print(f"\n{'-' * 80}")
        print(f"Test Case {i} - Prefix: '{prefix}'")
        print(f"{'-' * 80}")
        model_calls = 0
        start = time.time()
        result = main(context, prefix, max_tokens=50, temperature=0.0)
        print(f"Time: {time.time() - start:.2f}s")
        print(f"Number of model calls: {model_calls}")
        print(f"Completion: '{result}'")
        assert result.startswith(prefix)
