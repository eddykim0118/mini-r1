"""
Stage 3, Step 2: the Countdown task + the verifier reward (the core of RLVR).

- generate_countdown()  : make a solvable (numbers, target) puzzle.
- make_prompt()         : format a puzzle as chat messages for the model.
- format_reward()       : small bonus for using the <think>/<answer> structure.
- correctness_reward()  : big reward if the <answer> equation is valid AND hits the target.

Security note: model output is UNTRUSTED. We never eval() it. Expressions are parsed with a
restricted AST evaluator that only understands numbers and + - * / — nothing else runs.

Run the self-tests:  uv run python stage3_capstone/countdown_task.py
"""

import ast
import operator
import random
import re

FORMAT_BONUS = 0.1      # well-formed <think>/<answer> tags
PARTIAL_REWARD = 0.2    # valid expression using exactly the right numbers, but wrong value
CORRECT_REWARD = 1.0    # expression actually hits the target (dominant, to avoid reward hacking)

SYSTEM_PROMPT = (
    "You are solving the Countdown number game. Given a list of numbers and a target, "
    "use each number exactly once with + - * / and parentheses to reach the target.\n"
    "First reason step by step inside <think> </think>, then give ONLY the final expression "
    "inside <answer> </answer>. Example: <answer> (3 + 5) * 2 </answer>."
)

# One worked example, shown as a completed turn. Small models imitate a *shown* format far
# better than they follow a *described* one -- this bootstraps format compliance (cold-start fix).
_EXAMPLE_USER = "Numbers: [2, 3, 5]\nTarget: 16"
_EXAMPLE_ASSISTANT = (
    "<think>I need 16 from 2, 3, 5. 3 + 5 = 8, and 8 * 2 = 16.</think>\n"
    "<answer>(3 + 5) * 2</answer>"
)


# ---- Task generation ----
def generate_countdown(n_numbers: int = 3, low: int = 1, high: int = 25,
                       target_low: int = 1, target_high: int = 100) -> tuple[list[int], int]:
    """Return (numbers, target) for a puzzle that is guaranteed to have a solution."""
    while True:
        numbers = [random.randint(low, high) for _ in range(n_numbers)]
        # Fold the numbers with random ops -> the result is reachable using each number once,
        # so a valid solution provably exists (with the right parentheses).
        value = numbers[0]
        for n in numbers[1:]:
            value = random.choice([value + n, value - n, value * n])
        if target_low <= value <= target_high:
            return sorted(numbers), value


def make_prompt(numbers: list[int], target: int) -> list[dict]:
    """Chat-format a puzzle, including one worked example so the model imitates the format."""
    user = f"Numbers: {numbers}\nTarget: {target}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _EXAMPLE_USER},
            {"role": "assistant", "content": _EXAMPLE_ASSISTANT},
            {"role": "user", "content": user}]


# ---- Safe arithmetic evaluation (never eval() untrusted text) ----
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}


def _safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression using only +,-,*,/ and unary minus. Raises on anything else."""
    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return _ev(ast.parse(expr, mode="eval"))


# ---- Parsing helpers ----
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/(). ]+$")


def _text(completion) -> str:
    """Handle both plain-string completions and conversational [{'role','content'}] ones."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def _extract_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def _score_one(text: str, numbers: list[int], target: int) -> float:
    answer = _extract_answer(text)
    if answer is None or not _ALLOWED_CHARS.match(answer):
        return 0.0
    used = [int(x) for x in re.findall(r"\d+", answer)]
    if sorted(used) != sorted(numbers):   # must use exactly the given numbers, once each
        return 0.0
    try:
        value = _safe_eval(answer)
    except Exception:
        return 0.0
    # Graded: right numbers in a valid expression earns partial credit even if the value is
    # off; hitting the target earns the dominant reward. This gives GRPO a gradient to climb.
    return CORRECT_REWARD if abs(value - target) < 1e-6 else PARTIAL_REWARD


# ---- Reward functions (TRL passes `completions` + dataset columns as kwargs) ----
def format_reward(completions, **kwargs) -> list[float]:
    return [FORMAT_BONUS if _FORMAT_RE.search(_text(c)) else 0.0 for c in completions]


def correctness_reward(completions, numbers, target, **kwargs) -> list[float]:
    return [_score_one(_text(c), n, t) for c, n, t in zip(completions, numbers, target)]


# ---- Self-tests: prove the verifier can't be fooled ----
if __name__ == "__main__":
    # Case 1: correct format + correct math using each number once -> 0.1 format, 1.0 correctness
    c1 = "<think>7*8=56, 56-2... let's see</think><answer>(3 + 5) * 2</answer>"
    assert format_reward([c1]) == [0.1]
    assert correctness_reward([c1], numbers=[[2, 3, 5]], target=[16]) == [1.0]

    # Case 2: right numbers, valid expr, WRONG value -> partial credit 0.2 (the new tier)
    c2 = "<think>guessing</think><answer>2 + 3 + 5</answer>"  # = 10, not 16, but uses 2,3,5 once
    assert format_reward([c2]) == [0.1]
    assert correctness_reward([c2], numbers=[[2, 3, 5]], target=[16]) == [0.2]

    # Case 3: right value but WRONG numbers (reuses 4, ignores given set) -> correctness no
    c3 = "<think>...</think><answer>4 * 4</answer>"  # = 16 but not the given numbers
    assert correctness_reward([c3], numbers=[[2, 3, 5]], target=[16]) == [0.0]

    # Case 4: no tags at all -> no format bonus, no correctness
    c4 = "The answer is (3 + 5) * 2 = 16"
    assert format_reward([c4]) == [0.0]
    assert correctness_reward([c4], numbers=[[2, 3, 5]], target=[16]) == [0.0]

    # Case 5: injection attempt -> safely rejected (no crash, no reward)
    c5 = "<think>hack</think><answer>__import__('os').system('echo pwned')</answer>"
    assert format_reward([c5]) == [0.1]           # it *looks* formatted...
    assert correctness_reward([c5], numbers=[[2, 3, 5]], target=[16]) == [0.0]  # ...but earns nothing

    # Case 6: the generator always produces solvable puzzles
    random.seed(0)
    for _ in range(1000):
        nums, tgt = generate_countdown()
        assert 1 <= tgt <= 100 and len(nums) == 3

    print("All verifier self-tests passed.")
    nums, tgt = generate_countdown()
    print(f"Sample puzzle -> Numbers: {nums}  Target: {tgt}")
