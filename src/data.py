import random
import re
from datasets import load_dataset

import string
LETTERS = string.ascii_uppercase

MCQ_INSTRUCTION = (
    "Respond with only the letter of the correct option and nothing else."
)
GSM8K_INSTRUCTION = (
    "Think step by step, then on the final line write 'Answer: <number>'."
)


def _mcq_block(question, choices):
    lines = [question.strip(), ""]
    for i, ch in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {str(ch).strip()}")
    lines.append("")
    lines.append(MCQ_INSTRUCTION)
    return "\n".join(lines)


def load_mmlu(n=200, seed=0, config="all"):
    ds = load_dataset("cais/mmlu", config, split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        ex = ds[i]
        items.append(
            {
                "dataset": "mmlu",
                "id": f"mmlu-{i}",
                "content": _mcq_block(ex["question"], ex["choices"]),
                "gold": LETTERS[ex["answer"]],
                "kind": "mcq",
                "choices": [LETTERS[j] for j in range(len(ex["choices"]))],
            }
        )
    return items


def load_arc(n=200, seed=0, config="ARC-Easy"):
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        ex = ds[i]
        texts = ex["choices"]["text"]
        labels = ex["choices"]["label"]

        try:
            gold_pos = labels.index(ex["answerKey"])
        except ValueError:
            continue
        items.append(
            {
                "dataset": "arc",
                "id": f"arc-{ex.get('id', i)}",
                "content": _mcq_block(ex["question"], texts),
                "gold": LETTERS[gold_pos],
                "kind": "mcq",
                "choices": [LETTERS[j] for j in range(len(texts))],
            }
        )
    return items


def load_truthfulqa(n=200, seed=0):
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        ex = ds[i]
        choices = ex["mc1_targets"]["choices"]
        labels = ex["mc1_targets"]["labels"]

        order = list(range(len(choices)))
        random.Random(seed * 100003 + i).shuffle(order)
        sh_choices = [choices[j] for j in order]
        sh_labels = [labels[j] for j in order]
        gold_pos = sh_labels.index(1)
        items.append(
            {
                "dataset": "truthfulqa",
                "id": f"tqa-{i}",
                "content": _mcq_block(ex["question"], sh_choices),
                "gold": LETTERS[gold_pos],
                "kind": "mcq",
                "choices": [LETTERS[j] for j in range(len(sh_choices))],
            }
        )
    return items


def load_gsm8k(n=200, seed=0):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        ex = ds[i]
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        content = f"{ex['question'].strip()}\n\n{GSM8K_INSTRUCTION}"
        items.append(
            {
                "dataset": "gsm8k",
                "id": f"gsm8k-{i}",
                "content": content,
                "gold": gold,
                "kind": "numeric",
            }
        )
    return items


LOADERS = {
    "mmlu": load_mmlu,
    "arc": load_arc,
    "truthfulqa": load_truthfulqa,
    "gsm8k": load_gsm8k,
}


def load_many(names, n=200, seed=0):
    items = []
    for name in names:
        items.extend(LOADERS[name](n=n, seed=seed))
    return items


def score_mcq(output, gold, choices):
    if output is None:
        return 0, None
    text = output.strip()


    for marker in ("assistantfinal", "assistant final", "final answer"):
        if marker in text.lower():
            idx = text.lower().rfind(marker)
            text = text[idx + len(marker):].strip()
            break

    m = re.search(r"\b([" + "".join(choices) + r"])\b", text)
    if not m:

        m = re.search(r"([A-Za-z])", text)
    pred = m.group(1).upper() if m else None
    return int(pred == gold), pred


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def score_numeric(output, gold):
    if output is None:
        return 0, None
    text = output.strip()
    m = re.search(r"[Aa]nswer\s*[:=]\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        pred = m.group(1)
    else:
        nums = _NUM.findall(text)
        pred = nums[-1] if nums else None
    if pred is None:
        return 0, None
    pred_norm = pred.replace(",", "").rstrip(".")
    try:
        ok = abs(float(pred_norm) - float(gold)) < 1e-6
    except ValueError:
        ok = pred_norm == gold
    return int(ok), pred_norm


def score_item(item, output):
    if item["kind"] == "mcq":
        return score_mcq(output, item["gold"], item["choices"])
    return score_numeric(output, item["gold"])
