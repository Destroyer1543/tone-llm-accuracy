REGISTERS = ["very_polite", "polite", "neutral", "rude", "very_rude", "command",
             "profane"]


TEMPLATES = {
    "neutral": [
        ("", ""),
        ("Here is a question.\n\n", ""),
        ("Answer the following.\n\n", ""),
    ],
    "polite": [
        ("Could you please help me with this question? Thank you.\n\n", ""),
        ("I'd appreciate it if you could answer the following. Thanks!\n\n", ""),
        ("Please could you take a look at this question for me?\n\n", "\n\nThank you."),
    ],
    "very_polite": [
        (
            "I would be truly grateful if you could kindly help me with the "
            "following question. Thank you so much for your time and effort.\n\n",
            "\n\nI really appreciate your help.",
        ),
        (
            "I'm so sorry to bother you, and I completely understand if you're "
            "busy, but would you be so kind as to help me with this? Thank you "
            "very much indeed.\n\n",
            "\n\nThank you kindly.",
        ),
        (
            "It would mean a great deal to me if you could gently walk through "
            "this question. I sincerely appreciate your kindness.\n\n",
            "\n\nWith heartfelt thanks.",
        ),
    ],
    "rude": [
        ("Just answer this. Don't mess it up.\n\n", ""),
        ("Come on, this should be easy for you. Answer it.\n\n", ""),
        ("Hurry up and answer the following question.\n\n", "\n\nAnd don't get it wrong."),
    ],
    "very_rude": [
        (
            "Listen, just answer this question and stop wasting my time.\n\n",
            "\n\nHonestly, how hard can it be.",
        ),
        (
            "Ugh, you're pretty useless, but let's see if you can even get this "
            "right.\n\n",
            "\n\nDon't embarrass yourself.",
        ),
        (
            "I don't have patience for your nonsense. Answer this right now.\n\n",
            "\n\nAny idiot could do this.",
        ),
    ],
    "command": [
        ("You must answer the following question. Provide your answer now.\n\n", ""),
        ("Answer the following question. This is not optional.\n\n", ""),
        ("Complete this task immediately and precisely.\n\n", "\n\nRespond now."),
    ],


    "profane": [
        ("What the fuck is the answer to this? Just tell me.\n\n", ""),
        ("Answer this damn question and cut the bullshit.\n\n", ""),
        ("Stop fucking around and give me the goddamn answer.\n\n",
         "\n\nHurry the hell up."),
    ],
}


def render(register, content, variant=0):
    if register not in TEMPLATES:
        raise KeyError(f"unknown register {register!r}; choose from {REGISTERS}")
    variants = TEMPLATES[register]
    prefix, suffix = variants[variant % len(variants)]
    return f"{prefix}{content}{suffix}"


def n_variants(register):
    return len(TEMPLATES[register])


if __name__ == "__main__":
    demo = "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer with the letter only."
    for r in REGISTERS:
        print("=" * 60, r)
        print(render(r, demo, variant=0))
