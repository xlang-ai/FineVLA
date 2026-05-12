"""
Automatic evaluation: compare VLM answer against ground truth.

Each function returns (correct: bool, parsed_answer: str).
"""

import re
from typing import Dict, Tuple


def eval_yes_no(model_answer: str, gt_answer: str) -> Tuple[bool, str]:
    """Extract yes/no from model answer and compare."""
    text = model_answer.strip().lower()
    # Try to find yes or no
    if text in ("yes", "no"):
        parsed = text
    elif text.startswith("yes"):
        parsed = "yes"
    elif text.startswith("no"):
        parsed = "no"
    else:
        # Search for yes/no in text
        yes_match = re.search(r'\byes\b', text)
        no_match = re.search(r'\bno\b', text)
        if yes_match and not no_match:
            parsed = "yes"
        elif no_match and not yes_match:
            parsed = "no"
        else:
            parsed = text[:20]  # unparseable, keep first 20 chars
    return parsed == gt_answer.strip().lower(), parsed


def eval_number(model_answer: str, gt_answer: str) -> Tuple[bool, str]:
    """Extract number from model answer and compare."""
    text = model_answer.strip()
    # Find first number in text
    match = re.search(r'\d+', text)
    if match:
        parsed = match.group()
    else:
        # Try word numbers
        word_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10",
        }
        parsed = None
        for word, num in word_map.items():
            if word in text.lower():
                parsed = num
                break
        if parsed is None:
            parsed = text[:20]

    gt_num = re.search(r'\d+', gt_answer.strip())
    gt_parsed = gt_num.group() if gt_num else gt_answer.strip()
    return parsed == gt_parsed, parsed


def eval_multiple_choice(
    model_answer: str, gt_letter: str, shuffled_options: list,
) -> Tuple[bool, str]:
    """Extract chosen option letter and compare with GT letter.

    Args:
        model_answer: raw model response
        gt_letter: correct answer letter after option shuffling (e.g. "C")
        shuffled_options: options in shuffled order
    """
    text = model_answer.strip()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Build letter → option text mapping
    letter_to_text = {}
    for i, opt in enumerate(shuffled_options):
        letter_to_text[letters[i]] = opt.strip()

    # Try to parse model answer as letter
    parsed_letter = None

    # Case 1: single letter
    first_char = text[0].upper() if text else ""
    if first_char in letter_to_text:
        if len(text) == 1 or not text[1].isalpha():
            parsed_letter = first_char

    # Case 2: search for "Answer: X" pattern
    if not parsed_letter:
        match = re.search(r'(?:answer|选择?)[:\s]*([A-Z])\b', text, re.IGNORECASE)
        if match:
            parsed_letter = match.group(1).upper()

    # Case 3: match option text
    if not parsed_letter:
        text_lower = text.lower()
        for i, opt in enumerate(shuffled_options):
            if opt.strip().lower() in text_lower:
                parsed_letter = letters[i]
                break

    if parsed_letter:
        parsed_display = letter_to_text.get(parsed_letter, parsed_letter)
    else:
        parsed_display = text[:30]

    correct = parsed_letter == gt_letter if (parsed_letter and gt_letter) else False
    return correct, parsed_display


def evaluate(model_answer: str, qa: dict, prompt_extra: dict = None) -> Tuple[bool, str]:
    """Dispatch to the right evaluator based on answer_type.

    Args:
        model_answer: raw model response
        qa: question dict with answer_type, answer, options, etc.
        prompt_extra: extra info from build_prompt (shuffled_options, gt_letter for MC)
    """
    answer_type = qa.get("answer_type", "")
    gt = qa["answer"]
    extra = prompt_extra or {}

    if answer_type == "yes_no":
        return eval_yes_no(model_answer, gt)
    elif answer_type == "number":
        return eval_number(model_answer, gt)
    elif answer_type == "multiple_choice":
        gt_letter = extra.get("gt_letter")
        shuffled_options = extra.get("shuffled_options", qa.get("options", []))
        if not gt_letter:
            # Fallback: find GT in shuffled options
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            gt_lower = gt.strip().lower()
            for i, opt in enumerate(shuffled_options):
                if opt.strip().lower() == gt_lower:
                    gt_letter = letters[i]
                    break
        return eval_multiple_choice(model_answer, gt_letter, shuffled_options)
    else:
        # Fallback: exact match
        parsed = model_answer.strip()
        return parsed.lower() == gt.strip().lower(), parsed
