"""
Prompt templates for VQA evaluation.

build_batch_prompt: combines all questions for a sample into one prompt,
asks model to return structured JSON answers.
"""

import random
from typing import Dict, List, Tuple

SYSTEM_PROMPT = (
    "You are an expert robot manipulation video analyst. "
    "You will be shown frames from a robot manipulation video with one or more camera views. "
    "Each view is labeled (e.g., [View: head_rgb], [View: left_wrist_rgb]). "
    "Use information from ALL views to answer the questions. "
    "Answer based ONLY on what you observe in the video frames. "
    "Be precise and concise."
)

BATCH_TEMPLATE = """\
Watch the video frames carefully and answer ALL of the following questions about this robot manipulation video.

{questions_block}

You MUST return your answers as a JSON object. Use the FULL question_id shown in brackets [] as the key (NOT the short Q1/Q2 label):
```json
{{
  "answers": {{
    "<full_question_id_in_brackets>": "<your_answer>",
    ...
  }}
}}
```

Rules:
- For yes_no questions: answer EXACTLY "yes" or "no"
- For number questions: answer with JUST a number (e.g. "3")
- For multiple_choice questions: answer with JUST the option letter (e.g. "B")
- Do NOT include explanations in the JSON values — only the short answer
- You MUST answer ALL questions"""


def _shuffle_options(qa: dict) -> Tuple[List[str], str]:
    """Shuffle MC options deterministically, return (shuffled_options, gt_letter)."""
    options = qa.get("options", [])
    gt_answer = qa.get("answer", "")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    qid = qa.get("question_id", "")
    shuffled = list(options)
    random.Random(qid).shuffle(shuffled)

    gt_lower = gt_answer.strip().lower()
    gt_letter = None
    for i, opt in enumerate(shuffled):
        if opt.strip().lower() == gt_lower:
            gt_letter = letters[i]
            break

    return shuffled, gt_letter


def build_batch_prompt(qas: List[Dict]) -> Tuple[str, Dict[str, Dict]]:
    """Build a single prompt containing all questions for one sample.

    Args:
        qas: list of QA dicts for one sample

    Returns:
        (prompt_text, extras_by_qid)
        extras_by_qid: {question_id: {"shuffled_options": [...], "gt_letter": "C"}} for MC questions
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    question_blocks = []
    extras_by_qid = {}

    for i, qa in enumerate(qas, 1):
        qid = qa["question_id"]
        qtype = qa["answer_type"]
        question = qa["question"]

        if qtype == "yes_no":
            block = f"Q{i}. [{qid}] (yes/no)\n{question}"

        elif qtype == "number":
            block = f"Q{i}. [{qid}] (number)\n{question}"

        elif qtype == "multiple_choice":
            shuffled, gt_letter = _shuffle_options(qa)
            options_text = "  ".join(f"{letters[j]}. {opt}" for j, opt in enumerate(shuffled))
            block = f"Q{i}. [{qid}] (multiple_choice)\n{question}\nOptions: {options_text}"
            extras_by_qid[qid] = {
                "shuffled_options": shuffled,
                "gt_letter": gt_letter,
            }
        else:
            block = f"Q{i}. [{qid}]\n{question}"

        question_blocks.append(block)

    questions_block = "\n\n".join(question_blocks)
    prompt = BATCH_TEMPLATE.format(questions_block=questions_block)
    return prompt, extras_by_qid
