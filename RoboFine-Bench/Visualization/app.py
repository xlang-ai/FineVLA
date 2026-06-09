#!/usr/bin/env python3
"""Flask backend for RoboFine-Bench Visualization."""

import json
import os
import re
from flask import Flask, jsonify, request, send_from_directory, render_template

app = Flask(__name__)

# -- Paths (relative to this script's location) ------------------------------
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(VIZ_DIR, ".."))

EVAL_DATA_DIR = os.path.join(BASE, "EvalData")
EVAL_SET_FILE = os.path.join(EVAL_DATA_DIR, "EvalSets.json")
QA_FILE = os.path.join(EVAL_DATA_DIR, "QAEvalSets.json")
CAPTION_RESULT_DIR = os.path.join(BASE, "caption_eval", "result", "caption")
DIRECT_ALIGN_DIR = os.path.join(BASE, "caption_eval", "result", "DirectAlign")
VQA_DIR = os.path.join(BASE, "vqa_eval", "results")

# -- In-memory data stores ---------------------------------------------------
SAMPLES = {}
QA_MAP = {}
CAPTION_DATA = {}
JUDGE_DATA = {}
MODEL_LIST = []
CAPTION_MODEL_LIST = []

VQA_DATA = {}
VQA_MODEL_LIST = []

ATOMIC_SCORED = {}
ATOMIC_RAW = {}
ATOMIC_COMBOS = []
GT_ATOMIC_FACTS = {}


def _load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _model_key_from_caption_file(fname):
    m = re.match(r"(.+?)_\d{8}_\d{6}_CaptionResult\.jsonl$", fname)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)_CaptionResult\.jsonl$", fname)
    return m.group(1) if m else None


def load_all_data():
    global MODEL_LIST

    # 1. EvalSets.json
    with open(EVAL_SET_FILE, "r", encoding="utf-8") as f:
        for rec in json.load(f):
            SAMPLES[rec["sample_id"]] = rec
    print(f"Loaded {len(SAMPLES)} samples from EvalSets.json")

    # 2. QAEvalSets.json
    if os.path.isfile(QA_FILE):
        with open(QA_FILE, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                QA_MAP[rec["sample_id"]] = rec
        print(f"Loaded {len(QA_MAP)} QA records from QAEvalSets.json")

    # 3. CaptionResult files from caption_eval/result/caption/{easy,hard}/
    if os.path.isdir(CAPTION_RESULT_DIR):
        scan_dirs = [CAPTION_RESULT_DIR]
        for subdir in ("easy", "hard"):
            sd = os.path.join(CAPTION_RESULT_DIR, subdir)
            if os.path.isdir(sd):
                scan_dirs.append(sd)
        for scan_dir in scan_dirs:
            subdir_name = os.path.basename(scan_dir)
            suffix = f" [{subdir_name}]" if scan_dir != CAPTION_RESULT_DIR else ""
            for fname in os.listdir(scan_dir):
                if not fname.endswith("_CaptionResult.jsonl"):
                    continue
                if fname.endswith(".bak_image"):
                    continue
                mk_base = _model_key_from_caption_file(fname)
                if not mk_base:
                    continue
                mk = f"{mk_base}{suffix}" if suffix else mk_base
                if mk in CAPTION_DATA:
                    continue
                CAPTION_DATA[mk] = {}
                for rec in _load_jsonl(os.path.join(scan_dir, fname)):
                    CAPTION_DATA[mk][rec["sample_id"]] = rec
                print(f"Loaded caption data for model '{mk}': {len(CAPTION_DATA[mk])} samples")

    MODEL_LIST = sorted(JUDGE_DATA.keys())
    CAPTION_MODEL_LIST.extend(sorted(CAPTION_DATA.keys()))
    print(f"Caption models: {CAPTION_MODEL_LIST}")

    # 4. VQA results
    _load_vqa_data()

    # 5. DirectAlign results
    _load_direct_align_data()

    # 6. GT atomic facts
    _load_gt_atomic_facts()


def _load_vqa_data():
    global VQA_MODEL_LIST
    if not os.path.isdir(VQA_DIR):
        print(f"VQA directory not found: {VQA_DIR}")
        return
    for fname in sorted(os.listdir(VQA_DIR)):
        if not fname.endswith("_vqa_result.jsonl"):
            continue
        mk = fname.replace("_vqa_result.jsonl", "")
        VQA_DATA[mk] = {}
        for rec in _load_jsonl(os.path.join(VQA_DIR, fname)):
            sid = rec.get("sample_id")
            if sid:
                VQA_DATA[mk].setdefault(sid, []).append(rec)
        print(f"Loaded VQA data for '{mk}': {len(VQA_DATA[mk])} samples")
    VQA_MODEL_LIST = sorted(VQA_DATA.keys())
    print(f"VQA models: {VQA_MODEL_LIST}")


def _load_direct_align_data():
    """Load DirectAlign results from caption_eval/result/DirectAlign/{easy,hard}/{model}/"""
    global ATOMIC_COMBOS
    if not os.path.isdir(DIRECT_ALIGN_DIR):
        print(f"DirectAlign directory not found: {DIRECT_ALIGN_DIR}")
        return

    for mode in ("easy", "hard"):
        mode_dir = os.path.join(DIRECT_ALIGN_DIR, mode)
        if not os.path.isdir(mode_dir):
            continue
        for model_name in sorted(os.listdir(mode_dir)):
            model_dir = os.path.join(mode_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            judge_tag = f"direct_{mode}"
            key = (model_name, judge_tag)
            if key in ATOMIC_SCORED or key in ATOMIC_RAW:
                continue

            scored_path = os.path.join(model_dir, "scored_results.jsonl")
            if os.path.isfile(scored_path):
                ATOMIC_SCORED[key] = {}
                for rec in _load_jsonl(scored_path):
                    ATOMIC_SCORED[key][rec["sample_id"]] = rec

            raw_path = os.path.join(model_dir, "direct_align_raw.jsonl")
            if os.path.isfile(raw_path):
                ATOMIC_RAW[key] = {}
                for rec in _load_jsonl(raw_path):
                    ATOMIC_RAW[key][rec["sample_id"]] = rec

            n_scored = len(ATOMIC_SCORED.get(key, {}))
            n_raw = len(ATOMIC_RAW.get(key, {}))
            ATOMIC_COMBOS.append({"caption_model": model_name, "judge_tag": judge_tag})
            print(f"Loaded DirectAlign '{model_name}' ({mode}): {n_scored} scored, {n_raw} raw")

    print(f"AtomicEval combos: {ATOMIC_COMBOS}")


def _load_gt_atomic_facts():
    if not os.path.isdir(EVAL_DATA_DIR):
        return
    for fname in sorted(os.listdir(EVAL_DATA_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        judge_key = None
        m = re.match(r"GT_AtomicFacts_?(.*)\.jsonl$", fname)
        if m:
            judge_key = m.group(1) if m.group(1) else "default"
        if judge_key is None:
            continue
        if judge_key in GT_ATOMIC_FACTS:
            continue
        GT_ATOMIC_FACTS[judge_key] = {}
        for rec in _load_jsonl(os.path.join(EVAL_DATA_DIR, fname)):
            GT_ATOMIC_FACTS[judge_key][rec["sample_id"]] = rec
        print(f"Loaded GT atomic facts '{judge_key}': {len(GT_ATOMIC_FACTS[judge_key])} samples")


# -- Helpers -----------------------------------------------------------------

def _sample_summary(rec):
    meta = rec.get("meta", {})
    gt = rec.get("GT", [])
    description = gt[0] if gt else ""
    return {
        "sample_id": rec["sample_id"],
        "description": description,
        "dataset": rec.get("dataset", ""),
        "robot_type": rec.get("robot_type", ""),
        "duration_sec": meta.get("duration_sec"),
        "view_names": meta.get("view_names", []),
    }


# -- API routes --------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples")
def api_samples():
    q = request.args.get("q", "").lower()
    items = []
    for sid, rec in SAMPLES.items():
        if q:
            gt = rec.get("GT", [])
            gt_text = " ".join(gt).lower()
            if q not in sid.lower() and q not in rec.get("dataset", "").lower() and q not in gt_text:
                continue
        items.append(_sample_summary(rec))
    return jsonify(items)


@app.route("/api/samples/<sample_id>")
def api_sample_detail(sample_id):
    rec = SAMPLES.get(sample_id)
    if not rec:
        return jsonify({"error": "not found"}), 404

    meta = rec.get("meta", {})
    views_raw = rec.get("views", {})
    view_names = meta.get("view_names", [])

    video_urls = {}
    for i, (vk, url) in enumerate(views_raw.items()):
        label = view_names[i] if i < len(view_names) else vk
        video_urls[label] = url

    human_review = rec.get("GT", [])
    gt_fine_steps = rec.get("fineGrainedSteps", [])

    qa = QA_MAP.get(sample_id, {"qas": [], "status": "missing"})

    return jsonify({
        "sample_id": sample_id,
        "dataset": rec.get("dataset", ""),
        "robot_type": rec.get("robot_type", ""),
        "duration_sec": meta.get("duration_sec"),
        "views": list(video_urls.keys()),
        "video_urls": video_urls,
        "human_review": human_review,
        "gt_fine_steps": gt_fine_steps,
        "instruction_raw": rec.get("instruction_raw", ""),
        "has_raw_steps": rec.get("has_raw_steps"),
        "steps_raw": rec.get("steps_raw"),
        "qas": qa.get("qas", []),
        "qa_status": qa.get("status", "missing"),
    })


@app.route("/api/models")
def api_models():
    return jsonify(MODEL_LIST)


@app.route("/api/caption/<model>/<sample_id>")
def api_caption(model, sample_id):
    cap = CAPTION_DATA.get(model, {}).get(sample_id)
    if not cap:
        return jsonify({"error": "not found"}), 404
    steps = cap.get("fineGrainedSteps") or cap.get("caption_result") or []
    return jsonify({
        "sample_id": sample_id,
        "model": model,
        "refinedInstruction": cap.get("refinedInstruction") or cap.get("instruction_raw", ""),
        "fineGrainedSteps": steps,
        "call_success": cap.get("call_success"),
        "num_views": cap.get("num_views"),
        "num_frames": cap.get("num_frames"),
        "elapsed_sec": cap.get("elapsed_sec"),
        "token_usage": cap.get("token_usage"),
    })


@app.route("/api/judge/<model>/<sample_id>")
def api_judge(model, sample_id):
    jd = JUDGE_DATA.get(model, {}).get(sample_id)
    if not jd:
        return jsonify({"error": "not found"}), 404
    return jsonify(jd)


@app.route("/api/caption_models")
def api_caption_models():
    return jsonify(CAPTION_MODEL_LIST)


@app.route("/api/vqa/<sample_id>")
def api_vqa(sample_id):
    qa_rec = QA_MAP.get(sample_id, {})
    qa_lookup = {}
    for q in qa_rec.get("qas", []):
        qa_lookup[q["question_id"]] = q

    result = {}
    for mk in VQA_MODEL_LIST:
        for rec in VQA_DATA[mk].get(sample_id, []):
            qid = rec["question_id"]
            if qid not in result:
                qa_info = qa_lookup.get(qid, {})
                gt_answer = qa_info.get("answer", "") or rec.get("gt_answer", "")
                result[qid] = {
                    "question_id": qid,
                    "capability": rec.get("capability", ""),
                    "error_type": rec.get("error_type", ""),
                    "answer_type": rec.get("answer_type", ""),
                    "question": rec.get("question", ""),
                    "gt_answer": gt_answer,
                    "options": qa_info.get("options", []),
                    "mode": qa_info.get("mode", ""),
                    "models": {},
                }
            result[qid]["models"][mk] = {
                "answer": rec.get("model_answer_parsed", ""),
                "correct": rec.get("correct", False),
            }
    by_cap = {}
    for qid, info in result.items():
        cap = info["capability"]
        by_cap.setdefault(cap, []).append(info)
    return jsonify({
        "sample_id": sample_id,
        "vqa_models": VQA_MODEL_LIST,
        "by_capability": by_cap,
    })


@app.route("/api/atomic_models")
def api_atomic_models():
    return jsonify(ATOMIC_COMBOS)


@app.route("/api/atomic/<caption_model>/<judge_tag>/<sample_id>")
def api_atomic(caption_model, judge_tag, sample_id):
    key = (caption_model, judge_tag)
    scored = ATOMIC_SCORED.get(key, {}).get(sample_id)
    raw = ATOMIC_RAW.get(key, {}).get(sample_id)

    is_direct = judge_tag.startswith("direct_")
    gt_key = "default"
    gt_facts = GT_ATOMIC_FACTS.get(gt_key, {}).get(sample_id)

    if not scored and not raw:
        return jsonify({"error": "not found"}), 404

    alignment = None
    if raw:
        if is_direct:
            gt_evals = raw.get("gt_evaluations", [])
            halluc = raw.get("hallucinated_actions", [])
            summary = raw.get("summary", {})

            gt_fact_lookup = {}
            if gt_facts:
                for cr in gt_facts.get("capability_results", []):
                    for f in cr.get("gt_atomic_facts", []):
                        gt_fact_lookup[f["fact_id"]] = f

            by_cap = {}
            for ev in gt_evals:
                cap = ev.get("capability", "unknown")
                by_cap.setdefault(cap, [])
                enriched = dict(ev)
                gt_f = gt_fact_lookup.get(ev.get("fact_id", ""))
                if gt_f:
                    enriched["fact_text"] = gt_f.get("fact_text", "")
                    enriched["step_anchor"] = gt_f.get("step_anchor")
                    enriched["object"] = gt_f.get("object")
                by_cap[cap].append(enriched)

            alignment = {
                "type": "direct_align",
                "by_capability": by_cap,
                "hallucinated_actions": halluc,
                "summary": summary,
            }
        else:
            alignment = raw.get("raw_response", raw)

    return jsonify({
        "sample_id": sample_id,
        "caption_model": caption_model,
        "judge_tag": judge_tag,
        "scores": scored,
        "alignment": alignment,
        "gt_atomic_facts": gt_facts,
    })


@app.route("/api/gt_atomic_facts")
def api_gt_atomic_facts_info():
    result = {}
    for judge_key, data in GT_ATOMIC_FACTS.items():
        result[judge_key] = len(data)
    return jsonify(result)


@app.route("/api/gt_atomic_facts/<judge_key>/<sample_id>")
def api_gt_atomic_facts_detail(judge_key, sample_id):
    facts = GT_ATOMIC_FACTS.get(judge_key, {}).get(sample_id)
    if not facts:
        return jsonify({"error": "not found"}), 404
    return jsonify(facts)


@app.route("/api/stats/<model>")
def api_stats(model):
    jd = JUDGE_DATA.get(model, {})
    if not jd:
        return jsonify({"error": "no judge data for model"}), 404

    total_correct = 0
    total_questions = 0
    cap_correct = {}
    cap_total = {}

    for sid, rec in jd.items():
        total_correct += rec.get("total_correct", 0)
        total_questions += rec.get("total_questions", 0)

        qa_rec = QA_MAP.get(sid, {})
        qas = qa_rec.get("qas", [])
        qa_id_to_cap = {q["question_id"]: q.get("capability", "unknown") for q in qas}

        for qr in rec.get("qa_results", []):
            cap = qa_id_to_cap.get(qr["question_id"], "unknown")
            cap_total[cap] = cap_total.get(cap, 0) + 1
            if qr.get("correct"):
                cap_correct[cap] = cap_correct.get(cap, 0) + 1

    cap_breakdown = []
    for cap in sorted(cap_total.keys()):
        c = cap_correct.get(cap, 0)
        t = cap_total[cap]
        cap_breakdown.append({
            "capability": cap,
            "correct": c,
            "total": t,
            "accuracy": round(c / t, 4) if t else 0,
        })

    return jsonify({
        "model": model,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "accuracy": round(total_correct / total_questions, 4) if total_questions else 0,
        "capability_breakdown": cap_breakdown,
    })


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_all_data()
    app.run(host="0.0.0.0", port=5001, debug=False)
