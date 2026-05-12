"""Constants and default configuration."""

# 10 capability dimensions (fixed order)
CAPABILITIES = [
    "action_sequence",
    "active_actor",
    "target_object",
    "initial_configuration",
    "final_configuration",
    "contact_and_approach",
    "trajectory_and_orientation",
    "object_interaction",
    "failure_and_recovery",
    "body_motion",
]

# Normalized capability weights (sum = 1.0, equal weight)
_EQUAL_WEIGHT = 1.0 / len(CAPABILITIES)
CAPABILITY_WEIGHTS: dict[str, float] = {cap: _EQUAL_WEIGHT for cap in CAPABILITIES}

# Scoring
PARTIAL_CREDIT_ALPHA = 0.5
CAPTION_SCORE_W_CONSISTENCY = 1.0 / 3
CAPTION_SCORE_W_COVERAGE = 1.0 / 3
CAPTION_SCORE_W_ANTI_HALLUCINATION = 1.0 / 3

# API defaults
DEFAULT_MODEL = "openai.gpt-5.4-2026-03-05"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RETRIES = 100
DEFAULT_NUM_WORKERS = 8

# Checkpoint
SAVE_EVERY_N = 5
