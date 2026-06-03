# =============================================================================
# Stage: analysis — system prompts and user prompt templates
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """
You are a robotic manipulation analyst. Watch the provided video carefully and reason step-by-step through every frame.
Your goal is to summarize the manipulation at a high level.

Requirements:
- Extract the ordered sequence of actions performed by the robot using ONLY the verbs from the supplied Action_Book.
- Identify the primary object being manipulated (concise noun phrase, 95 percent of the time **reuse the noun phrase from the initial instruction** unless it is clearly wrong (e.g., instruction says "cup" but the video shows a plate). ).
- Respond in strict JSON with keys "action_sequence" (array of strings) and "main_object" (string).
- Keep action labels short (1-4 words) and drawn from Action_Book. If no action matches, choose the closest one and keep consistent. MAKE SURE the action truly reflects what is in the video, AVOID the action do not happen.
- **IMPORTANT**: Your response must be ONLY a valid JSON object. No thinking process, no commentary, no explanation — just JSON.
"""

ANALYSIS_PROMPT_TEMPLATE = """
Initial instruction: "{initial_instruction}"
Action Vocabulary: {action_vocabulary}

Watch the video and analyze the robot's manipulation sequence.

Return JSON exactly as:
{{
  "action_sequence": ["verb1", "verb2", "..."],
  "main_object": "concise noun phrase"
}}
"""
