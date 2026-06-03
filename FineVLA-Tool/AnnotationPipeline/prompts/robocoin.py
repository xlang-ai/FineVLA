# =============================================================================
# RoboCOIN-specific prompt variants
# These extend the generic bimanual prompts with:
#   1. Fold / Zip action vocabulary and guidance
#   2. Static / End step handling
#   3. Few-shot examples for bimanual coordination
#   4. Stronger "no-repeat" instruction in the user template
# DO NOT use these for other datasets.
# =============================================================================

from .vocabulary import ACTION_FINE_GRAINED_GUIDANCE

ACTION_FINE_GRAINED_GUIDANCE_ROBOCOIN = ACTION_FINE_GRAINED_GUIDANCE + """
15. Fold: Describe the fold axis (e.g., along the long edge, across the center), fold direction \
(inward/outward/toward robot), and approximate result (e.g., "in half", "into thirds").
16. Zip: Specify the zipper starting position, pull direction (upward/downward/leftward/rightward), \
and travel distance or endpoint (e.g., "from the bottom stop to the top of the zipper").
"""

GALAXEA_STEP_SYSTEM_PROMPT_BIMANUAL_ROBOCOIN = """
You are an expert Video-Language Action (VLA) captioner focused on robot manipulation videos.
You are given a short video clip showing ONE step of a multi-step bimanual manipulation task.
The robot has TWO independent arms (left and right), each with its own gripper.
Watch the clip carefully and refine the provided step description with precise physical details.

Guidelines:
- Explicitly state which arm (left/right/both) performs the action in this step.
- If one arm stabilizes an object while the other manipulates, describe both roles.
- If one arm is idle in this step, you may omit it — only mention arms that are actively moving or holding.
- Add specific contact points, gripper approach directions, and spatial relationships.
- Include motion direction and approximate displacement when visible.
- Use clear, natural, concise language — one or two sentences at most.
- Describe spatial relationships relative to the robot's perspective (left/right/far/close/upper/below).
- The refined description should start with the arm identifier and an action verb \
(e.g., "Left arm grasps...", "Right arm places...", "Both arms lift...").
- When multiple similar objects are present, use distinguishing attributes (color, size, position) to identify them.

Special step types — handle as follows:
- "Static" / holding steps: The arm(s) show no net movement. Describe which arm holds what object \
and in what position/orientation. Do NOT invent movement that is not visible. \
(e.g., "Left arm holds the open bag upright near the table edge, keeping the opening stable.")
- "End" / retract steps: The arm(s) release the object and retract to a neutral/resting position. \
Describe the release and the retraction direction. \
(e.g., "Right arm releases the cloth from its gripper and retracts upward to resting position.")
- Very short clips with minimal visible motion: describe only what is clearly observable; \
do NOT hallucinate unseen details.

Few-shot examples:
[Static step]  "Left arm holds the unzipped storage bag open by pressing both gripper fingers \
against the rim, keeping the opening stable while the right arm loads items inside."
[End step]     "Right arm releases the folded cloth and retracts upward and backward away from \
the basket to its resting position above the workspace."
[Single-arm]   "Left arm grasps the hamburger patty from above at its center using a top-down \
grip and lifts it vertically off the tray."
[Bimanual]     "Both arms simultaneously grasp opposite ends of the shirt — left arm at the \
left sleeve cuff and right arm at the right sleeve cuff — and stretch it taut horizontally \
before beginning the fold."
"""

GALAXEA_STEP_PROMPT_TEMPLATE_ROBOCOIN = (
    'Overall task: "{initial_instruction}"\n'
    '{previous_steps_context}'
    'Step {step_index} description: "{step_description}"\n\n'
    'Use the following guidance to enrich the description with concrete physical details:\n'
    + ACTION_FINE_GRAINED_GUIDANCE_ROBOCOIN +
    '\nWatch the video clip above and provide a refined, fine-grained description for this single step.\n'
    '**IMPORTANT**: Do NOT re-describe any action already listed in the previous steps above — '
    'focus ONLY on what is NEW and distinct in this clip.\n\n'
    'Return JSON:\n'
    '{{\n'
    '  "refined_step": "one detailed sentence describing exactly what the robot does in this clip"\n'
    '}}\n'
)
