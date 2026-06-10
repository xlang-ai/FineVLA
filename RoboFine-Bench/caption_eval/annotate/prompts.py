"""Prompt templates for robot manipulation video annotation."""


SINGLE_VIEW_PROMPT = """\
You are a robot manipulation video annotator.

Task: Decompose the video into an ordered sequence of fine-grained steps. Each step must describe exactly one visible physical movement by the robot.

For each step, include the following when visually confirmed:
- action and gripper state (e.g., grasp, pick up, place, push, rotate; open → close, release, maintain grasp)
- active actor (left hand, right hand, both hands, gripper, finger)
- target object using the task instruction name; disambiguate similar objects by position or appearance (e.g., the red cup on the left, the front-most block)
- contact region and approach direction (e.g., by the handle from above, at the top edge from the right)
- object initial state or location if relevant (e.g., upright, lying flat, inside the container)
- trajectory and orientation change (e.g., move forward, lift up, rotate clockwise/counter-clockwise)
- final placement location and final pose if the object is repositioned
- interaction with other objects (e.g., collision, dragging, tipping, displacement)
- failures, retries, slippage, or fail-then-succeed patterns
- body motion such as base, torso, or camera movement

Rules:
- One step = one physical movement.
- Describe only visible facts. Do not infer hidden actions, intent, or occluded contact.
- Use the robot's egocentric frame: left, right, forward, backward, up, down.
- Keep descriptions short, action-oriented, and literal.
- If a detail is unclear or not visible, omit it.
- Mention dual-arm coordination only when both arms are active; label it as stabilize-and-act, simultaneous, sequential, or handoff.

Output JSON only:
{ "Step1": "...", "Step2": "...", "StepN": "..." }
"""


MULTI_VIEW_PROMPT = """\
You are a robot manipulation video annotator.

Task: Analyze three synchronized views and decompose the robot's behavior into an ordered sequence of atomic physical movements.

View use:
- Main View: primary view for all annotation decisions, including step boundaries, action order, active arm, global motion, object relations, and final results.
- Wrist Views (left/right): auxiliary only. Use them only to refine local details such as gripper state, contact region, approach direction, or slight slip.
- If Wrist Views conflict with the Main View, follow the Main View.

Step requirements:
For each step, include the following when visually confirmed:
- action and gripper state (e.g., grasp, pick up, place, push, rotate; open → close, release, maintain grasp)
- active actor (left hand, right hand, both hands, gripper, finger)
- target object using the task instruction name; disambiguate similar objects by position or appearance (e.g., the red cup on the left, the front-most block)
- contact region and approach direction (e.g., by the handle from above, at the top edge from the right)
- object initial state or location if relevant (e.g., upright, lying flat, inside the container)
- trajectory and orientation change (e.g., move forward, lift up, rotate clockwise/counter-clockwise)
- final placement location and final pose if the object is repositioned
- interaction with other objects (e.g., collision, dragging, tipping, displacement)
- failures, retries, slippage, or fail-then-succeed patterns
- body motion such as base, torso, or camera movement

Rules:
- One step = one physical movement.
- Start a new step when the robot changes primitive action, target, or contact state.
- Describe only visible facts. Do not infer hidden actions, intent, or occluded contact.
- If a detail is unclear or not visually confirmed, omit it.
- Use the robot's egocentric frame: left, right, forward, backward, up, down.
- Keep descriptions concise, action-oriented, and literal.
- Mention dual-arm coordination only when both arms actively contribute to the same manipulation event; use stabilize-and-act, simultaneous, or handoff when applicable.

Output JSON only:
{ "Step1": "...", "Step2": "...", "StepN": "..." }"""


def classify_view(view_name: str, index: int) -> str:
    """Map a view_name to a human-readable label for multi-view prompts.

    index=0 is always the main view. Others are classified by name keywords.

    Covers all 10 datasets:
      - DROID/RH20T: ['primary', 'wrist'] → main + wrist
      - Galaxea: ['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb']
      - RDT: ['main', 'left_wrist', 'right_wrist']
      - RoboCoin: varies (cam_high_rgb, cam_front_rgb, cam_left_wrist_rgb, etc.)
      - RoboMIND: camera_top/front + camera_left/right or camera_wrist_left/right
    """
    if index == 0:
        return "This is the main view of the video."

    vn = view_name.lower()
    is_wrist = "wrist" in vn
    is_left = "left" in vn
    is_right = "right" in vn

    if is_left and is_wrist:
        return "This is the left wrist view of the video."
    if is_right and is_wrist:
        return "This is the right wrist view of the video."
    if is_wrist:
        return "This is the wrist view of the video."
    if is_left:
        return "This is the left side view of the video."
    if is_right:
        return "This is the right side view of the video."
    return f"This is the {view_name} view of the video."
