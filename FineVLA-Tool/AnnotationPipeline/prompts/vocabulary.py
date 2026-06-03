# =============================================================================
# Action Vocabulary and Shared Prompt Components
# =============================================================================

ACTION_VOCABULARY = [
    "Grasp", "Grab", "Drag", "Pick up", "Place", "Move", "Rotate",
    "Insert", "Push", "Wipe", "Sweep", "Knock", "Release", "Press"
]

ACTION_FINE_GRAINED_GUIDANCE = """
1. Grasp: Specify the contact point and gripper approach direction(from above, from the left, from the right, from the front, from the back).
2. Grab: Identify the portion being grabbed and the direction of approach.
3. Drag: Describe the drag direction and distance.
4. Pick up: State the contact point, grasp direction (from above, from the left, from the right, from the front, from the back), and lift path.
5. Place: Indicate the target location and final object orientation (e.g., upright, lying flat).
6. Move: Clarify movement direction and displacement (approximate distance).
7. Rotate: Provide rotation axis, direction (clockwise/counter-clockwise).
8. Insert: Describe insertion direction and alignment cues.
9. Push: Note the push contact point and direction.
10. Wipe: Outline the wiped surface area and sweeping direction.
11. Sweep: Describe the swept region and motion direction.
12. Knock: Specify the contact point and approach direction.
13. Release: Mention release location and resulting object orientation (e.g., upright, lying flat).
14. Press: Specify the contact point, pressing direction, and press depth/force.
"""

# wait to add more
FEW_SHOT_EXAMPLES = """
1.
    "action_sequence": ["Pick up", "Rotate", "Place"],
    "main_object": "ceramic bowl",
    "Fine_Grained": [
        "Pick up the ceramic bowl from the right far edge.",
        "Rotate the bowl clockwise for two circle.",
        "Place the bowl at the center of the table."
    ]
2.
    "action_sequence": ["Grasp", "Lift", "Move", "Stack", "Release"],
    "main_object": "white paper cup",
    "Fine_Grained": [
        "Grasp the white paper cup on the right by its top right rim from above.",
        "Lift it vertically.",
        "Move it horizontally to the left to align with the other cup.",
        "Lower the cup into the white paper cup on the left to stack them.",
        "Release the grip and retract the arm upwards."
    ]
"""
