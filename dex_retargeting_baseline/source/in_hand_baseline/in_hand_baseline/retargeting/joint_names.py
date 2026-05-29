"""Joint-name constants shared by the right-hand retargeting pipeline."""

INSPIRE_HAND_ACTION_JOINT_NAMES = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]

RIGHT_INSPIRE_DEXPILOT_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
]

RIGHT_INSPIRE_ACTION_INDICES = [
    INSPIRE_HAND_ACTION_JOINT_NAMES.index(name) for name in RIGHT_INSPIRE_DEXPILOT_JOINT_NAMES
]

