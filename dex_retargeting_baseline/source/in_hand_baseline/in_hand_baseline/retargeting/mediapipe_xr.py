"""MediaPipe hand tracking device that emits IsaacLab OpenXR-style hand data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


_MP_TO_OPENXR = {
    "wrist": 0,
    "thumb_metacarpal": 1,
    "thumb_proximal": 2,
    "thumb_distal": 3,
    "thumb_tip": 4,
    "index_metacarpal": 5,
    "index_proximal": 5,
    "index_intermediate": 6,
    "index_distal": 7,
    "index_tip": 8,
    "middle_metacarpal": 9,
    "middle_proximal": 9,
    "middle_intermediate": 10,
    "middle_distal": 11,
    "middle_tip": 12,
    "ring_metacarpal": 13,
    "ring_proximal": 13,
    "ring_intermediate": 14,
    "ring_distal": 15,
    "ring_tip": 16,
    "little_metacarpal": 17,
    "little_proximal": 17,
    "little_intermediate": 18,
    "little_distal": 19,
    "little_tip": 20,
}


class MediaPipeXRDevice(DeviceBase):
    """Read a camera/video stream with MediaPipe and expose OpenXR-like hand poses."""

    def __init__(self, cfg: MediaPipeXRDeviceCfg, retargeters: list[RetargeterBase] | None = None):
        super().__init__(retargeters)
        self.cfg = cfg
        self._callbacks: dict[Any, Callable] = {}
        self._cv2 = None
        self._hands = None
        self._capture = None
        self._previous_right = self._make_default_hand()
        self._previous_left = self._make_default_hand()
        self._open()

    def __del__(self):
        if getattr(self, "_capture", None) is not None:
            self._capture.release()

    def reset(self):
        self._previous_right = self._make_default_hand()
        self._previous_left = self._make_default_hand()

    def add_callback(self, key: Any, func: Callable):
        self._callbacks[key] = func

    def _get_raw_data(self) -> dict:
        frame = self._read_frame()
        if frame is None:
            return {
                DeviceBase.TrackingTarget.HAND_LEFT: self._previous_left,
                DeviceBase.TrackingTarget.HAND_RIGHT: self._previous_right,
            }

        image_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self._hands.process(image_rgb)

        right_hand = self._select_right_hand(result)
        if right_hand is not None:
            self._previous_right = self._landmarks_to_openxr(right_hand)

        return {
            DeviceBase.TrackingTarget.HAND_LEFT: self._previous_left,
            DeviceBase.TrackingTarget.HAND_RIGHT: self._previous_right,
        }

    def _open(self):
        try:
            import cv2
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MediaPipe retargeting requires both `mediapipe` and `opencv-python` in the IsaacLab Python environment."
            ) from exc

        self._cv2 = cv2
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.cfg.max_num_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        source = self.cfg.video_path if self.cfg.video_path else self.cfg.camera_index
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open MediaPipe video source: {source}")

    def _read_frame(self):
        ok, frame = self._capture.read()
        if ok:
            return frame
        if self.cfg.loop_video and self.cfg.video_path:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok:
                return frame
        return None

    def _select_right_hand(self, result):
        if not result.multi_hand_world_landmarks and not result.multi_hand_landmarks:
            return None

        world_landmarks = result.multi_hand_world_landmarks or result.multi_hand_landmarks
        handedness = result.multi_handedness or []
        fallback = world_landmarks[0]
        for hand_landmarks, hand_info in zip(world_landmarks, handedness):
            label = hand_info.classification[0].label
            if label == self.cfg.right_hand_label:
                return hand_landmarks
        return fallback

    def _landmarks_to_openxr(self, hand_landmarks) -> dict[str, np.ndarray]:
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        points *= np.array(self.cfg.position_scale, dtype=np.float32)
        if self.cfg.flip_y:
            points[:, 1] *= -1.0
        if self.cfg.flip_z:
            points[:, 2] *= -1.0

        wrist_quat = _estimate_wrist_quat(points)
        hand = {}
        for joint_name in HAND_JOINT_NAMES:
            if joint_name == "palm":
                pos = 0.5 * (points[0] + points[9])
            else:
                pos = points[_MP_TO_OPENXR[joint_name]]
            hand[joint_name] = np.array([pos[0], pos[1], pos[2], *wrist_quat], dtype=np.float32)
        return hand

    @staticmethod
    def _make_default_hand() -> dict[str, np.ndarray]:
        default_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return {name: default_pose.copy() for name in HAND_JOINT_NAMES}


def _estimate_wrist_quat(points: np.ndarray) -> np.ndarray:
    wrist = points[0]
    index = points[5] - wrist
    pinky = points[17] - wrist
    middle = points[9] - wrist

    x_axis = _normalize(index)
    z_axis = _normalize(np.cross(index, pinky))
    if np.linalg.norm(z_axis) < 1.0e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if np.dot(y_axis, middle) < 0.0:
        y_axis *= -1.0
        z_axis *= -1.0

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _quat_from_matrix(rot)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1.0e-6:
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _quat_from_matrix(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rot)))
        if i == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1.0e-6)
    return quat


@dataclass
class MediaPipeXRDeviceCfg(DeviceCfg):
    """Configuration for MediaPipe-backed OpenXR-style hand tracking."""

    video_path: str | None = None
    camera_index: int = 0
    loop_video: bool = True
    right_hand_label: str = "Right"
    position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    flip_y: bool = True
    flip_z: bool = True
    max_num_hands: int = 1
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    retargeters: list[RetargeterCfg] = field(default_factory=list)
    class_type: type[DeviceBase] | None = MediaPipeXRDevice

