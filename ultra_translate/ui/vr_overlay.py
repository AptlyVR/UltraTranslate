from __future__ import annotations

import ctypes
import math
import warnings

import numpy as np
import openvr
from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter
from PyQt6.QtWidgets import QApplication, QWidget
from scipy.spatial.transform import Rotation

from ultra_translate.runtime import current_integrity_level, process_integrity_level


class OpenVrOverlayService(QObject):
    status_changed = pyqtSignal(str)

    def __init__(self, panel: QWidget, settings) -> None:
        super().__init__()
        self.panel = panel
        self.settings = settings
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(11)
        self._timer.timeout.connect(self._tick)
        self._vr_active = False
        self._overlay = None
        self._vr_system = None
        self._overlay_handle = None
        self._mouse_buttons = Qt.MouseButton.NoButton
        self._input_enabled = None
        self._panel_dirty = True
        self._last_panel_size = (0, 0)
        self._last_alpha: float | None = None
        self._head_follow_position_offsets: list[float | None] = [None, None, None]
        self._errors = 0

    def start(self) -> bool:
        if self._vr_active:
            return True
        try:
            openvr.init(openvr.VRApplication_Overlay)
            self._overlay = openvr.VROverlay()
            self._vr_system = openvr.VRSystem()
            self._overlay_handle = self._overlay.createOverlay(
                "ultratranslate.overlay",
                "UltraTranslate",
            )
            self._overlay.setOverlayWidthInMeters(
                self._overlay_handle,
                self.settings.width_meters,
            )
            self._overlay.setOverlayMouseScale(
                self._overlay_handle,
                self._mouse_scale(self.panel.width(), self.panel.height()),
            )
            self._configure_interaction_flags()
            self._set_input_method(self.settings.interactive, force=True)
            self._panel_dirty = True
            self._last_alpha = None
            self._reset_head_follow_position_offsets()
            self._timer.setInterval(self._preferred_tick_interval_ms())
            self._overlay.showOverlay(self._overlay_handle)
            self._vr_active = True
            self._timer.start()
            self.status_changed.emit(
                f"SteamVR overlay active at {int(round(1000.0 / max(self._timer.interval(), 1)))} Hz."
            )
            return True
        except Exception as exc:
            self._cleanup_overlay_runtime()
            self.status_changed.emit(self._format_start_error(exc))
            return False

    def stop(self) -> None:
        self._timer.stop()
        self._cleanup_overlay_runtime()

    def mark_dirty(self) -> None:
        self._panel_dirty = True

    def update_settings(self, settings) -> None:
        self.settings = settings
        self._panel_dirty = True
        self._reset_head_follow_position_offsets()
        if self._overlay and self._overlay_handle:
            try:
                self._overlay.setOverlayWidthInMeters(self._overlay_handle, settings.width_meters)
                self._configure_interaction_flags()
                self._set_input_method(settings.interactive, force=True)
            except Exception as exc:
                self.status_changed.emit(f"Overlay settings error: {exc}")

    def is_active(self) -> bool:
        return self._vr_active

    def _tick(self) -> None:
        if not self._vr_active or not self._overlay or not self._overlay_handle:
            return
        try:
            poses = self._device_poses()
            overlay_matrix = self._overlay_absolute_matrix(poses)
            self._apply_transform(poses, overlay_matrix)
            self._render_panel()
            alpha = self._effective_alpha(poses, overlay_matrix)
            allow_input = self.settings.interactive and alpha > 0.025
            if self._last_alpha is None or abs(alpha - self._last_alpha) >= 0.005:
                self._overlay.setOverlayAlpha(self._overlay_handle, alpha)
                self._last_alpha = alpha
            self._set_input_method(allow_input)
            if allow_input:
                self._poll_events()
            self._errors = 0
        except Exception as exc:
            self._errors += 1
            if self._errors in (1, 30):
                self.status_changed.emit(f"Overlay tick error: {exc}")
            if self._errors > 300:
                self.status_changed.emit("Overlay stopped after repeated errors.")
                self.stop()

    def _render_panel(self) -> None:
        if self._overlay is None or self._overlay_handle is None:
            return
        width = max(self.panel.width(), 1)
        height = max(self.panel.height(), 1)
        current_size = (width, height)
        if not self._panel_dirty and current_size == self._last_panel_size:
            return
        image = QImage(width, height, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.panel.render(painter, QPoint())
        painter.end()

        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        raw_buffer = (ctypes.c_char * image.sizeInBytes()).from_address(int(ptr))
        self._overlay.setOverlayMouseScale(
            self._overlay_handle,
            self._mouse_scale(width, height),
        )
        self._overlay.setOverlayRaw(
            self._overlay_handle,
            raw_buffer,
            width,
            height,
            4,
        )
        self._panel_dirty = False
        self._last_panel_size = current_size

    def _apply_transform(self, poses, absolute_matrix: np.ndarray | None = None) -> None:
        if self._overlay is None or self._overlay_handle is None:
            return
        if absolute_matrix is None:
            absolute_matrix = self._overlay_absolute_matrix(poses)
        if absolute_matrix is None:
            return
        if self.settings.origin_mode == "world":
            self._overlay.setOverlayTransformAbsolute(
                self._overlay_handle,
                openvr.TrackingUniverseStanding,
                self._numpy_to_matrix(absolute_matrix),
            )
            return
        device_index = self._origin_device_index()
        device_matrix = self._tracked_pose_matrix(poses, device_index)
        if device_matrix is None:
            return
        relative_matrix = np.linalg.inv(device_matrix) @ absolute_matrix
        self._overlay.setOverlayTransformTrackedDeviceRelative(
            self._overlay_handle,
            device_index,
            self._numpy_to_matrix(relative_matrix),
        )

    def _effective_alpha(self, poses, overlay_matrix: np.ndarray | None = None) -> float:
        base_alpha = max(0.0, min(1.0, self.settings.opacity))
        if not self.settings.look_at_fade_enabled:
            return base_alpha

        if overlay_matrix is None:
            overlay_matrix = self._overlay_absolute_matrix(poses)
        hmd_matrix = self._tracked_pose_matrix(poses, openvr.k_unTrackedDeviceIndex_Hmd)
        if overlay_matrix is None or hmd_matrix is None:
            return base_alpha

        overlay_pos = overlay_matrix[:3, 3]
        hmd_pos = hmd_matrix[:3, 3]
        gaze_distance = self.settings.look_at_distance
        if gaze_distance <= 0.0:
            gaze_distance = float(np.linalg.norm(overlay_pos - hmd_pos))
        else:
            gaze_distance += 0.20
        gaze_distance = max(0.01, gaze_distance)

        forward = -hmd_matrix[:3, 2]
        norm = float(np.linalg.norm(forward))
        if norm <= 0.0:
            return base_alpha
        forward = forward / norm
        gaze_point = hmd_pos + (forward * gaze_distance)
        offset_distance = float(np.linalg.norm(overlay_pos - gaze_point))
        capped_distance = min(gaze_distance, 1.0)
        fade_rate = max(0.2, self.settings.look_at_sensitivity * 10.0)

        alpha = max(0.0, min(1.0, (offset_distance * -fade_rate) + ((capped_distance - 0.1) * 10.0)))
        min_alpha = 0.0 if self.settings.hide_when_not_looked_at else max(
            0.0,
            min(1.0, self.settings.look_at_min_opacity),
        )
        if min_alpha <= base_alpha:
            return min_alpha + ((base_alpha - min_alpha) * alpha)
        return base_alpha + ((min_alpha - base_alpha) * (1.0 - alpha))

    def _overlay_absolute_matrix(self, poses) -> np.ndarray | None:
        hmd_matrix = self._tracked_pose_matrix(poses, openvr.k_unTrackedDeviceIndex_Hmd)
        if self.settings.origin_mode == "world":
            base_position = np.array(
                [
                    self.settings.world_x,
                    self.settings.world_y,
                    self.settings.world_z,
                ],
                dtype=np.float32,
            )
            position = self._apply_head_position_follow(base_position, hmd_matrix)
            base_rotation = self._compose_numpy_matrix(
                0.0,
                0.0,
                0.0,
                self.settings.world_pitch,
                self.settings.world_yaw,
                self.settings.world_roll,
            )[:3, :3]
            rotation = self._resolved_rotation_matrix(
                position,
                base_rotation,
                hmd_matrix,
                self.settings.world_pitch,
                self.settings.world_yaw,
                self.settings.world_roll,
            )
            return self._matrix_from_rotation_translation(rotation, position)

        device_index = self._origin_device_index()
        device_matrix = self._tracked_pose_matrix(poses, device_index)
        if device_matrix is None:
            return None

        base_matrix = device_matrix @ self._compose_numpy_matrix(
            self.settings.x_offset,
            self.settings.y_offset,
            self.settings.z_offset,
            self.settings.pitch,
            self.settings.yaw,
            self.settings.roll,
        )
        position = self._apply_head_position_follow(base_matrix[:3, 3], hmd_matrix)
        rotation = self._resolved_rotation_matrix(
            position,
            base_matrix[:3, :3],
            hmd_matrix,
            self.settings.pitch,
            self.settings.yaw,
            self.settings.roll,
        )
        return self._matrix_from_rotation_translation(rotation, position)

    def _poll_events(self) -> None:
        event = openvr.VREvent_t()
        while True:
            has_event, event = self._overlay.pollNextOverlayEvent(self._overlay_handle, event)
            if not has_event:
                return
            if event.eventType == openvr.VREvent_MouseMove:
                self._dispatch_mouse_event(QEvent.Type.MouseMove, event.data.mouse.x, event.data.mouse.y)
            elif event.eventType == openvr.VREvent_MouseButtonDown:
                self._mouse_buttons = Qt.MouseButton.LeftButton
                self._dispatch_mouse_event(QEvent.Type.MouseButtonPress, event.data.mouse.x, event.data.mouse.y)
            elif event.eventType == openvr.VREvent_MouseButtonUp:
                self._dispatch_mouse_event(QEvent.Type.MouseButtonRelease, event.data.mouse.x, event.data.mouse.y)
                self._mouse_buttons = Qt.MouseButton.NoButton

    def _dispatch_mouse_event(self, kind: QEvent.Type, x: float, y: float) -> None:
        local = QPointF(x, y)
        target = self.panel.childAt(int(round(local.x())), int(round(local.y()))) or self.panel
        target_point = target.mapFrom(self.panel, local.toPoint())
        target_local = QPointF(target_point)
        event = QMouseEvent(
            kind,
            target_local,
            target_local,
            target_local,
            Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
            self._mouse_buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(target, event)

    def _set_input_method(self, enabled: bool, force: bool = False) -> None:
        if self._overlay is None or self._overlay_handle is None:
            return
        if not force and self._input_enabled == enabled:
            return
        self._overlay.setOverlayInputMethod(
            self._overlay_handle,
            openvr.VROverlayInputMethod_Mouse if enabled else openvr.VROverlayInputMethod_None,
        )
        self._input_enabled = enabled

    def _configure_interaction_flags(self) -> None:
        if self._overlay is None or self._overlay_handle is None:
            return
        try:
            self._overlay.setOverlayFlag(
                self._overlay_handle,
                openvr.VROverlayFlags_HideLaserIntersection,
                False,
            )
            self._overlay.setOverlayFlag(
                self._overlay_handle,
                openvr.VROverlayFlags_MakeOverlaysInteractiveIfVisible,
                True,
            )
            self._overlay.setOverlayFlag(
                self._overlay_handle,
                openvr.VROverlayFlags_SendVRDiscreteScrollEvents,
                True,
            )
            self._overlay.setOverlayFlag(
                self._overlay_handle,
                openvr.VROverlayFlags_SendVRSmoothScrollEvents,
                True,
            )
            self._overlay.setOverlayFlag(
                self._overlay_handle,
                openvr.VROverlayFlags_SendVRTouchpadEvents,
                True,
            )
        except Exception:
            pass

    def _device_poses(self):
        if self._vr_system is None:
            return []
        return self._vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.011,
            openvr.k_unMaxTrackedDeviceCount,
        )

    def _origin_device_index(self) -> int:
        if not self._vr_system:
            return openvr.k_unTrackedDeviceIndex_Hmd
        if self.settings.origin_mode == "right_wrist":
            role = openvr.TrackedControllerRole_RightHand
        elif self.settings.origin_mode == "head":
            return openvr.k_unTrackedDeviceIndex_Hmd
        else:
            role = openvr.TrackedControllerRole_LeftHand
        index = self._vr_system.getTrackedDeviceIndexForControllerRole(role)
        if index == openvr.k_unTrackedDeviceIndexInvalid:
            return openvr.k_unTrackedDeviceIndex_Hmd
        return index

    def _tracked_pose_matrix(self, poses, device_index: int) -> np.ndarray | None:
        if not poses:
            return None
        try:
            pose = poses[device_index]
        except Exception:
            return None
        if not getattr(pose, "bPoseIsValid", False):
            return None
        return self._matrix_to_numpy(pose.mDeviceToAbsoluteTracking)

    def _resolved_rotation_matrix(
        self,
        position: np.ndarray,
        base_rotation: np.ndarray,
        hmd_matrix: np.ndarray | None,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
    ) -> np.ndarray:
        if hmd_matrix is None or not any(self._rotation_follow_axes()):
            return base_rotation.astype(np.float32)

        follow_rotation = self._follow_rotation_matrix(
            position,
            hmd_matrix[:3, 3],
            hmd_matrix[:3, 1],
            pitch_deg,
            yaw_deg,
            roll_deg,
        )
        if all(self._rotation_follow_axes()):
            return follow_rotation

        base_pitch, base_yaw, base_roll = self._rotation_angles(base_rotation)
        follow_pitch, follow_yaw, follow_roll = self._rotation_angles(follow_rotation)
        use_pitch, use_yaw, use_roll = self._rotation_follow_axes()
        return Rotation.from_euler(
            "zyx",
            [
                follow_roll if use_roll else base_roll,
                follow_yaw if use_yaw else base_yaw,
                follow_pitch if use_pitch else base_pitch,
            ],
            degrees=True,
        ).as_matrix().astype(np.float32)

    def _follow_rotation_matrix(
        self,
        position: np.ndarray,
        target: np.ndarray,
        up_hint: np.ndarray,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
    ) -> np.ndarray:
        direction = target - position
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            direction = (direction / direction_norm).astype(np.float32)

        up = np.array(up_hint, dtype=np.float32)
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1e-6:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            up = up / up_norm

        right = np.cross(up, direction)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1e-6:
            fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(fallback_up, direction)
            right_norm = float(np.linalg.norm(right))
            if right_norm <= 1e-6:
                fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                right = np.cross(fallback_up, direction)
                right_norm = float(np.linalg.norm(right))
        right = right / max(right_norm, 1e-6)
        aligned_up = np.cross(direction, right)
        aligned_up = aligned_up / max(float(np.linalg.norm(aligned_up)), 1e-6)

        face_matrix = np.eye(4, dtype=np.float32)
        face_matrix[:3, 0] = right
        face_matrix[:3, 1] = aligned_up
        face_matrix[:3, 2] = direction
        return (
            face_matrix @ self._compose_numpy_matrix(0.0, 0.0, 0.0, pitch_deg, yaw_deg, roll_deg)
        )[:3, :3].astype(np.float32)

    def _apply_head_position_follow(
        self,
        base_position: np.ndarray,
        hmd_matrix: np.ndarray | None,
    ) -> np.ndarray:
        if hmd_matrix is None:
            return np.array(base_position, dtype=np.float32)
        follow_axes = self._position_follow_axes()
        if not any(follow_axes):
            self._reset_head_follow_position_offsets()
            return np.array(base_position, dtype=np.float32)

        hmd_position = hmd_matrix[:3, 3]
        adjusted = np.array(base_position, dtype=np.float32)
        for axis, enabled in enumerate(follow_axes):
            if not enabled:
                self._head_follow_position_offsets[axis] = None
                continue
            if self._head_follow_position_offsets[axis] is None:
                self._head_follow_position_offsets[axis] = float(base_position[axis] - hmd_position[axis])
            adjusted[axis] = hmd_position[axis] + float(self._head_follow_position_offsets[axis] or 0.0)
        return adjusted

    def _position_follow_axes(self) -> tuple[bool, bool, bool]:
        return (
            self.settings.follow_head_position_x,
            self.settings.follow_head_position_y,
            self.settings.follow_head_position_z,
        )

    def _rotation_follow_axes(self) -> tuple[bool, bool, bool]:
        return (
            self.settings.follow_head_rotation_pitch,
            self.settings.follow_head_rotation_yaw,
            self.settings.follow_head_rotation_roll,
        )

    def _reset_head_follow_position_offsets(self) -> None:
        self._head_follow_position_offsets = [None, None, None]

    def _preferred_tick_interval_ms(self) -> int:
        if self._vr_system is None:
            return 11
        try:
            frequency = float(
                self._vr_system.getFloatTrackedDeviceProperty(
                    openvr.k_unTrackedDeviceIndex_Hmd,
                    openvr.Prop_DisplayFrequency_Float,
                )
            )
        except Exception:
            return 11
        if frequency <= 1.0:
            return 11
        return max(11, int(round(1000.0 / min(frequency, 90.0))))

    def _cleanup_overlay_runtime(self) -> None:
        if self._overlay and self._overlay_handle:
            try:
                self._overlay.hideOverlay(self._overlay_handle)
                self._overlay.destroyOverlay(self._overlay_handle)
            except Exception:
                pass
        if self._overlay or self._vr_system or self._vr_active:
            try:
                openvr.shutdown()
            except Exception:
                pass
        self._overlay = None
        self._overlay_handle = None
        self._vr_system = None
        self._input_enabled = None
        self._last_alpha = None
        self._vr_active = False
        self._reset_head_follow_position_offsets()

    def _format_start_error(self, exc: Exception) -> str:
        if self._matches_init_error(exc, openvr.VRInitError_IPC_NamespaceUnavailable):
            current = current_integrity_level() or "Unknown"
            vrserver = process_integrity_level("vrserver.exe") or "Unavailable"
            steam = process_integrity_level("steam.exe") or "Unavailable"
            guidance = "Restart SteamVR and launch UltraTranslate at the same integrity level."
            if current == "High" and vrserver == "Medium":
                guidance = "UltraTranslate is elevated while SteamVR is not. Relaunch UltraTranslate from a normal shell."
            elif current == "Medium" and vrserver == "High":
                guidance = "SteamVR is elevated while UltraTranslate is not. Restart SteamVR normally."
            detail = openvr.getVRInitErrorAsEnglishDescription(openvr.VRInitError_IPC_NamespaceUnavailable)
            return f"SteamVR overlay unavailable: {detail}. UltraTranslate={current}, vrserver={vrserver}, steam={steam}. {guidance}"
        return f"SteamVR overlay unavailable: {exc}"

    @staticmethod
    def _matches_init_error(exc: Exception, code: int) -> bool:
        text = str(exc)
        return str(code) in text or openvr.getVRInitErrorAsEnglishDescription(code) in text

    @staticmethod
    def _rotation_angles(matrix: np.ndarray) -> tuple[float, float, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roll_deg, yaw_deg, pitch_deg = Rotation.from_matrix(matrix[:3, :3]).as_euler("zyx", degrees=True)
        return float(pitch_deg), float(yaw_deg), float(roll_deg)

    @staticmethod
    def _matrix_from_rotation_translation(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation[:3, :3]
        matrix[:3, 3] = np.asarray(position, dtype=np.float32)
        return matrix

    def _compose_numpy_matrix(
        self,
        x_pos: float,
        y_pos: float,
        z_pos: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
    ) -> np.ndarray:
        return self._matrix_to_numpy(
            self._compose_matrix(
                x_pos,
                y_pos,
                z_pos,
                pitch_deg,
                yaw_deg,
                roll_deg,
            )
        )

    @staticmethod
    def _mouse_scale(width: int = 900, height: int = 1180) -> openvr.HmdVector2_t:
        scale = openvr.HmdVector2_t()
        scale.v[0] = float(width)
        scale.v[1] = float(height)
        return scale

    @staticmethod
    def _compose_matrix(
        x_pos: float,
        y_pos: float,
        z_pos: float,
        pitch_deg: float,
        yaw_deg: float,
        roll_deg: float,
    ) -> openvr.HmdMatrix34_t:
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        roll = math.radians(roll_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cr, sr = math.cos(roll), math.sin(roll)
        rot = [
            [cy * cr, -cy * sr, sy],
            [sp * sy * cr + cp * sr, cp * cr - sp * sy * sr, -sp * cy],
            [sp * sr - cp * sy * cr, cp * sy * sr + sp * cr, cp * cy],
        ]
        mat = openvr.HmdMatrix34_t()
        mat.m[0][0] = rot[0][0]
        mat.m[0][1] = rot[0][1]
        mat.m[0][2] = rot[0][2]
        mat.m[0][3] = x_pos
        mat.m[1][0] = rot[1][0]
        mat.m[1][1] = rot[1][1]
        mat.m[1][2] = rot[1][2]
        mat.m[1][3] = y_pos
        mat.m[2][0] = rot[2][0]
        mat.m[2][1] = rot[2][1]
        mat.m[2][2] = rot[2][2]
        mat.m[2][3] = z_pos
        return mat

    @staticmethod
    def _matrix_to_numpy(matrix) -> np.ndarray:
        array = np.eye(4, dtype=np.float32)
        for row in range(3):
            for column in range(4):
                array[row, column] = matrix.m[row][column]
        return array

    @staticmethod
    def _numpy_to_matrix(matrix: np.ndarray) -> openvr.HmdMatrix34_t:
        output = openvr.HmdMatrix34_t()
        for row in range(3):
            for column in range(4):
                output.m[row][column] = float(matrix[row, column])
        return output
