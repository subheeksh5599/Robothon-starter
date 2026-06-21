from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
    import glfw
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}")

ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "scene.xml"
assert SCENE_XML.exists(), f"Scene not found: {SCENE_XML}"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
HAND_JOINTS = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]
CYLINDER_NAMES = ["cylinder_red", "cylinder_green", "cylinder_blue", "cylinder_purple"]
HOLE_SITES = ["hole_a", "hole_b", "hole_c", "hole_d"]

OPEN_HAND = np.array([0.1] * 16)
CLOSE_HAND = np.array([
    2.0, 0.0, 1.5, 1.8,   # index: mcp, rot, pip, dip
    2.0, 0.0, 1.5, 1.8,   # middle
    2.0, 0.0, 1.5, 1.8,   # ring
    2.0, 0.0, 2.0, 1.5,   # thumb: cmc, axl, mcp, ipl
])
HOME_ARM = np.array([0, -0.6, 0, -2.2, 0, 2.0, 0.8])


class BlockForgePro:
    def __init__(self, headless: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.headless = headless

        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.arm_qpos_idx = [self.model.jnt_qposadr[i] for i in self.arm_joint_ids]
        self.arm_dof_idx = [self.model.jnt_dofadr[i] for i in self.arm_joint_ids]
        self.nv_arm = len(self.arm_dof_idx)

        self.hand_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in HAND_JOINTS]
        self.hand_qpos_idx = [self.model.jnt_qposadr[i] for i in self.hand_joint_ids]
        self.hand_dof_idx = [self.model.jnt_dofadr[i] for i in self.hand_joint_ids]
        self.nv_hand = len(self.hand_dof_idx)

        all_acts = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        self.arm_act_idx = [i for i, n in enumerate(all_acts) if n and n.startswith("actuator")]
        self.hand_act_idx = [i for i, n in enumerate(all_acts) if n and "act" in n and "actuator" not in n]

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")  # LEAP hand base

        self.jac = np.empty((6, self.model.nv))
        self.diag = 1e-3 * np.eye(6)

        self.mode = "idle"
        self.arm_ctrl = HOME_ARM.copy()
        self.hand_ctrl = OPEN_HAND.copy()
        self.recorded: list[dict] = []
        self._dt = self.model.opt.timestep

        self.renderer: mujoco.Renderer | None = None
        self.width = 640
        self.height = 480
        if not headless:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)

        self._auto: dict = {
            "phase": "approach",
            "cyl_idx": 0,
            "hole_idx": 0,
            "phase_t": 0.0,
        }

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_idx].copy()

    def hand_qpos(self) -> np.ndarray:
        return self.data.qpos[self.hand_qpos_idx].copy()

    def set_arm(self, values: np.ndarray):
        for idx, v in zip(self.arm_qpos_idx, values):
            self.data.qpos[idx] = v

    def set_hand(self, values: np.ndarray):
        for idx, v in zip(self.hand_qpos_idx, values):
            self.data.qpos[idx] = v

    def ik_solve(self, target_pos: np.ndarray, target_quat: np.ndarray | None = None) -> np.ndarray:
        """Jacobian pseudoinverse IK for the arm."""
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        pos_err = target_pos - ee_pos
        mujoco.mj_jacBody(self.model, self.data, self.jac[:3], self.jac[3:], self.ee_body_id)

        if target_quat is not None:
            ee_mat = self.data.xmat[self.ee_body_id].reshape(3, 3)
            tmat = np.empty(9)
            mujoco.mju_quat2Mat(tmat, target_quat)
            tmat = tmat.reshape(3, 3)
            R_err = tmat @ ee_mat.T
            orn_err = np.zeros(3)
            mujoco.mju_mat2Vel(orn_err, R_err.flatten(), 1.0)
            err = np.concatenate([pos_err, orn_err])
        else:
            err = np.concatenate([pos_err, np.zeros(3)])

        J = self.jac[:, self.arm_dof_idx]
        delta_q = J.T @ np.linalg.solve(J @ J.T + self.diag, err * 0.5)
        current = self.arm_qpos()
        raw_target = current + delta_q
        for i, jid in enumerate(self.arm_joint_ids):
            lo, hi = self.model.jnt_range[jid]
            raw_target[i] = np.clip(raw_target[i], lo, hi)
        # Smooth interpolation to avoid large accelerations
        blend = 0.25  # blend factor per call (low-pass)
        return blend * raw_target + (1 - blend) * current

    def cylindrical_to_xyz(self, cname: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cname)
        return self.data.xpos[bid].copy() if bid >= 0 else np.zeros(3)

    def hole_xyz(self, hname: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, hname)
        return self.data.site_xpos[sid].copy() if sid >= 0 else np.zeros(3)

    def step(self):
        self.data.ctrl[self.arm_act_idx] = self.arm_ctrl
        self.data.ctrl[self.hand_act_idx] = self.hand_ctrl
        mujoco.mj_step(self.model, self.data)

    def record(self, t: float):
        self.recorded.append({
            "t": round(float(t), 3),
            "arm": self.arm_qpos().round(4).tolist(),
            "hand": self.hand_qpos().round(4).tolist(),
            "cylinders": {n: self.cylindrical_to_xyz(n).round(4).tolist() for n in CYLINDER_NAMES},
            "mode": self.mode,
        })

    def autonomous_step(self):
        self._auto["phase_t"] += self._dt
        cyl_name = CYLINDER_NAMES[self._auto["cyl_idx"]]
        hole_name = HOLE_SITES[self._auto["hole_idx"]]
        cyl_pos = self.cylindrical_to_xyz(cyl_name)
        hole_pos = self.hole_xyz(hole_name)
        ee_pos = self.data.xpos[self.ee_body_id].copy()

        phase = self._auto["phase"]

        if phase == "approach":
            above = cyl_pos + np.array([0, 0, 0.15])
            self.arm_ctrl = self.ik_solve(above)
            self.hand_ctrl = OPEN_HAND
            if np.linalg.norm(ee_pos - above) < 0.03:
                self._auto["phase"] = "descend"
                self._auto["phase_t"] = 0

        elif phase == "descend":
            grasp = cyl_pos + np.array([0, 0, 0.04])
            self.arm_ctrl = self.ik_solve(grasp)
            self.hand_ctrl = OPEN_HAND
            if np.linalg.norm(ee_pos - grasp) < 0.02:
                self._auto["phase"] = "close"
                self._auto["phase_t"] = 0

        elif phase == "close":
            self.hand_ctrl = CLOSE_HAND
            if self._auto["phase_t"] > 0.6:
                self._auto["phase"] = "lift"
                self._auto["phase_t"] = 0

        elif phase == "lift":
            above = hole_pos + np.array([0, 0, 0.18])
            self.arm_ctrl = self.ik_solve(above)
            self.hand_ctrl = CLOSE_HAND
            if np.linalg.norm(ee_pos - above) < 0.03:
                self._auto["phase"] = "hover"
                self._auto["phase_t"] = 0

        elif phase == "hover":
            above = hole_pos + np.array([0, 0, 0.12])
            self.arm_ctrl = self.ik_solve(above)
            self.hand_ctrl = CLOSE_HAND
            if np.linalg.norm(ee_pos - above) < 0.02:
                self._auto["phase"] = "insert"
                self._auto["phase_t"] = 0

        elif phase == "insert":
            insert = hole_pos + np.array([0, 0, 0.06])
            self.arm_ctrl = self.ik_solve(insert)
            self.hand_ctrl = CLOSE_HAND
            if self._auto["phase_t"] > 1.0:
                self._auto["phase"] = "release"
                self._auto["phase_t"] = 0

        elif phase == "release":
            self.hand_ctrl = OPEN_HAND
            if self._auto["phase_t"] > 0.5:
                self._auto["cyl_idx"] = (self._auto["cyl_idx"] + 1) % len(CYLINDER_NAMES)
                self._auto["hole_idx"] = (self._auto["hole_idx"] + 1) % len(HOLE_SITES)
                self._auto["phase"] = "approach"
                self._auto["phase_t"] = 0

    def render_frame(self, cam: str = "overhead") -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()


def run_demo(video_path: Path, traj_path: Path, duration: float = 30.0, fps: int = 30):
    forge = BlockForgePro(headless=True)
    total_frames = int(duration * fps)
    frame_skip = max(1, int(1.0 / (fps * forge._dt)))
    frames: list[np.ndarray] = []
    sim_t = 0.0
    cam = "front"

    print(f"Recording {duration}s demo ({total_frames} frames) at {fps}fps...")

    for fi in range(total_frames):
        for _ in range(frame_skip):
            forge.mode = "autonomous"
            forge.autonomous_step()
            forge.step()
            sim_t += forge._dt
            if int(sim_t * 20) % 3 == 0:
                forge.record(sim_t)

        if fi % (2 * fps) == 0:
            cam = "front" if cam == "overhead" else "overhead"
        frames.append(forge.render_frame(cam))

        if fi % (fps * 5) == 0 and fi > 0:
            print(f"  {fi}/{total_frames} ({100*fi//total_frames}%)")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing video...")
    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
    except Exception as exc:
        video_path = video_path.with_suffix(".gif")
        iio.imwrite(video_path, np.asarray(frames), fps=fps)

    summary = {
        "project": "BlockForge Pro",
        "robot": "Franka Emika Panda + LEAP Hand (16-DOF dexterous)",
        "task": "Precision cylinder insertion into target holes",
        "video": str(video_path),
        "duration_s": duration,
        "fps": fps,
        "trajectory_points": len(forge.recorded),
        "samples": forge.recorded[::max(1, len(forge.recorded) // 200)],
    }
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {video_path}")
    return summary


def run_interactive():
    forge = BlockForgePro(headless=False)

    if not glfw.init():
        sys.exit("GLFW init failed")
    window = glfw.create_window(forge.width, forge.height,
                                 "BlockForge Pro - Panda + LEAP Hand", None, None)
    glfw.make_context_current(window)

    ctx = mujoco.MjrContext(forge.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(forge.model, maxgeom=10000)
    vp = mujoco.MjrRect(0, 0, forge.width, forge.height)
    cam.lookat[:] = [0.45, -0.15, 0.75]
    cam.distance = 1.8
    cam.elevation = -28
    cam.azimuth = 130

    sim_t = 0.0

    print("\n=== BlockForge Pro ===")
    print("  A = Autonomous  |  T = Teleop  |  H = Home")
    print("  ESC = Exit")
    print("  Teleop: W/S=arm_X  A/D=arm_Y  Q/E=arm_Z  J/L=yaw  I/K=pitch  U/O=roll")
    print("          1-4 = finger presets  |  G = grip  |  Space = open")

    def kb(win, key, scancode, action, mods):
        nonlocal forge
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        s = 0.05
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_A:
            forge.mode = "autonomous"
            print("[AUTONOMOUS]")
        elif key == glfw.KEY_T:
            forge.mode = "teleop"
            print("[TELEOP]")
        elif key == glfw.KEY_H:
            forge.mode = "idle"
            forge.arm_ctrl = HOME_ARM.copy()
            forge.hand_ctrl = OPEN_HAND.copy()
            print("[IDLE]")
        elif forge.mode == "teleop":
            c = forge.arm_ctrl
            if key == glfw.KEY_W: c[0] += s
            elif key == glfw.KEY_S: c[0] -= s
            elif key == glfw.KEY_D: c[1] += s
            elif key == glfw.KEY_A: c[1] -= s
            elif key == glfw.KEY_E: c[2] += s
            elif key == glfw.KEY_Q: c[2] -= s
            elif key == glfw.KEY_J: c[3] += s
            elif key == glfw.KEY_L: c[3] -= s
            elif key == glfw.KEY_I: c[4] += s
            elif key == glfw.KEY_K: c[4] -= s
            elif key == glfw.KEY_U: c[5] += s
            elif key == glfw.KEY_O: c[5] -= s
            elif key == glfw.KEY_1: forge.hand_ctrl = OPEN_HAND
            elif key == glfw.KEY_2: forge.hand_ctrl = CLOSE_HAND * 0.5
            elif key == glfw.KEY_3: forge.hand_ctrl = CLOSE_HAND
            elif key == glfw.KEY_G: forge.hand_ctrl = CLOSE_HAND
            elif key == glfw.KEY_SPACE: forge.hand_ctrl = OPEN_HAND
    glfw.set_key_callback(window, kb)

    while not glfw.window_should_close(window):
        if forge.mode == "autonomous":
            forge.autonomous_step()
        elif forge.mode == "idle":
            forge.arm_ctrl = HOME_ARM.copy()
            forge.hand_ctrl = OPEN_HAND.copy()
        forge.step()
        sim_t += forge._dt
        if int(sim_t * 20) % 3 == 0:
            forge.record(sim_t)

        mujoco.mjv_updateScene(forge.model, forge.data, opt, mujoco.MjvPerturb(),
                                cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(vp, scn, ctx)
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.001)

    glfw.terminate()
    if forge.recorded:
        out = ROOT / "outputs" / "interactive.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"points": len(forge.recorded)}, indent=2))
        print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser(description="BlockForge Pro - Panda + LEAP Hand")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "demo.mp4")
    p.add_argument("--trajectory", type=Path, default=ROOT / "outputs" / "trajectory.json")
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if args.demo:
        run_demo(args.output, args.trajectory, args.duration, args.fps)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
