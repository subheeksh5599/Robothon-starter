"""HandiForge — Autonomous Precision Assembly with Franka Emika Panda.

7-DOF robotic arm sorting colored cubes to matching target zones.
Jacobian IK + autonomous stacking policy + keyboard teleop + multi-camera demo.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
    import glfw
except ImportError as exc:
    sys.exit(f"Missing: {exc}")

ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "scene.xml"

ARM_J = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
GRIPPER = "finger_joint1"

CUBES = ["cube_red","cube_blue","cube_green","cube_yellow"]
TARGETS = ["target_red","target_blue","target_green","target_yellow"]
STAGES = ["stage_red","stage_blue","stage_green","stage_yellow"]
CUBE_COLORS = {"cube_red":"red","cube_blue":"blue","cube_green":"green","cube_yellow":"yellow"}

ARM_HOME = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.8])

class HandiForge:
    def __init__(self, headless=False):
        self.m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        self.w, self.h = 640, 480

        jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_J]
        self._arm_qidx = [self.m.jnt_qposadr[i] for i in jids]
        self._arm_didx = [self.m.jnt_dofadr[i] for i in jids]
        self._ee = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self._jac = np.empty((6, self.m.nv))
        self._gripper_qid = self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER)]
        self._tendon_act = 7

        self.mode = "idle"
        self.arm_ctrl = ARM_HOME.copy()
        self.gripper_open = 0.04; self.gripper_closed = 0.0
        self.gripper_target = self.gripper_open
        self.stats = {"cubes_placed": 0, "grasp_attempts": 0, "cycles": 0, "score": 0}

        self._au = {"phase":"approach","cube_i":0,"tgt_i":0,"t":0.0}
        self.records = []

        self._r = None
        if not headless:
            self._r = mujoco.Renderer(self.m, self.h, self.w)

    def arm_q(self): return self.d.qpos[self._arm_qidx].copy()
    def ee(self):     return self.d.xpos[self._ee].copy()

    def _site_xyz(self, name):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.d.site_xpos[sid].copy()

    def _body_xyz(self, name):
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.d.xpos[bid].copy()

    def ik(self, xyz: np.ndarray) -> np.ndarray:
        e6 = np.zeros(6); e6[:3] = xyz - self.ee()
        mujoco.mj_jacBody(self.m, self.d, self._jac[:3], self._jac[3:], self._ee)
        J = self._jac[:, self._arm_didx]
        dq = J.T @ np.linalg.solve(J @ J.T + 5e-4*np.eye(6), e6 * 0.5)
        q = self.arm_q() + dq
        for i, jname in enumerate(ARM_J):
            lo, hi = self.m.jnt_range[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jname)]
            q[i] = np.clip(q[i], lo, hi)
        return 0.55 * q + 0.45 * self.arm_q()

    def step(self):
        self.d.ctrl[0:7] = self.arm_ctrl
        self.d.ctrl[self._tendon_act] = 255.0 * (self.gripper_target / 0.04)
        mujoco.mj_step(self.m, self.d)

    def record(self, t):
        self.records.append({
            "t": round(float(t),3), "arm": self.arm_q().round(4).tolist(),
            "gripper": round(float(self.d.qpos[self._gripper_qid]),4),
            "cubes": {n:self._body_xyz(n).round(3).tolist() for n in CUBES},
            "phase": self._au["phase"], "stats": dict(self.stats),
        })

    def cube_on_target(self, cube_name, target_name):
        cp = self._body_xyz(cube_name)
        tp = self._site_xyz(target_name)
        return np.linalg.norm(cp[:2] - tp[:2]) < 0.04 and abs(cp[2] - tp[2] - 0.03) < 0.04

    def autopilot(self):
        ci, ti = self._au["cube_i"], self._au["tgt_i"]
        cube_pos = self._body_xyz(CUBES[ci])
        stage_xyz = self._site_xyz(STAGES[ci])
        target_xyz = self._site_xyz(TARGETS[ti])
        ee = self.ee()
        ph = self._au["phase"]
        self._au["t"] += self.dt

        if ph == "approach":
            above = cube_pos + np.array([0, 0, 0.10])
            self.arm_ctrl = self.ik(above)
            self.gripper_target = self.gripper_open
            if np.linalg.norm(ee - above) < 0.04:
                self._au["phase"] = "descend"; self._au["t"] = 0

        elif ph == "descend":
            grasp = cube_pos + np.array([0, 0, 0.03])
            self.arm_ctrl = self.ik(grasp)
            if np.linalg.norm(ee - grasp) < 0.02:
                self._au["phase"] = "grasp"; self._au["t"] = 0
                self.stats["grasp_attempts"] += 1

        elif ph == "grasp":
            self.gripper_target = self.gripper_closed
            if self._au["t"] > 0.4:
                self._au["phase"] = "lift"; self._au["t"] = 0

        elif ph == "lift":
            above_tgt = target_xyz + np.array([0, 0, 0.16])
            self.arm_ctrl = self.ik(above_tgt)
            self.gripper_target = self.gripper_closed
            if np.linalg.norm(ee - above_tgt) < 0.04:
                self._au["phase"] = "place"; self._au["t"] = 0

        elif ph == "place":
            place = target_xyz + np.array([0, 0, 0.05])
            self.arm_ctrl = self.ik(place)
            self.gripper_target = self.gripper_closed
            if np.linalg.norm(ee - place) < 0.02:
                self._au["phase"] = "release"; self._au["t"] = 0

        elif ph == "release":
            self.gripper_target = self.gripper_open
            if self._au["t"] > 0.3:
                if self.cube_on_target(CUBES[ci], TARGETS[ti]):
                    self.stats["cubes_placed"] += 1
                    self.stats["score"] += 25
                self.stats["cycles"] += 1
                self._au["cube_i"] = (ci + 1) % 4
                self._au["tgt_i"] = (ti + 1) % 4
                self._au["phase"] = "approach"; self._au["t"] = 0

    def render(self, cam="front"):
        if self._r is None:
            self._r = mujoco.Renderer(self.m, self.h, self.w)
        self._r.update_scene(self.d, camera=cam)
        return self._r.render().copy()


def run_demo(out_vid, out_traj, duration=45, fps=30):
    f = HandiForge(headless=True)
    h = ARM_HOME.copy()
    # Phase 0: let cubes settle (1000 steps, no arm movement)
    for _ in range(1000):
        f.d.ctrl[0:7] = np.zeros(7); f.d.ctrl[7] = 255
        f.gripper_target = f.gripper_open
        mujoco.mj_step(f.m, f.d)
    # Phase 1: settle arm to home (200 steps)
    for _ in range(200):
        f.arm_ctrl = h; f.gripper_target = f.gripper_open
        f.step()
    f._au = {"phase":"approach","cube_i":0,"tgt_i":0,"t":0.0}
    nf = int(duration * fps)
    skip = max(1, int(1 / (fps * f.dt)))
    frames, t, ci = [], 0.0, 0
    cams = ["front","overhead","side","closeup"]

    print(f"HandiForge  |  {duration}s @ {fps}fps  |  {nf}f")
    for fi in range(nf):
        for _ in range(skip):
            f.mode = "autonomous"
            f.autopilot()
            f.step()
            t += f.dt
            if int(t*20) % 3 == 0: f.record(t)
        if fi % int(2.5*fps) == 0: ci = (ci+1)%4
        frames.append(f.render(cams[ci]))
        if fi % (5*fps) == 0 and fi:
            print(f"  {fi}/{nf} ({100*fi//nf}%)  placed:{f.stats['cubes_placed']}/{f.stats['cycles']}  score:{f.stats['score']}")

    out_vid.parent.mkdir(parents=True,exist_ok=True)
    try: iio.imwrite(out_vid, np.asarray(frames), fps=fps, codec="libx264")
    except: out_vid=out_vid.with_suffix(".gif"); iio.imwrite(out_vid, np.asarray(frames), fps=fps)

    out_traj.parent.mkdir(parents=True,exist_ok=True)
    out_traj.write_text(json.dumps({
        "project":"HandiForge — Precision Assembly",
        "robot":"Franka Emika Panda (7-DOF) with parallel-jaw gripper",
        "task":"Autonomous color-coded cube sorting and precision placement",
        "narrative":"A Franka Panda arm identifies and sorts colored cubes to matching target zones using inverse kinematics and autonomous state machine control. Demonstrates precision pick-and-place for automated assembly workflows.",
        "sensors":["framepos cube trackers (4x)"],
        "control":["Jacobian pseudoinverse IK","multi-phase autonomous policy",
                    "gripper force control","precision placement verification"],
        "stats":f.stats,
        "video":str(out_vid),"fps":fps,"duration":duration,
        "trajectory_points":len(f.records),
        "samples":f.records[::max(1,len(f.records)//150)],
    },indent=2))
    print(f"Done → {out_vid}  |  score:{f.stats['score']}/100")


def run_interactive():
    f = HandiForge(headless=False)
    glfw.init()
    win = glfw.create_window(f.w,f.h,"HandiForge — Cube Sorting",None,None)
    glfw.make_context_current(win)
    ctx = mujoco.MjrContext(f.m, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera(); opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(f.m,maxgeom=10000)
    vp = mujoco.MjrRect(0,0,f.w,f.h)
    cam.lookat[:]=[0.45,0,0.45]; cam.distance=1.5; cam.elevation=-28; cam.azimuth=140
    t=0.0
    print("\n═══ HandiForge ═══  A=Auto  T=Teleop  H=Home  ESC=Quit")
    def cb(win,k,sc,act,mods):
        nonlocal f
        if act not in (1,2): return
        s=0.05
        if k==256: glfw.set_window_should_close(win,True)
        elif k==65: f.mode="autonomous"; f._au["phase"]="approach"; f._au["t"]=0
        elif k==84: f.mode="teleop"
        elif k==72: f.mode="idle"; f.arm_ctrl=ARM_HOME.copy(); f.gripper_target=f.gripper_open
        elif f.mode=="teleop":
            c=f.arm_ctrl
            d={87:(0,s),83:(0,-s),68:(1,s),65:(1,-s),69:(2,s),81:(2,-s),
               74:(3,s),76:(3,-s),73:(4,s),75:(4,-s),85:(5,s),79:(5,-s)}
            if k in d: i,v=d[k]; c[i]+=v
            elif k==71: f.gripper_target=f.gripper_closed
            elif k==32: f.gripper_target=f.gripper_open
    glfw.set_key_callback(win,cb)
    while not glfw.window_should_close(win):
        if f.mode=="autonomous": f.autopilot()
        elif f.mode=="idle": f.arm_ctrl=ARM_HOME.copy(); f.gripper_target=f.gripper_open
        f.step(); t+=f.dt
        if int(t*20)%3==0: f.record(t)
        mujoco.mjv_updateScene(f.m,f.d,opt,mujoco.MjvPerturb(),cam,mujoco.mjtCatBit.mjCAT_ALL,scn)
        mujoco.mjr_render(vp,scn,ctx); glfw.swap_buffers(win); glfw.poll_events(); time.sleep(0.001)
    glfw.terminate()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--output",type=Path,default=ROOT/"outputs"/"demo.mp4")
    ap.add_argument("--trajectory",type=Path,default=ROOT/"outputs"/"trajectory.json")
    ap.add_argument("--duration",type=float,default=50.0)
    ap.add_argument("--fps",type=int,default=30)
    a = ap.parse_args()
    run_demo(a.output,a.trajectory,a.duration,a.fps) if a.demo else run_interactive()

if __name__=="__main__":
    main()
