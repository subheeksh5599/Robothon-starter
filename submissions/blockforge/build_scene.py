"""Build composite scene: Franka Panda arm + LEAP Hand + precision peg-in-hole task."""
from __future__ import annotations
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Read the body chain from panda_nohand.xml (everything inside worldbody)
# and the actuator section
MENAGERIE = Path("/tmp/mujoco_menagerie")
panda_nohand = (MENAGERIE / "franka_emika_panda" / "panda_nohand.xml").read_text(encoding="utf-8")
leap_hand = (MENAGERIE / "leap_hand" / "right_hand.xml").read_text(encoding="utf-8")

def inner(xml: str, tag: str) -> str:
    """Extract everything between <tag ...> and </tag>."""
    import re
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
    if not m:
        raise ValueError(f"Tag <{tag}> not found in XML")
    return m.group(1).strip()


# Method: include both full models via <include> and weld the hand to the arm.
# This avoids XML surgery entirely.

scene = """<mujoco model="blockforge_pro">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option integrator="implicitfast" impratio="10" cone="elliptic"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="130" elevation="-25" offwidth="640" offheight="480"/>
  </visual>

  <default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
    </default>
    <default class="panda/visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2"/>
    </default>
    <default class="panda/collision">
      <geom type="mesh" group="3"/>
    </default>

    <!-- LEAP hand defaults at root level -->
    <geom solimp="0.999 0.999 0.001 0.0001 1" solref="0.0001 1" friction=".2"/>
    <position kp="3.0" kv="0.01"/>
    <joint damping="0.03" frictionloss="0.001"/>

    <default class="visual_leap">
      <geom group="1" type="mesh" contype="0" conaffinity="0" density="0" material="leap_black"/>
    </default>
    <default class="collision_leap">
      <geom material="leap_black"/>
    </default>
    <default class="tip">
      <geom type="mesh" mesh="tip" friction="0.5" material="leap_white"/>
    </default>
    <default class="thumb_tip">
      <geom type="mesh" mesh="thumb_tip" friction="0.5" material="leap_white"/>
    </default>
    <default class="mcp">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-0.314 2.23"/>
      <position ctrlrange="-0.314 2.23"/>
    </default>
    <default class="rot">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-1.047 1.047"/>
      <position ctrlrange="-1.047 1.047"/>
    </default>
    <default class="pip">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-0.506 1.885"/>
      <position ctrlrange="-0.506 1.885"/>
    </default>
    <default class="dip">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-0.366 2.042"/>
      <position ctrlrange="-0.366 2.042"/>
    </default>
    <default class="thumb_cmc">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-0.349 2.094"/>
      <position ctrlrange="-0.349 2.094"/>
    </default>
    <default class="thumb_axl">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-0.349 2.094"/>
      <position ctrlrange="-0.349 2.094"/>
    </default>
    <default class="thumb_mcp">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-0.47 2.443"/>
      <position ctrlrange="-0.47 2.443"/>
    </default>
    <default class="thumb_ipl">
      <joint pos="0 0 0" axis="0 0 -1" limited="true" range="-1.34 1.88"/>
      <position ctrlrange="-1.34 1.88"/>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.2 0.3 0.5" rgb2="0.05 0.05 0.1"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.25 0.28 0.35" rgb2="0.18 0.20 0.25" markrgb="0.4 0.4 0.5"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="6 6" reflectance="0.15"/>

    <material name="table_surface" rgba="0.55 0.45 0.35 1"/>
    <material name="table_leg" rgba="0.4 0.3 0.2 1"/>
    <material name="wall_grey" rgba="0.5 0.5 0.5 1"/>

    <material name="peg_red" rgba="0.95 0.2 0.2 1"/>
    <material name="peg_green" rgba="0.2 0.85 0.3 1"/>
    <material name="peg_blue" rgba="0.2 0.4 0.95 1"/>
    <material name="peg_yellow" rgba="0.95 0.85 0.15 1"/>
    <material name="peg_purple" rgba="0.7 0.2 0.9 1"/>

    <!-- Panda meshes -->
__PANDA_MESHES__

    <!-- LEAP meshes -->
__LEAP_MESHES__
  </asset>

  <worldbody>
    <light name="key" pos="2 0 2.5" dir="-0.5 0 -1" directional="false"
      diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <light name="fill" pos="-1 2 1.5" dir="0.5 -1 -0.5" directional="false"
      diffuse="0.4 0.4 0.5" specular="0 0 0"/>
    <geom name="floor" size="0 0 0.025" type="plane" material="groundplane"/>

    <!-- === FIXED BASE for Franka arm === -->

    <body name="panda_base" pos="0 0 0">
      __PANDA_BODY_CHAIN__
    </body>

    <!-- === LEAP HAND (attached via weld to panda attachment body) === -->

    <!-- Weld target site on the Franka attachment body -->
    <!-- We add a helper body that the equality weld will anchor to -->
    <body name="leap_anchor" pos="0 0 0" mocap="true">
      <geom type="sphere" size="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>

    __LEAP_BODY_CHAIN__

    <!-- Table + task objects -->
    <body name="table" pos="0.45 -0.15 0.0">
      <geom name="table_top" type="box" size="0.4 0.35 0.02" pos="0 0 0.78" material="table_surface"/>
      <geom name="leg_fl" type="cylinder" size="0.025 0.38" pos="-0.3 -0.25 0.4" material="table_leg"/>
      <geom name="leg_fr" type="cylinder" size="0.025 0.38" pos="0.3 -0.25 0.4" material="table_leg"/>
      <geom name="leg_bl" type="cylinder" size="0.025 0.38" pos="-0.3 0.25 0.4" material="table_leg"/>
      <geom name="leg_br" type="cylinder" size="0.025 0.38" pos="0.3 0.25 0.4" material="table_leg"/>
    </body>

    <!-- Hole board (precision targets) -->
    <body name="hole_board" pos="0.45 -0.15 0.80">
      <geom name="board_base" type="box" size="0.18 0.18 0.015" material="table_leg"/>
      <site name="hole_a" type="cylinder" size="0.012 0.001" pos="-0.07 -0.07 0.015" rgba="0.95 0.1 0.1 0.5"/>
      <site name="hole_b" type="cylinder" size="0.012 0.001" pos="0.07 -0.07 0.015" rgba="0.1 0.9 0.1 0.5"/>
      <site name="hole_c" type="cylinder" size="0.012 0.001" pos="-0.07 0.07 0.015" rgba="0.1 0.4 0.95 0.5"/>
      <site name="hole_d" type="cylinder" size="0.012 0.001" pos="0.07 0.07 0.015" rgba="0.9 0.85 0.1 0.5"/>
    </body>

    <!-- Small cylinders for insertion task -->
    <body name="cylinder_red" pos="0.4 -0.1 0.90">
      <freejoint/>
      <geom name="cyl_red" type="cylinder" size="0.01 0.035" material="peg_red" mass="0.015"/>
    </body>
    <body name="cylinder_green" pos="0.5 -0.1 0.90">
      <freejoint/>
      <geom name="cyl_green" type="cylinder" size="0.01 0.035" material="peg_green" mass="0.015"/>
    </body>
    <body name="cylinder_blue" pos="0.5 -0.05 0.90">
      <freejoint/>
      <geom name="cyl_blue" type="cylinder" size="0.01 0.035" material="peg_blue" mass="0.015"/>
    </body>
    <body name="cylinder_purple" pos="0.4 -0.05 0.90">
      <freejoint/>
      <geom name="cyl_purple" type="cylinder" size="0.01 0.035" material="peg_purple" mass="0.015"/>
    </body>

    <!-- Containment walls -->
    <geom name="wall_left" type="box" size="0.02 0.35 0.12" pos="0.05 -0.15 0.88" material="wall_grey"/>
    <geom name="wall_right" type="box" size="0.02 0.35 0.12" pos="0.85 -0.15 0.88" material="wall_grey"/>
    <geom name="wall_back" type="box" size="0.42 0.02 0.12" pos="0.45 0.2 0.88" material="wall_grey"/>

    <camera name="overhead" pos="0.45 -0.15 1.5" xyaxes="1 0 0 0 1 0" mode="fixed"/>
    <camera name="front" pos="0.45 -0.9 1.15" xyaxes="1 0 0 0 0.4 0.916" mode="fixed"/>
    <camera name="closeup" pos="0.45 -0.35 1.05" xyaxes="1 0 0 0 0.6 0.8" mode="fixed"/>
  </worldbody>

__PANDA_ACTUATORS__
__LEAP_ACTUATORS__

  <sensor>
    <!-- LEAP finger sensors -->
    __LEAP_SENSORS__
    <!-- Cylinder tracking -->
    <framepos name="cyl_red_pos" objtype="body" objname="cylinder_red"/>
    <framepos name="cyl_green_pos" objtype="body" objname="cylinder_green"/>
    <framepos name="cyl_blue_pos" objtype="body" objname="cylinder_blue"/>
    <framepos name="cyl_purple_pos" objtype="body" objname="cylinder_purple"/>
    <!-- Fingertip touch -->
    <touch name="if_tip_touch" site="if_tip_site"/>
    <touch name="mf_tip_touch" site="mf_tip_site"/>
    <touch name="rf_tip_touch" site="rf_tip_site"/>
    <touch name="th_tip_touch" site="th_tip_site"/>
    <force name="if_tip_force" site="if_tip_site"/>
    <force name="mf_tip_force" site="mf_tip_site"/>
    <force name="rf_tip_force" site="rf_tip_site"/>
    <force name="th_tip_force" site="th_tip_site"/>
  </sensor>

  <equality>
    <!-- Weld LEAP hand palm to Franka attachment body -->
    <weld name="hand_to_arm" body1="attachment" body2="palm"
      solimp="0.95 0.99 0.001" solref="0.002 1"/>
  </equality>

  <contact>
    __PANDA_CONTACTS__
    __LEAP_CONTACTS__
  </contact>

  <keyframe>
    <key name="home" qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0"
      ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>"""

# Extract what we need from source XMLs
panda_asset_inner = inner(panda_nohand, "asset")
panda_wb_inner = inner(panda_nohand, "worldbody")
panda_act_inner = inner(panda_nohand, "actuator")
panda_ctc_inner = inner(panda_nohand, "contact")

leap_asset_inner = inner(leap_hand, "asset")
# Rename to avoid conflicts with Panda materials
leap_asset_inner = leap_asset_inner.replace('name="black"', 'name="leap_black"')
leap_asset_inner = leap_asset_inner.replace('name="white"', 'name="leap_white"')
# Also fix LEAP default materials referencing old names
leap_asset_inner = leap_asset_inner.replace('material="black"', 'material="leap_black"')
leap_asset_inner = leap_asset_inner.replace('material="white"', 'material="leap_white"')
leap_wb_inner = inner(leap_hand, "worldbody")
leap_act_inner = inner(leap_hand, "actuator")
leap_sensor_inner = inner(leap_hand, "sensor")
leap_ctc_inner = inner(leap_hand, "contact")

# Patch LEAP body tree to use modified class names  
leap_wb_inner = leap_wb_inner.replace('class="visual"', 'class="visual_leap"')
leap_wb_inner = leap_wb_inner.replace('class="collision"', 'class="collision_leap"')

# Add fingertip touch sites to LEAP hand palm body
leap_wb_inner = leap_wb_inner.replace(
    '<geom name="if_tip" class="tip"/>',
    '<geom name="if_tip" class="tip"/>\n              <site name="if_tip_site" size="0.003"/>'
)
leap_wb_inner = leap_wb_inner.replace(
    '<geom name="mf_tip" class="tip"/>',
    '<geom name="mf_tip" class="tip"/>\n              <site name="mf_tip_site" size="0.003"/>'
)
leap_wb_inner = leap_wb_inner.replace(
    '<geom name="rf_tip" class="tip"/>',
    '<geom name="rf_tip" class="tip"/>\n              <site name="rf_tip_site" size="0.003"/>'
)
leap_wb_inner = leap_wb_inner.replace(
    '<geom name="th_tip" class="thumb_tip"/>',
    '<geom name="th_tip" class="thumb_tip"/>\n              <site name="th_tip_site" size="0.003"/>'
)

# Substitute
scene = scene.replace("__PANDA_MESHES__", panda_asset_inner)
scene = scene.replace("__LEAP_MESHES__", leap_asset_inner)
scene = scene.replace("__PANDA_BODY_CHAIN__", panda_wb_inner)
scene = scene.replace("__LEAP_BODY_CHAIN__", leap_wb_inner)
scene = scene.replace("__PANDA_ACTUATORS__", f"<actuator>\n{panda_act_inner}\n  </actuator>")
scene = scene.replace("__LEAP_ACTUATORS__", f"<actuator>\n{leap_act_inner}\n  </actuator>")
scene = scene.replace("__LEAP_SENSORS__", leap_sensor_inner)
scene = scene.replace("__PANDA_CONTACTS__", panda_ctc_inner)
scene = scene.replace("__LEAP_CONTACTS__", leap_ctc_inner)

out = HERE / "scene.xml"
out.write_text(scene, encoding="utf-8")
print(f"Written {out} ({len(scene)} chars)")
print("Substitutions verified:", all(marker not in scene for marker in [
    "__PANDA_MESHES__", "__LEAP_MESHES__", "__PANDA_BODY_CHAIN__",
    "__LEAP_BODY_CHAIN__", "__PANDA_ACTUATORS__", "__LEAP_ACTUATORS__",
    "__LEAP_SENSORS__", "__PANDA_CONTACTS__", "__LEAP_CONTACTS__"
]))
