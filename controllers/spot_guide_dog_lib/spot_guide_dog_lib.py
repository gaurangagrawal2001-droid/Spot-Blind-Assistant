"""Shared code for the Spot guide-dog controllers (train/demo/obstacle).

Imported via sys.path.append of this directory from sibling controllers,
since Webots only adds each controller's own directory to sys.path.

Webots basics for students: in Webots, a "controller" is just a Python
script attached to a robot in the simulation. The robot has "devices" --
motors, sensors, LEDs, etc. -- that you talk to by name (the same names you
gave them in the .wbt world file / robot's PROTO). You look a device up
with `robot.getDevice("device name")`, which is why you'll see long lists
of exact device name strings below: they have to match the world file
exactly, character for character.
"""

# Webots' Spot (Boston Dynamics-style quadruped) has 4 legs, each with 3
# motors: shoulder abduction (leg swings sideways, away from the body),
# shoulder rotation (leg swings forward/backward), and elbow (knee-like
# joint that bends the lower leg). That's 4 legs * 3 motors = 12 motors
# total. This list's order matters -- it's the same order used everywhere
# else in this file (STAND_POSE, WALK_POSES) and in the RL environment, so
# position i in any of those lists always refers to motor i here.
MOTOR_NAMES = [
    "front left shoulder abduction motor",
    "front left shoulder rotation motor",
    "front left elbow motor",
    "front right shoulder abduction motor",
    "front right shoulder rotation motor",
    "front right elbow motor",
    "rear left shoulder abduction motor",
    "rear left shoulder rotation motor",
    "rear left elbow motor",
    "rear right shoulder abduction motor",
    "rear right shoulder rotation motor",
    "rear right elbow motor",
]

# Spot has 8 LEDs (4 on each side) we use to visually signal the robot's
# current decision: green while walking ("GO"), red while stopped ("STOP").
# This is purely cosmetic -- it doesn't affect the AI or the physics -- but
# it's a handy way to see what the trained policy is "thinking" when you
# watch the demo run.
LED_NAMES = [
    "left top led",
    "left middle up led",
    "left middle down led",
    "left bottom led",
    "right top led",
    "right middle up led",
    "right middle down led",
    "right bottom led",
]

# 5 distance sensors ("ds" = distance sensor) mounted around Spot, used to
# detect obstacles. These are the robot's only "eyes" in this project: the
# RL agent never sees an image, only these 5 numbers (plus a couple of
# bookkeeping values), which is what makes this a small, easy-to-train
# observation space.
SENSOR_NAMES = ["ds_fan_0", "ds_fan_1", "ds_fan_2", "ds_fan_3", "ds_fan_4"]

# STAND_POSE is a target angle (in radians) for each of the 12 motors above,
# in the same order as MOTOR_NAMES. This is Spot standing still and level.
# Order per leg: [shoulder abduction, shoulder rotation, elbow]
STAND_POSE = [-0.1, 0.0, 0.0, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.0]

# Stylized quasi-static trot: diagonal pairs (FL+RR) and (FR+RL) alternate
# swinging forward (with a foot lift on the elbow) vs. pushing back.
# These exact angles are a starting guess -- tune visually in Webots
# (see plan verification step 1) before trusting them for training.
_SWING = 0.25  # how far a leg swings forward/backward while walking
_LIFT = -0.15  # how much the elbow bends to lift a foot off the ground
_PUSH = 0.05  # slight elbow bend on the "stance" (pushing) leg for grip

# WALK_POSES is a 4-step walking cycle: each entry is a full 12-motor pose
# (same format as STAND_POSE) that GaitController smoothly moves through,
# one after another, looping back to the start. Real quadrupeds don't
# actually pause in a neutral stance between swings, but including that
# neutral pose here keeps the motion simple and stable, which is more
# important than realism for a first version of this project.
WALK_POSES = [
    # FL+RR swing forward & lift, FR+RL push back (stance)
    [-0.1, _SWING, _LIFT, 0.1, -_SWING, _PUSH, -0.1, -_SWING, _PUSH, 0.1, _SWING, _LIFT],
    # neutral transition
    list(STAND_POSE),
    # FR+RL swing forward & lift, FL+RR push back (stance)
    [-0.1, -_SWING, _PUSH, 0.1, _SWING, _LIFT, -0.1, _SWING, _LIFT, 0.1, -_SWING, _PUSH],
    # neutral transition
    list(STAND_POSE),
]

# LED colors as 24-bit hex (0xRRGGBB), the format Webots' LED.set() expects.
LED_COLORS = {"red": 0xFF0000, "green": 0x00FF00, "off": 0x000000}


def set_leds(led_devices, color):
    """Set every LED in led_devices to the given named color ("red",
    "green", or "off"). led_devices is a list of Webots LED device objects,
    e.g. the `leds` list built from LED_NAMES via robot.getDevice()."""
    value = LED_COLORS[color]
    for led in led_devices:
        led.set(value)


class GaitController:
    """Non-blocking, incremental version of the stand/sit interpolation
    pattern from spot_stand_sit.py. Each step_walk()/step_stand() call
    advances the motor targets by exactly one timestep's worth of
    interpolation and returns -- it never calls robot.step() itself, so
    the caller (the RL env or the demo loop) stays in control of stepping
    the simulation exactly once per control tick.

    Why "non-blocking"? A naive gait implementation might do something like
    "move to pose A, then sleep, then move to pose B" in one function call.
    That would freeze whoever calls it until the whole motion finishes. This
    class instead does a *tiny* bit of motion each time you call
    step_walk()/step_stand(), so the RL training loop (or the demo loop)
    can call robot.step() every single simulation tick and still choose a
    new action (GO/STOP) on the very next tick if it wants to.
    """

    def __init__(self, motors, timestep, pose_duration_ms=400):
        self.motors = motors
        self.timestep = timestep  # simulation milliseconds per step() call
        self.pose_duration_ms = pose_duration_ms  # time to blend into a new pose
        self.current_positions = list(STAND_POSE)
        self.segment_start = list(STAND_POSE)
        self.segment_target = list(STAND_POSE)
        self.segment_elapsed_ms = 0
        self.walk_cycle_index = 0
        self._mode = "stand"
        self._apply(self.current_positions)

    def _apply(self, positions):
        # Send each motor its target angle. setPosition() is Webots'
        # built-in way to command a rotational motor -- the physics engine
        # then drives the joint toward that angle over subsequent steps.
        for motor, pos in zip(self.motors, positions):
            motor.setPosition(pos)

    def _start_segment(self, target):
        # Begin smoothly moving from wherever the legs currently are to a
        # new target pose, over the next pose_duration_ms milliseconds.
        self.segment_start = list(self.current_positions)
        self.segment_target = list(target)
        self.segment_elapsed_ms = 0

    def _advance_segment(self):
        # Linear interpolation ("lerp"): move `progress` of the way from
        # segment_start to segment_target, where progress goes from 0.0
        # (just started) to 1.0 (fully arrived) as time elapses. This is
        # what makes the legs glide smoothly between poses instead of
        # snapping instantly, which would look jerky and stress the physics
        # engine.
        self.segment_elapsed_ms += self.timestep
        progress = min(1.0, self.segment_elapsed_ms / self.pose_duration_ms)
        self.current_positions = [
            start + (target - start) * progress
            for start, target in zip(self.segment_start, self.segment_target)
        ]
        self._apply(self.current_positions)
        return progress >= 1.0  # True once this segment has fully arrived

    def step_walk(self):
        """Advance one timestep's worth of walking motion. Call this every
        simulation step while the AI's chosen action is "GO"."""
        if self._mode != "walk":
            # Just switched into walking mode: start the very first leg
            # of the walk cycle from wherever the robot currently is.
            self._mode = "walk"
            self.walk_cycle_index = 0
            self._start_segment(WALK_POSES[self.walk_cycle_index])
        reached = self._advance_segment()
        if reached:
            # Finished blending into the current pose in the cycle --
            # move on to the next one (wrapping back to 0 after the last).
            self.walk_cycle_index = (self.walk_cycle_index + 1) % len(WALK_POSES)
            self._start_segment(WALK_POSES[self.walk_cycle_index])

    def step_stand(self):
        """Advance one timestep's worth of motion back toward the standing
        pose. Call this every simulation step while the AI's chosen action
        is "STOP"."""
        if self._mode != "stand":
            self._mode = "stand"
            self._start_segment(STAND_POSE)
        self._advance_segment()

    def reset_to_stand(self):
        """Snap immediately (no smoothing) back to the standing pose. Used
        at the start of every RL episode so each episode begins from the
        exact same, known posture."""
        self._mode = "stand"
        self.current_positions = list(STAND_POSE)
        self.segment_start = list(STAND_POSE)
        self.segment_target = list(STAND_POSE)
        self.segment_elapsed_ms = 0
        self._apply(self.current_positions)
