"""Self-supervising controller for the moving obstacle box.

Moves back and forth sinusoidally across the path. This is random every
time, because the training controller (spot_guide_dog_train.py) writes a
new random phase/period into this node's "customData" field at the start
of every RL episode -- see this file's while loop below for how that gets
picked up.

This controller is its OWN Supervisor (separate from Spot's), attached to
the obstacle box itself -- in Webots, each object that needs to control
its own behavior gets its own controller script. Spot's Supervisor can
still reach into this node from the outside (to randomize it), but this
script is what actually moves it every simulation step.
"""
import math

from controller import Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
# "translation" is the field in the world file holding this object's
# [x, y, z] position; getField() + getSFVec3f()/setSFVec3f() is the
# standard Webots way to read/write it from a controller.
translation_field = robot.getSelf().getField("translation")

AMPLITUDE = 0.5  # how far (meters) the obstacle swings from its center
DEFAULT_PERIOD_S = 4.0  # how many seconds one full back-and-forth swing takes, if not overridden


def parse_custom_data(raw):
    """customData is just a plain string field Webots lets any controller
    read or write -- it's a simple way to pass small bits of data between
    two different controllers (here: Spot's training Supervisor telling
    this obstacle "start your sine wave at this phase, with this period").
    We encode it as "phase=1.23;period=4.56" and parse it back out here."""
    phase = 0.0
    period = DEFAULT_PERIOD_S
    for part in raw.split(";"):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        try:
            if key == "phase":
                phase = float(value)
            elif key == "period":
                period = float(value)
        except ValueError:
            pass
    return phase, period


last_custom_data = None
center_y = translation_field.getSFVec3f()[1]
t = 0.0  # elapsed time (seconds) since the current oscillation started

while robot.step(timestep) != -1:
    t += timestep / 1000.0  # timestep is in milliseconds; convert to seconds
    raw = robot.getCustomData()
    if raw != last_custom_data:
        # Supervisor just wrote a new randomization (and set translation
        # to the new center position right before this) -- resync so the
        # oscillation is centered on the new spot instead of the old one.
        last_custom_data = raw
        center_y = translation_field.getSFVec3f()[1]
        t = 0.0
    phase, period = parse_custom_data(raw)
    # Classic sine-wave motion: y oscillates smoothly between
    # center_y - AMPLITUDE and center_y + AMPLITUDE, completing one full
    # cycle every `period` seconds, offset by `phase` (so different
    # episodes don't all start the swing at the exact same point).
    y = center_y + AMPLITUDE * math.sin(2 * math.pi * t / period + phase)
    pos = translation_field.getSFVec3f()
    translation_field.setSFVec3f([pos[0], y, pos[2]])
