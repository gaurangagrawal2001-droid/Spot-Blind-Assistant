"""Demo/inference controller: loads the trained PPO policy and runs it
continuously (no episodes/resets) so you can watch Spot walk the path,
stop for obstacles, and signal via console + LEDs.

This file is the "use the trained AI" counterpart to
spot_guide_dog_train.py (which is the "train the AI" side). It does NOT
use gymnasium.Env at all -- there's no reset(), no reward, no episodes --
because once a policy is trained we don't need any of that RL machinery
anymore. We just: read sensors -> ask the trained neural network what to
do -> act on its answer -> repeat, forever. This loop is often called
"inference" (using a trained model to make decisions) as opposed to
"training" (updating the model's weights based on reward).
"""
import os
import sys

import numpy as np
from controller import Robot

# Webots only puts each controller's own folder on sys.path, so the shared
# helper module (spot_guide_dog_lib.py, one level up) has to be added by hand
# before we can import from it. This line must run before the import below,
# which is why the import isn't at the very top of the file with the others.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "spot_guide_dog_lib"))
from spot_guide_dog_lib import (
    GaitController,
    LED_NAMES,
    MOTOR_NAMES,
    SENSOR_NAMES,
    set_leds,
)

SENSOR_MAX_RANGE_M = 1.5
MAX_STEPS_FOR_NORM = 300  # matches training's step-count normalization

# A plain Webots "Robot" (as opposed to the "Supervisor" used during
# training) can only read its own sensors and drive its own motors -- it
# can't teleport itself or other objects around the scene. That's fine
# here: unlike training, the demo never needs to reset anything, it just
# runs continuously.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

motors = [robot.getDevice(name) for name in MOTOR_NAMES]
leds = [robot.getDevice(name) for name in LED_NAMES]
sensors = [robot.getDevice(name) for name in SENSOR_NAMES]
for sensor in sensors:
    sensor.enable(timestep)

model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_spot_guide_dog.zip")
# Imported here (rather than at the top with the other imports) so this
# controller fails fast with the sensor/motor setup above already visible
# in Webots, and to keep the "load the trained model" step visually next
# to where it's actually used just below.
from stable_baselines3 import PPO

# PPO.load() reads back the neural network weights that
# spot_guide_dog_train.py saved to models/ppo_spot_guide_dog.zip after
# training. You must train at least once (or have a saved model already)
# before this line will succeed.
model = PPO.load(model_path)

gait = GaitController(motors, timestep)

last_action = 0
step_count = 0

# Webots calls robot.step(timestep) to advance the simulation by one
# timestep; it returns -1 once the simulation is told to stop (e.g. the
# Webots window closes), which ends this while loop.
while robot.step(timestep) != -1:
    # Build the exact same 7-number observation shape used during
    # training (see SpotGuideDogEnv._get_obs in spot_guide_dog_train.py) --
    # the trained network only knows how to interpret input in that same
    # shape, so this must stay in sync with the training environment.
    raw = [sensor.getValue() for sensor in sensors]
    norm = [float(np.clip(value / SENSOR_MAX_RANGE_M, 0.0, 1.0)) for value in raw]
    obs = np.array(
        norm + [float(last_action), min(1.0, step_count / MAX_STEPS_FOR_NORM)],
        dtype=np.float32,
    )

    # Ask the trained policy what to do. deterministic=True means "always
    # pick the network's single best action" rather than sampling randomly
    # from its predicted probabilities -- randomness is useful during
    # training (to explore), but for a demo we want the policy's most
    # confident choice every time.
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    if action == 1:  # GO
        gait.step_walk()
        set_leds(leds, "green")
        print("GO")
    else:  # STOP
        gait.step_stand()
        set_leds(leds, "red")
        print("STOP")

    last_action = action
    step_count += 1
