"""Training controller: SpotGuideDogEnv is a gymnasium.Env that composes a
Webots Supervisor (not multiple-inheritance -- Supervisor.step and
gym.Env.step would collide). Run this file directly (as Spot's
`controller` field) to train a PPO stop/go policy and save it to
../../models/ppo_spot_guide_dog.zip.

Build order note: get the gait/sensors/reward visually correct first by
running with RL_MANUAL_SMOKE_TEST=1 set (drives a hardcoded GO/STOP
pattern instead of training), before trusting the PPO run.

--- New to Gymnasium / reinforcement learning (RL)? Read this first. ---

Reinforcement learning is "learn by trial and error, guided by a reward
score." An *agent* (here, Spot) repeatedly:
  1. looks at the world (an "observation" -- some numbers describing what
     it currently senses),
  2. picks an "action" (here: 0 = STOP, 1 = GO),
  3. the world changes and hands back a "reward" (a number saying how good
     that action turned out to be) plus a new observation,
  4. repeat, trying to earn as much total reward as possible over time.

Gymnasium (the `gym` package here) is a standard interface for this loop:
any environment that implements `reset()` and `step()` with the right
shapes can be plugged into off-the-shelf training algorithms like PPO
(imported below from stable_baselines3) without those algorithms needing
to know anything about Webots, quadruped robots, or wildfire drones. That
plug-and-play design is the entire point of subclassing `gym.Env` here.

A full run through reset() -> step() -> step() -> ... -> "done" is called
an "episode." This environment's episode ends either because Spot reached
the goal or collided with an obstacle ("terminated" -- the task itself is
over), or because it ran out of time without either happening ("truncated"
-- we gave up, not because the task ended naturally).
"""
import math
import os
import random
import sys
from typing import ClassVar

import gymnasium as gym
import numpy as np
from controller import Supervisor
from gymnasium import spaces

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

MAX_EPISODE_STEPS = 300
GOAL_X = 3.2
GOAL_RADIUS_M = 0.25
COLLISION_DIST_M = 0.15
SENSOR_MAX_RANGE_M = 1.5
START_TRANSLATION = [0.0, 0.0, 0.0]  # z filled in from the node's own default height
START_ROTATION = [0.0, 0.0, 1.0, 0.0]
MOVING_OBSTACLE_X = 1.8


class SpotGuideDogEnv(gym.Env):
    # gymnasium.Env expects a "metadata" dict describing supported render
    # modes; we don't render, so it's empty. ClassVar tells the type checker
    # this dict is a shared class-level constant, not meant to be mutated
    # per-instance (which is otherwise a common source of bugs with mutable
    # class attributes).
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # A Webots "Supervisor" is a special robot controller that -- unlike
        # a plain "Robot" -- is also allowed to inspect and move other nodes
        # in the simulation (here: teleporting Spot and the moving obstacle
        # back to their start positions between episodes). A plain Robot
        # can only read its own sensors and drive its own motors.
        self.sup = Supervisor()
        # getBasicTimeStep() is the simulation's millisecond step size set
        # in the world file; we step the simulation by exactly this many
        # milliseconds each time step() below is called.
        self.timestep = int(self.sup.getBasicTimeStep())

        # Look up each named device once and keep the handles around --
        # getDevice() is the standard Webots API for connecting a Python
        # controller to the actual hardware/sensors defined in the world.
        self.motors = [self.sup.getDevice(name) for name in MOTOR_NAMES]
        self.leds = [self.sup.getDevice(name) for name in LED_NAMES]
        self.sensors = [self.sup.getDevice(name) for name in SENSOR_NAMES]
        for sensor in self.sensors:
            # Sensors are off by default in Webots (to save CPU); enable()
            # turns one on and tells it how often (in ms) to refresh its
            # reading -- we want a fresh reading every simulation step.
            sensor.enable(self.timestep)

        self.spot_node = self.sup.getSelf()
        self.translation_field = self.spot_node.getField("translation")
        self.rotation_field = self.spot_node.getField("rotation")
        # Use Spot's own starting height/orientation as the reset pose,
        # so we don't have to guess the correct standing z offset.
        self._start_translation = list(self.translation_field.getSFVec3f())
        self._start_rotation = list(self.rotation_field.getSFRotation())

        # getFromDef looks up a node by the DEF name given to it in the
        # world file (here "MOVING_OBSTACLE") -- this is how a Supervisor
        # reaches into and controls a *different* object in the scene.
        self.moving_obstacle_node = self.sup.getFromDef("MOVING_OBSTACLE")
        self.moving_obstacle_translation = self.moving_obstacle_node.getField("translation")
        self.moving_obstacle_custom_data = self.moving_obstacle_node.getField("customData")
        self._moving_obstacle_start_y = self.moving_obstacle_translation.getSFVec3f()[1]

        self.gait = GaitController(self.motors, self.timestep)

        # --- Gymnasium setup: describe the shape of observations/actions ---
        # observation_space describes what _get_obs() below returns: a
        # vector of 7 numbers, each between 0.0 and 1.0. PPO (and any other
        # Gymnasium-compatible algorithm) reads this to build its neural
        # network's input layer -- it never needs to be told *what* the 7
        # numbers mean, just their shape and range.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        # action_space: Discrete(2) means "pick one of exactly 2 choices,"
        # which step() below interprets as 0 = STOP, 1 = GO.
        self.action_space = spaces.Discrete(2)  # 0 = STOP, 1 = GO

        self.step_count = 0
        self.last_action = 0

    def _get_obs(self):
        """Build the observation vector the agent sees each step: the 5
        distance-sensor readings (normalized to 0-1, where 1.0 means
        "nothing detected within range" and 0.0 means "something is
        touching the sensor"), plus the last action taken and how far
        through the episode we are. Including the last action and episode
        progress gives the policy a little bit of memory/context beyond
        the current instant, without needing a more complex recurrent
        network."""
        raw = [sensor.getValue() for sensor in self.sensors]
        norm = [float(np.clip(value / SENSOR_MAX_RANGE_M, 0.0, 1.0)) for value in raw]
        return np.array(
            norm + [float(self.last_action), min(1.0, self.step_count / MAX_EPISODE_STEPS)],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        """Gymnasium calls this at the start of every episode (including
        the very first one). It must put the simulation back into a known
        starting state and return the first observation. Returning a fresh
        (obs, info) pair -- rather than a reward -- is part of the standard
        Gymnasium reset() contract; `info` is an (here unused) dict for
        any extra debug data you want to hand back.

        gymnasium.Env.reset requires seed/options to be keyword-only (the
        leading "*" enforces that), so this signature has to match exactly
        or Gym-aware type checkers (and some Gym wrappers) will complain.
        """
        super().reset(seed=seed)
        # A seeded random generator makes the obstacle's random starting
        # position/timing reproducible when you pass the same seed -- handy
        # for debugging a specific episode, while still being random
        # (varied) across episodes during normal training.
        rng = random.Random(seed)

        self.sup.simulationResetPhysics()
        self.translation_field.setSFVec3f(self._start_translation)
        self.rotation_field.setSFRotation(self._start_rotation)
        self.gait.reset_to_stand()

        # Randomize the moving obstacle's lane position and motion timing
        # each episode. Without this randomization the agent could just
        # memorize one fixed obstacle pattern instead of learning a policy
        # that generalizes to varied obstacle behavior -- a classic RL
        # pitfall called "overfitting to the training environment."
        rand_y = self._moving_obstacle_start_y + rng.uniform(-0.4, 0.4)
        rand_phase = rng.uniform(0.0, 2 * math.pi)
        rand_period = rng.uniform(3.0, 5.0)
        self.moving_obstacle_translation.setSFVec3f([MOVING_OBSTACLE_X, rand_y, 0.1])
        self.moving_obstacle_custom_data.setSFString(f"phase={rand_phase:.3f};period={rand_period:.3f}")

        self.step_count = 0
        self.last_action = 0
        set_leds(self.leds, "off")

        # Advance the simulation exactly one tick so the freshly-reset
        # sensors/physics have a value to report before we build the first
        # observation below (right after reset, sensor readings can still
        # reflect the *previous* episode's state until a step happens).
        self.sup.step(self.timestep)
        return self._get_obs(), {}

    def step(self, action):
        """Gymnasium calls this once per timestep with the action the
        policy chose. We apply that action to the robot, advance the
        simulation by one timestep, and report back what happened. The
        5-tuple returned here -- (observation, reward, terminated,
        truncated, info) -- is the standard Gymnasium step() contract that
        every training algorithm (PPO included) expects."""
        action = int(action)
        self.last_action = action

        if action == 1:  # GO
            self.gait.step_walk()
            set_leds(self.leds, "green")
        else:  # STOP
            self.gait.step_stand()
            set_leds(self.leds, "red")
        print("GO" if action == 1 else "STOP")

        # Advance the physics simulation by one timestep. Webots returns -1
        # from step() if the simulation was told to quit (e.g. the Webots
        # window was closed), which we treat as "end the episode" below.
        sim_status = self.sup.step(self.timestep)
        self.step_count += 1

        obs = self._get_obs()
        # obs[:5] are the 5 normalized distance-sensor readings; the
        # smallest one tells us how close the nearest obstacle is.
        min_dist_norm = min(obs[:5])
        collided = min_dist_norm < (COLLISION_DIST_M / SENSOR_MAX_RANGE_M)
        x_pos = self.translation_field.getSFVec3f()[0]
        reached_goal = x_pos >= GOAL_X - GOAL_RADIUS_M

        # "terminated" = the episode ended because of something that
        # matters to the task itself (success or failure). "truncated" =
        # we cut the episode short for a reason outside the task's own
        # logic (ran out of time, or the simulator was shut down).
        # Gymnasium keeps these separate so training algorithms can treat
        # them differently -- e.g. not blaming the policy for a truncation.
        terminated = False
        truncated = False

        # --- Reward shaping: this is what actually teaches the policy ---
        # A big negative reward for colliding, a big positive reward for
        # reaching the goal, and small step-by-step nudges the rest of the
        # time (slightly rewarding GO, slightly penalizing STOP) so the
        # agent is encouraged to keep moving toward the goal rather than
        # standing still forever to avoid risk.
        if collided:
            reward = -10.0
            terminated = True
        elif reached_goal:
            reward = 10.0
            terminated = True
        else:
            reward = 0.05 if action == 1 else -0.02
            if self.step_count >= MAX_EPISODE_STEPS:
                truncated = True

        if sim_status == -1:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        # Required by the gym.Env interface, but we don't provide a custom
        # render mode (see metadata above) -- Webots' own 3D window is
        # already how you watch the robot, so there's nothing extra to do.
        pass

    def close(self):
        # Required by the gym.Env interface for cleanup (closing files,
        # windows, etc). Nothing to clean up here.
        pass


def manual_smoke_test(env):
    """Hardcoded GO/STOP pattern to visually verify gait, sensors, and
    reward/collision behavior before trusting PPO training. See plan
    verification steps 1-4.

    This bypasses the AI entirely -- it just alternates GO for 100 steps,
    then STOP for 100 steps, on repeat -- so you can watch Spot in Webots
    and confirm the gait looks right, the LEDs match the action, and the
    reward/collision numbers printed to the console make sense, *before*
    spending time training a real policy on top of a possibly-buggy
    environment. Debugging a bad reward signal is much harder once you're
    also trying to debug a trained neural network's behavior on top of it.
    """
    obs, _ = env.reset()
    for i in range(2000):
        action = 1 if (i // 100) % 2 == 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        if i % 20 == 0:
            print(f"step={i} action={action} reward={reward:.3f} obs={np.round(obs, 2)}")
        if terminated or truncated:
            print(f"episode ended at step {i}, resetting")
            obs, _ = env.reset()


def main():
    env = SpotGuideDogEnv()

    # Setting the RL_MANUAL_SMOKE_TEST=1 environment variable before
    # running this controller skips training and just runs the hardcoded
    # GO/STOP pattern above -- see the smoke-test note in the module
    # docstring at the top of this file for why you'd want that first.
    if os.environ.get("RL_MANUAL_SMOKE_TEST") == "1":
        manual_smoke_test(env)
        return

    # stable_baselines3 (SB3) is a library of ready-made RL algorithms that
    # work with any Gymnasium environment. We import it down here (rather
    # than at the top of the file) so that the manual smoke test above
    # doesn't need SB3 installed at all -- it only imports if you're
    # actually about to train.
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    # Monitor is a thin Gymnasium wrapper that records per-episode reward
    # totals and lengths for logging -- it doesn't change how the
    # environment behaves, just tracks statistics SB3 prints/plots for you.
    env = Monitor(env)
    # PPO (Proximal Policy Optimization) is the specific RL algorithm doing
    # the learning. "MlpPolicy" means the policy (the thing deciding
    # GO/STOP) is a small fully-connected neural network (a "multi-layer
    # perceptron") that takes our 7-number observation as input and
    # outputs a GO/STOP decision.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,  # print training progress to the console
        n_steps=512,  # how many steps of experience to collect before each learning update
        batch_size=64,  # how many of those steps to look at per gradient step
        learning_rate=3e-4,  # how big a step to take when updating the network's weights
        gamma=0.99,  # discount factor: how much future reward matters vs. immediate reward
        device="cpu",  # this network is small enough that a GPU wouldn't help
    )
    # os.environ.get() always returns a string (or the default you pass), so
    # the default has to be a string too, not the int 50_000 -- int() below
    # then converts whichever string we ended up with.
    total_timesteps = int(os.environ.get("RL_TOTAL_TIMESTEPS", "50000"))
    # This is where the actual training loop runs: PPO repeatedly calls
    # env.reset()/env.step() to collect experience, then updates the
    # neural network's weights based on which actions led to good rewards,
    # for a total of `total_timesteps` simulation steps across many
    # episodes.
    model.learn(total_timesteps=total_timesteps)

    save_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_spot_guide_dog.zip")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()
