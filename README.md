# Spot Guide Dog

A Webots simulation where a Boston Dynamics-style **Spot** quadruped learns a
simple GO/STOP "guide dog" policy with reinforcement learning: walk forward
toward a goal, stop when an obstacle (static or moving) is too close.

Spot has no camera in this project — its only "eyes" are 5 distance sensors
mounted in a fan around its body. A PPO policy (via
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) is trained to
turn those 5 numbers into a binary decision each simulation tick.

## Project layout

```
controllers/
  spot_guide_dog_lib/    Shared code: motor/LED/sensor names, gait controller
  spot_guide_dog_train/  Gymnasium env + PPO training loop (Supervisor controller)
  spot_guide_dog_demo/   Loads the trained model and runs it live (inference only)
  moving_obstacle/       Controller for the sinusoidally-moving obstacle box
worlds/
  spot_guide_dog.wbt     Main world: Spot + goal marker + static/moving obstacles
  spot.wbt               Minimal world used for tuning the stand/walk gait
```

## Requirements

- [Webots](https://cyberbotics.com/) R2025a
- Python 3.10+ (matching the Python version Webots is configured to use)
- Python packages in `requirements.txt`

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Point Webots (or your editor, see `.vscode/settings.json`) at this `venv`'s
interpreter, and make sure Webots' own `controller` Python module is
importable (Webots ships this under
`<Webots install dir>/lib/controller/python`).

## Usage

1. Open `worlds/spot_guide_dog.wbt` in Webots.
2. **Train**: Spot's controller is set to `spot_guide_dog_train`. Run the
   simulation to train a PPO policy; the trained model is saved to
   `models/ppo_spot_guide_dog.zip`.
   - Set `RL_MANUAL_SMOKE_TEST=1` to sanity-check the gait, sensors, and
     reward signal with a hardcoded GO/STOP pattern instead of training.
   - Set `RL_TOTAL_TIMESTEPS` to change training length (default `50000`).
3. **Demo**: switch Spot's controller to `spot_guide_dog_demo` to load the
   saved model and watch it run continuously (no episodes/resets).

Trained models are not checked into this repo (see `.gitignore`) — train
locally to generate `models/ppo_spot_guide_dog.zip`.
