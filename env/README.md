# env/

Python Gym-style environment side of the Minecraft PvP stack. Owns the
observation spec (the frozen float-vector contract), the Gym env wrapper
(`mc_pvp_env.py`), the reward function, and the perception filter (FOV + LoS +
memory gating).

**Per-file ownership:**
- `mc_pvp_env.py` — Environment/bridge track
- `observation_spec.py` — contract (PerceptionFilter/contract)
- `perception_filter.py` — PerceptionFilter track
- `reward.py` — Reward/opponent track
