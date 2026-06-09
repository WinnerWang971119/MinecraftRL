"""Root conftest.py — ensures the repo root is on sys.path for pytest collection.

This file intentionally left minimal.  The presence of this file at the repo
root causes pytest to add the repo root to sys.path, making
``from env.observation_spec import ...`` and ``from agent.actions import ...``
work in tests even without an editable install (``pip install -e .``).
"""
