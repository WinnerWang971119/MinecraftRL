"""replay — Prioritized sequence replay buffer for DRQN.

Stores fixed-length episode sequences (not individual transitions) with
proportional priority.  Supports burn-in: sampled sequences include a
burn-in prefix that is used to warm the LSTM hidden state but whose
gradients are not propagated.  Required by DRQN training.

Owner: T15 (DQN core track)
# TODO(T15): implemented by task T15
"""

# TODO(T15): implemented by task T15
