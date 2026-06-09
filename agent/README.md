# agent/

DQN core: Dueling-DRQN network, prioritized sequence replay buffer, training
loop, action enum, training config, reward config, contract config (frozen
version pins + seed utilities), and the random policy used for the M1 tracer
bullet.

**Owner workstream:** DQN core track
(Note: `contract_config.py`, `reward_config.py`, and `seeding.py` are shared
contract artifacts used by all workstreams.)
