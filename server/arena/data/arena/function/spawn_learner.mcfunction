# arena:spawn_learner — pad-0 convenience wrapper around arena:spawn_learner_pad.
#
# Takes NO arguments. Resets learner_bot at anchor (0, 64, 0) to the template:
#   position (0.5, 64, 0.5), yaw -90 (facing the dummy, +X), health 20, food 20,
#   inventory exactly { iron_sword }, spawnpoint on pad 0, no active effects
#   (the heal/food restore uses instant effects that last one gametick).
#
# Multi-pad callers must NOT use this — they call the macro directly:
#     /function arena:spawn_learner_pad {x:<int>,z:<int>,learner:"<name>",nonce:<int>}
# See spawn_learner_pad.mcfunction for the argument contract and the template.

function arena:spawn_learner_pad {x:0,z:0,learner:"learner_bot",nonce:0}
