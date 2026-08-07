# arena:spawn_dummy — pad-0 convenience wrapper around arena:spawn_dummy_pad.
#
# Takes NO arguments. Resets dummy_bot at anchor (0, 64, 0) to the template:
#   position (3.5, 64, 0.5), yaw -90 (facing the learner), health 20, food 20,
#   empty inventory, knockback_resistance 1.0, movement_speed 0.0, spawnpoint
#   on pad 0.
#
# Multi-pad callers must NOT use this — they call the macro directly:
#     /function arena:spawn_dummy_pad {x:<int>,z:<int>,dummy:"<name>",nonce:<int>}
# See spawn_dummy_pad.mcfunction for the argument contract and the template.
#
# Attribute ids on this stack REQUIRE the `generic.` infix
# (`minecraft:generic.knockback_resistance`). The flattening that removed it
# landed in 1.21.2; Paper is pinned to 1.21.1. Verified live and against this
# repo's boot logs — see the header of spawn_dummy_pad.mcfunction before
# changing it.

function arena:spawn_dummy_pad {x:0,z:0,dummy:"dummy_bot",nonce:0}
