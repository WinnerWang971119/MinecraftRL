# arena:reset — pad-0 convenience wrapper around the arena:reset_pad macro.
#
# Takes NO arguments. It is the single-arena entry point that predates the pad
# topology and is kept byte-compatible on purpose (plan AC11): calling
#     /function arena:reset
# resets learner_bot / dummy_bot at anchor (0, 64, 0) exactly as before.
#
# Multi-pad callers must NOT use this — they call the macro directly:
#     /function arena:reset_pad {x:<int>,z:<int>,learner:"<name>",dummy:"<name>"}
# See reset_pad.mcfunction for the full argument contract.
#
# This does NOT re-run arena:setup / arena:setup_pad (geometry and gamerules are
# once-per-boot). After issuing this the bridge still runs its READ-BACK GATE
# before replying reset_ack — these commands are async/unacked.

function arena:reset_pad {x:0,z:0,learner:"learner_bot",dummy:"dummy_bot"}
