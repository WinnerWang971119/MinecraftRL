# arena:reset — per-episode convenience: reset BOTH bots in one call.
#
# The bridge's reset RPC (bridge/bot.js handleReset / T7a) can either issue the
# individual /tp + /effect + regear commands it already has, OR call this single
# opped function for the same effect:
#     /function arena:reset
#
# It does NOT re-run arena:setup (floor/gamerules are once-per-boot). It only
# re-places and re-gears the two bots. After issuing this, the bridge still runs
# its READ-BACK GATE (poll until learner matches the template or timeout) before
# replying reset_ack — the commands here are async/unacked like any others.

function arena:spawn_learner
function arena:spawn_dummy
