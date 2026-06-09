// actions.js — Action macro execution: translates discrete action IDs to bot calls.
//
// Receives an action integer (matching the Python actions.py enum) and
// executes the corresponding Mineflayer bot calls (movement keys, look,
// bot.attack, bot.activateItem).  Handles the ACTION_REPEAT tick window:
// holds input for the configured number of ticks and then releases it.
//
// Owner: T7b (Environment/bridge track)
// TODO(T7b): implemented by task T7b
