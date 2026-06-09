# Day-1 Version Compatibility Check (Tv)

**Status:** TODO — run before writing `agent/contract_config.py`

This check must confirm that `mineflayer`, `minecraft-data`, the Paper build,
`mineflayer-pvp`, and `mineflayer-pathfinder` all support the target Minecraft
version (1.21.1 per plan decision) before the version is pinned.  If any
component lags, move the pin to the highest version all four support.

**Owner:** Tv (Environment/bridge track)

## Checklist

- [ ] Paper 1.21.1 build available and stable
- [ ] `minecraft-data` has 1.21.1 data (check `minecraft-data/lib/index.js`)
- [ ] `mineflayer` supports 1.21.1 (check `mineflayer` changelog / issues)
- [ ] `mineflayer-pathfinder` supports 1.21.1
- [ ] `mineflayer-pvp` supports 1.21.1 (attack-cooldown combat required)
- [ ] 1.9+ attack-cooldown observable via bridge — confirmed by manual test

## Result

| Component | Confirmed version | Notes |
|-----------|------------------|-------|
| Minecraft | TBD | |
| Paper | TBD | |
| mineflayer | TBD | |
| minecraft-data | TBD | |
| mineflayer-pathfinder | TBD | |
| mineflayer-pvp | TBD | |
| Node | TBD | |
| Python | TBD | |
