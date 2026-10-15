Hardware presets. Pick one with `--hardware rtx4090` (or `MEDVLM_HARDWARE=rtx4090`).
Layering: base -> profile -> model -> hardware -> configs/local.yaml -> --set.
The batch sizes are starting points chosen from VRAM; `medvlm bench` measures the real optimum.
