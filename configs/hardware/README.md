Hardware presets. Pick one with `--hardware rtx4090` (or `MEDVLM_HARDWARE=rtx4090`).
Layering: base -> profile -> model -> hardware -> configs/local.yaml -> --set.
All presets use batch 1: batching changes bf16 gold log-probs (see the preset files). `medvlm bench`
measures faster sizes for throwaway runs only.
