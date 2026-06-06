## Summary

Adds the warm CPU image server helper plus a small serial benchmark utility for comparing useful image sizes on the same resident model process.

This also records the current strongest measured result from the `224-320` useful grid search on the unpacked CPU path:

- `224x224`, `4-step`: about `44.4s`
- `224x256`, `4-step`: about `50.0s`
- `256x224`, `4-step`: about `50.1s`
- `320x320`, `4-step`: about `85.7s`

## What is included

- `scripts/cpu_image_server.py`
  - keeps the VAE and unpacked transformer resident
  - exposes `/healthz` plus a loopback `/generate` endpoint
  - supports the same main runtime knobs used during CPU bring-up
- `scripts/benchmark_cpu_server_grid.py`
  - runs serial size probes against the warm server
  - saves the output PNGs
  - reports wall time plus simple image stats (`mean`, `std`)

## Why this is useful

The cold-start cost on CPU is large enough that one-shot command timings hide the real shape tradeoffs.

The warm server plus serial benchmark helper make it possible to compare:

- shape and aspect-ratio effects
- runtime at fixed step count
- output sanity across the same resident process

That was enough to map the useful `224-320`, 32-pixel grid and identify `224x224` as the fastest verified useful `4-step` route on this CPU path.

## Notes

- This is still CPU benchmarking and utility code, not a claim that the CPU path is broadly optimized.
- The benchmark is intentionally serial so results do not get contaminated by overlapping renders.
