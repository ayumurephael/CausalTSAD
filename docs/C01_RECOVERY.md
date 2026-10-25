# C01 operational recovery v1

The complete 1,901-file frozen cache and its before/after parameter hashes passed.
Four runs completed all 434 updates and 4,976,640 original scalars each. The first
rewired run failed while creating a cuBLAS handle during initial source validation,
before any optimizer update. The controller correctly stopped scheduling new jobs;
the other four training workers finished, but their development evaluation was
never scheduled. This is a technical interruption, not evidence against DAGs.

`recover_c01.py` verifies the existing cache and unchanged algorithm/config file
hashes, reuses exact copies of the four final weights, starts their evaluation,
and trains only the other eight runs. The failed zero-step attempt and all old
logs/code stay in place. Each remaining run still receives only one original
4,976,640-scalar pass. The total completed scientific optimization budget remains
59,719,680, not 16 training runs. Source/target features are not recomputed.

Shared GPUs are selected by remaining memory and an actual matmul/backward health
probe, allowing other users' processes to stay active. Memory is checked again
before each new job; the same 8GiB process cap and fixed training configuration
are retained. Device-health probes and failed-attempt overhead are separate.
The failed source-validation partial-forward count is unavailable because the
worker failed before reporting it: report unknown with a two-view source-query
upper bound of 3,936,256 scalars, never invent an exact zero cost.

The recovery has a 12h technical deadline with 3h fixed per-training limit and
6h stage limit inherited from the original configuration. It cannot extend
optimization budgets or select better checkpoints from target labels. It stops
new launches on errors, preserves completed work and touches no unrelated job.
