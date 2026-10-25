# C01 implementation

Current revision adds the complete bounded source/development cache, training,
evaluation and finite GPU queue described in [C01_PILOT.md](C01_PILOT.md).
Before a remote launch it remains untrained. The notes below describe the
earlier interface-only milestone and preserved loader failures, not current
pipeline coverage. Full-pipeline G0 results are separate from that old check.

Branch `proposal/c01-parent-innovation-v1`, created from shared local main `74ee248c1e6b6d0924b2b53f7cbc50c4f70c8c8b` / remote main `7d360d53e7b828e5728030dcd2be629841bf0742`. It does not contain B1 model or training files. A later proposal cannot replace these files in place.

Implemented interface: `FrozenTimeRCDHistory` verifies the supplied checkpoint SHA, freezes the original encoder and accepts only legal past history/masks for one channel at a time. `ParentInnovationHead` combines those histories using the normal directed topology and outputs a conditional distribution before receiving the query value. Nonparent values must not enter history normalization or feature generation.

Status: model interface implemented; training/cache/exposure protocol and GPU numerical checks are not yet complete. **No C01 model is trained or benchmarked at this revision.** The independently versioned B1 baseline pilot runs first. Do not infer that an implemented interface has passed the scientific tests.

The first interface check caught a checkpoint-loading issue: constructing a single-channel architecture omitted the multi checkpoint's same/different-channel bias embeddings. The loader now preserves the saved architecture and loads strictly; independent single-channel inputs do not justify discarding saved parameters. The failed snapshot and log remain on the server. This is a loader correction, not a performance finding.

Before C01 training, compare the same head with nonpretrained/frozen features and standalone B1, bind the original checkpoint and feature-cache hashes, and record all frozen-encoder computation. Root labels and normal twins are source supervision only; no target query labels, graph coefficients, true delays or clean target reference are inputs. C03 context inference is a separate proposal or explicitly named integration branch, not silently added here.

GPU interface checks now passed on H100 at model revision e1e4128: frozen encoder 35,709,568 parameters, new head 40,386; no optimizer or benchmark arrays used. This validates only the model interfaces, not a completed C01 training/evaluation pipeline. The runtime feature-count assertion is set to one after strict loading, preserving all multi checkpoint bias parameters.
