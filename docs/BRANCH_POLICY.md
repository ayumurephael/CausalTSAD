# Independent proposal history

1. Record the shared main commit before creating a proposal/baseline branch.
2. Keep each proposal's implementation, configuration and run protocol on its own branch. Common bug fixes may be cherry-picked, with the original commit recorded.
3. Never rename C01 code to C02 or overwrite a failed proposal with a new idea. Mark the old run's outcome, retaining its code and artifacts.
4. New hypotheses based on earlier results start from the shared base or a new documented common revision. An intentional C01+C03 combination uses an integration branch and lists both source revisions.
5. Commit before deployment. Bind remote code files, source plan, input arrays, starting weights and seeds with SHA256. Changes after launch require a new revision and run directory.
6. Retain fixed-budget final checkpoints, learning curves, negative results and aggregate evaluation. Large files remain on the authorized server and are referenced by relative path and hash, not uploaded to Git.
7. No force push, history rewriting, target-test tuning, credential files or private server documentation.

Initial order: a small standalone known-DAG baseline with graph controls, followed by C01 on an independent branch. Baseline findings can inform a new run protocol; they do not mutate the already deployed implementation.
