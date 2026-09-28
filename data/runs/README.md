# Archived runs

A local archive for runs worth keeping (published results, baselines to compare future
runs against), one folder per run: `data/runs/<YYYY-MM-DD>_<name>/`. Unlike `results/`,
which is scratch space, nothing here is ever overwritten.

The run folders are **git-ignored**: traces contain full model replies and can be large,
so they are not committed. Only this README is tracked. To share a run, attach its folder
(zipped, with the manifest) to a GitHub release or store it elsewhere, and cite the
manifest's checksums.

Each folder holds the trace files unchanged plus a `MANIFEST.md` with:

- the sha256 of every trace file (verify with `shasum -a 256 <file>`),
- the models, episode and error counts, and cost,
- the exact condition (endpoint, decoding, suite, matrix) and the config that produced it,
- the code version (newer traces also carry it in `Trace.provenance`),
- the commands to rebuild the reports and to repeat and compare the run.

Never edit an archived trace. To add a run, copy its `.jsonl` files here and write the
manifest.
