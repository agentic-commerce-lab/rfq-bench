# Archived runs

`results/` is git-ignored scratch space. Runs worth keeping (published results, baselines
to compare future runs against) are copied here, one folder per run:
`data/runs/<YYYY-MM-DD>_<name>/`.

Each folder holds the trace files unchanged plus a `MANIFEST.md` with:

- the sha256 of every trace file (verify with `shasum -a 256 <file>`),
- the models, episode and error counts, and cost,
- the exact condition (endpoint, decoding, suite, matrix) and the config that produced it,
- the code version (newer traces also carry it in `Trace.provenance`),
- the commands to rebuild the reports and to repeat and compare the run.

Never edit an archived trace. To add a run, copy its `.jsonl` here and write the manifest.
