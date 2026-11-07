# Historical benchmark artifacts

This directory preserves frozen manifests, experiment results, data/model cards,
and checksums from earlier studies. It is not a separate source-code distribution.

Use the repository entry point for current workflows:

```powershell
python pathway.py validate --help
python pathway.py split --help
python pathway.py report --help
python pathway.py verify benchmark/outputs/runs/2026-10-10_qwen25vl7b_confirmatory1000
```

Canonical implementations are in `../scripts/`, `../scoring/`, `../schemas/`,
`../configs/`, and `../annotations/`. Historical outputs retain their original
scorer labels and metrics; the cleanup did not rewrite frozen evidence.
See the [project README](../README.md), [protocol](../docs/benchmark_protocol.md),
and [cleanup notes](../docs/cleanup.md) for current behavior and limitations.
