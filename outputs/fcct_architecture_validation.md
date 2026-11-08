# FCCT/IRI architecture validation (version 0.2)

**This is an implementation validation report, not a numerical reproduction of the paper.**

- 30 FCCT tests passed: 29 full-suite tests plus one added ablation-report regression test.
- Real tiny Transformers decoder checks cover the complete tracing loop, clean activation restoration, cached decoding and attention extraction.
- Equation 2 now retains signed nonzero denominators without clipping.
- Figure 4 analysis and 12 named Table 6 controls have executable workflows; original author inputs/normalization remain unverified.
- All 1,000 prepared images, 1,371 tracing questions and 9,000 POPE questions are preserved; model and dataset checksums passed.
- Version-0.1 traces/profiles must be regenerated. New source/config checksums prevent incompatible resumes.
- No full trained-checkpoint scientific experiment was completed for version 0.2.

See [the architecture coverage audit](../docs/fcct_architecture_audit.md) for missing models, benchmarks, baselines and original artifacts.
