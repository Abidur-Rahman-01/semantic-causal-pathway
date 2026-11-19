"""R-FCCT: reliability extensions to FCCT/IRI (arXiv:2511.05923v3).

The original FCCT/IRI reconstruction in ``src/fcct`` is left unchanged and is
used as the reference implementation. This package adds, without modifying the
paper's definitions:

* contrastive decision-margin tracing over present *and* absent objects,
* drop-filtered, image-balanced ratio-of-means causal scores,
* bootstrap stability selection with reachability-aware source/target plans,
* held-out calibration of injection strength (no test-set tuning),
* protocol-matched evaluation (greedy and the VCD sampling protocol),
  prior-shift threshold controls, and paired cluster statistics.
"""

EXTENSION_VERSION = "0.3.0"
