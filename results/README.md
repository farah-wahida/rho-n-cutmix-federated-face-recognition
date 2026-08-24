# Results

The paper directory contains the machine-readable values reported in Tables IV–X of the published article. These files are reference outputs, not newly recomputed estimates.

| File | Paper content |
|---|---|
| table_iv_accuracy.csv | Recognition accuracy for all 40 runs |
| table_v_mia_max_softmax.csv | Max-softmax membership ROC-AUC |
| table_vi_mia_scores.csv | Max-softmax, loss, and entropy attack scores |
| table_vii_inversion.csv | Adaptive inversion and identity metrics |
| table_viii_nll.csv | NLL before and after temperature scaling |
| table_ix_verification.csv | EER and TAR@FAR for Defense A |
| table_x_latency.csv | Workstation preprocessing and inference latency |

Fresh executions write run artifacts below outputs/, which is intentionally ignored because checkpoints and reconstructed images can be large. Use scripts/reproduce_tables.py to aggregate fresh JSON outputs and scripts/validate_release.py to verify that the release covers all 40 configurations and all seven reference tables.

Dataset files and checkpoints are not redistributed because of licensing, privacy, and file-size constraints.
