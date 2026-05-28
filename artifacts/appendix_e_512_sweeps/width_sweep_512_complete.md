# Width Sweep 512-Token Budget - Complete Table

All rows are at max_new_tokens=512 and are paired against the 512-token clean (results_no_swap.jsonl, accuracy 0.524). The 256-token canonical main row (0.500 clean / 0.468 swap / 0.604 answer-changed / 0.862 stable-wrong-different) is reported separately and is not substituted by any 512-budget row.

| width | layers | clean_acc | condition_acc | accuracy_delta | answer_changed | sw_diff | repaired | broken | net | profile |
|------:|:-------|----------:|--------------:|---------------:|---------------:|--------:|---------:|-------:|----:|:--------|
| 2 | 8..9 | 0.524 | 0.54 | 0.016 [-0.032, 0.06] | 0.364 [0.304, 0.424] (91/250) | 0.5555555555555556 [0.4563106796116505, 0.652542372881356] (55/99) | 20 | 16 | 4 | needs_manual_review |
| 4 | 8..11 | 0.524 | 0.492 | -0.032 [-0.088, 0.024] | 0.472 [0.408, 0.536] (118/250) | 0.6938775510204082 [0.6, 0.78125] (68/98) | 21 | 29 | -8 | break_heavy |
| 6 | 8..13 | 0.524 | 0.504 | -0.02 [-0.076, 0.036] | 0.464 [0.4, 0.524] (116/250) | 0.6631578947368421 [0.5681818181818182, 0.7570093457943925] (63/95) | 24 | 29 | -5 | break_heavy |
| 8 | 8..15 | 0.524 | 0.516 | -0.008 [-0.068, 0.052] | 0.496 [0.432, 0.556] (124/250) | 0.7252747252747253 [0.6333333333333333, 0.8152173913043478] (66/91) | 28 | 30 | -2 | high_churn_low_net_change |
| 12 | 8..19 | 0.524 | 0.524 | 0.0 [-0.064, 0.06] | 0.54 [0.476, 0.6] (135/250) | 0.8160919540229885 [0.7323943661971831, 0.8941176470588236] (71/87) | 32 | 32 | 0 | high_churn_low_net_change |

## Notes

- Released metrics are under `artifacts/appendix_e_512_sweeps/`.
- Optional raw JSONL, when included, lives under `data/appendix_e_512_raw/`.
- The 256-token canonical row is tracked separately in `transition_accounting_all_conditions.csv`.
