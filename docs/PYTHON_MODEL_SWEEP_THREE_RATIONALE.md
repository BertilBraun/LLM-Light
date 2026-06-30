# Python Model Sweep Three Rationale

## Goal

Sweep three is a focused follow-up to the first two Python model sweeps. It tests the highest-signal findings without expanding into another broad architecture search.

The main questions are:

- Does the vocab-2000 improvement transfer from dense models to modern MoE?
- Does stronger router balancing improve modern MoE without hurting task performance?
- Does top-k=2 help when active parameters are matched against top-k=1?
- Can a deeper modern MoE near 10M active parameters beat the previous dense 9.6M-active winner?
- Is the 10M result due to MoE specifically, or does a modern dense vocab-2000 control perform similarly?

## What Worked

Modern architecture helped substantially. The modern MoE family was the strongest small-model direction, with `python_modern_moe_vocab4000_aux010` reaching `61.61% +/- 0.15 pp` aggregate final pass rate. That was far ahead of the classic MoE runs and ahead of dense models in the same active-parameter range.

Vocab reduction to 2000 looked useful. The dense vocab-2000 parameter-matched run reached `49.56% +/- 0.44 pp`, while the vocab-4000 dense baseline was `37.40% +/- 1.24 pp`. That is large enough to justify applying vocab 2000 to the next MoE runs.

Larger FFN capacity helped dense models. The modern dense FFN-large run was materially better than the baseline FFN-medium shape, although it still trailed modern MoE.

## What Did Not Work

QK normalization did not look promising in this setting. The qknorm dense variants were far below the non-qknorm dense baseline, even when paired with a higher learning rate or cosine schedule. Sweep three therefore keeps `query_key_normalization=false`.

Learning-rate schedules did not earn another slot. The classic MoE warmup/decay runs underperformed the fixed learning-rate baseline, and the qknorm cosine run was also weak. Sweep three keeps the fixed `0.001` learning rate.

Fill-in-the-middle did not justify another run. The FIM classic MoE run improved over some weak baselines but remained far behind modern MoE and did not address the main router/architecture question.

Wide shallow networks did not help. The small wide dense and MoE runs were among the weakest early results, so sweep three does not spend budget on wider shallow variants.

Classic MoE remains a lower-priority branch. It showed severe router dominance and much worse pass rate than modern MoE, so sweep three only uses modern MoE.

## Sweep Three Runs

| Experiment | Purpose | Active params | Total params |
| --- | --- | ---: | ---: |
| `python_modern_moe_vocab2000_aux010` | Small modern MoE vocab-2000 baseline with the previous auxiliary weight | 986,608 | 2,663,920 |
| `python_modern_moe_vocab2000_aux020` | Same active budget, stronger router auxiliary loss | 986,608 | 2,663,920 |
| `python_modern_moe_vocab2000_topk2_aux010` | Top-k=2 router test, resized to match the small top-k=1 active budget | 1,001,680 | 1,738,960 |
| `python_modern_moe_vocab2000_topk2_aux020` | Top-k=2 plus stronger auxiliary loss, same active budget as the top-k=2 aux-0.1 run | 1,001,680 | 1,738,960 |
| `python_modern_moe_deep10_vocab2000_aux020` | Main deep MoE challenger near 10M active parameters | 9,784,576 | 29,691,136 |
| `python_modern_dense_active10m_vocab2000` | Modern dense vocab-2000 control near the same active budget | 9,737,280 | 9,737,280 |

## Design Notes

The small top-k=2 models are not the same architecture as the top-k=1 models. They are deliberately smaller (`dimension=80`, `layers=4`, `expert_ffn=384`) so that using two active experts lands near the same active-parameter budget as the top-k=1 small models.

The big MoE uses 10 layers rather than 12 to test depth without making the architecture too narrow or too slow. It keeps `router_top_k=1` so the active-parameter comparison against the dense control is clean.

The big dense control is included because the 10M MoE comparison would otherwise be ambiguous. Both big runs use vocab 2000 and are matched within roughly 50k active parameters.
