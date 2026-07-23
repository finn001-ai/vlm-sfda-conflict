# Matched PLMatch 25% Proxy Control

## Contract

- Method: original PLMatch path, with no change to its pseudo-label rule,
  losses, optimizer, CLIP update, or model architecture.
- Source checkpoint environment: the same local VisDA-C source directory used
  by the DCCL proxy experiments.
- Adaptation set: deterministic class-proportional 25% proxy,
  `ratio=0.25`, `seed=2020`, 13,847 images.
- Evaluation set: complete VisDA-C validation list, 55,388 images.
- Budget: 4 cycles and 16 recorded checkpoints.
- Selection: final checkpoint. Oracle peak is reported separately and is not
  used to decide the control.

## Result

The PLMatch final and oracle peak coincide at Cycle 4, Iteration 868:

```text
PLMatch final       = 87.93
DCCL P1 final       = 87.83
delta               = +0.10 pp for PLMatch
predeclared margin  = +/-0.20 pp
decision            = matched_within_margin
```

Therefore, the matched control does not establish a significant macro-accuracy
advantage for either method. Numerically, original PLMatch is 0.10 pp above
DCCL P1, while the completed DCCL combined run (`87.96`) is 0.03 pp above
PLMatch. Both differences are too small to support a superiority claim from a
single seed.

## Class-level comparison against DCCL P1

| Class | PLMatch | DCCL P1 | PLMatch - DCCL |
|---|---:|---:|---:|
| aeroplane | 97.45 | 97.56 | -0.11 |
| bicycle | 83.63 | 86.01 | -2.38 |
| bus | 84.82 | 85.61 | -0.79 |
| car | 80.42 | 75.36 | +5.06 |
| horse | 96.87 | 96.27 | +0.60 |
| knife | 93.98 | 95.81 | -1.83 |
| motorcycle | 93.91 | 93.50 | +0.41 |
| person | 84.18 | 80.68 | +3.50 |
| plant | 92.46 | 91.91 | +0.55 |
| skateboard | 93.56 | 94.56 | -1.00 |
| train | 91.19 | 91.34 | -0.15 |
| truck | 62.74 | 65.34 | -2.60 |
| car/person/truck mean | 75.78 | 73.79 | +1.99 |
| other-nine mean | 91.99 | 92.51 | -0.52 |
| macro | 87.93 | 87.83 | +0.10 |

PLMatch's small macro advantage over P1 is not uniform. It comes mainly from
`car` and `person`, while `truck`, `bicycle`, `knife`, and `skateboard`
decrease. DCCL P1 instead retains a higher truck score and a 0.52 pp advantage
over the other nine classes on average. This is another car/person/truck
redistribution rather than a broad improvement.

## Consequence

The control closes the uncertainty about the local PLMatch base: current DCCL
does not provide a meaningful single-seed macro gain over its matched original
PLMatch baseline. Conversely, the control also does not prove that PLMatch is
meaningfully better, because `+0.10 pp` is inside the predeclared tie margin.

Do not present `87.93 > 87.83` as a demonstrated method advantage. Before a
paper claim, repeat the matched comparison with adaptation seeds 2021 and 2022
or provide a stronger mechanism-level change that improves macro accuracy
without merely exchanging car/person against truck and the remaining classes.
