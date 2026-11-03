# 1-eye calibrations: impacted participants and blocks

Generated 2026-11-03 by `scripts/preprocessing/qc/summarize_eye_calibration_impact.py` (groups: adults, kids). Infants not yet checked.

Before 2026-10-30 the experiment accepted a calibration that passed on one eye only, without logging which. Which eye passed was reconstructed from the per-eye validation accuracies (`METHODS_DECISIONS.md`, 2026-10-30). I2MC uses both eyes whenever their averaged error is ≤ 2.0°, and falls back to the experiment-accepted eye only when it isn't (`METHODS_DECISIONS.md`, 2026-11-03).

## Overview

| Group | Participants | Blocks | Accepted on 1 eye | Analyzed on 1 eye | Participants analyzed on 1 eye |
|---|---|---|---|---|---|
| adults | 53 | 212 | 13 (left 3, right 10) | 3 | 2 |
| kids | 22 | 88 | 1 (left 1, right 0) | 1 | 1 |

## Effect on the I2MC calibration check (≤ 2.0°)

- Blocks newly **included** (both-eyes average > 2.0°, accepted eye ≤ 2.0°): 4 -- these are the only blocks whose fixations change.
- Blocks newly **excluded**: 0

## Blocks accepted on 1 eye

| Group | Participant | Block | Block ID | Accepted eye | Both-eyes error (°) | Accepted-eye error (°) | Analyzed | I2MC check |
|---|---|---|---|---|---|---|---|---|
| adults | MW003 | 0 | slow | right | 1.19 | 1.11 | both | included |
| adults | MW005 | 2 | sesame | right | 2.18 | 2.33 | — | excluded |
| adults | MW015 | 0 | slow | right | 1.23 | 1.07 | both | included |
| adults | MW021 | 1 | slow | right | 2.86 | 1.68 | right | included (was excluded) |
| adults | MW022 | 0 | slow | left | 1.35 | 1.21 | both | included |
| adults | MW024 | 0 | sesame | right | 1.12 | 1.04 | both | included |
| adults | MW027 | 0 | slow | right | 1.62 | 1.09 | both | included |
| adults | MW027 | 1 | sesame | left | 2.71 | 2.34 | — | excluded |
| adults | MW028 | 0 | slow | left | 1.27 | 1.17 | both | included |
| adults | MW031 | 0 | frank | right | 2.15 | 1.53 | right | included (was excluded) |
| adults | MW031 | 1 | sesame | right | 3.02 | 1.89 | right | included (was excluded) |
| adults | MW031 | 2 | slow | right | 2.87 | 2.08 | — | excluded |
| adults | MW048 | 0 | sesame | right | 1.12 | 1.16 | both | included |
| kids | MWK020 | 2 | frank | left | 2.30 | 1.62 | left | included (was excluded) |

## Other notes

- kids/MWK004: lb_forced_validation_3_2: mid-block recalibration accepted on right eye (not used by I2MC)
