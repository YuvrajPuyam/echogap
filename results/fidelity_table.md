# SplatAD rendered-vs-real lidar fidelity, 10 PandaSet scenes (30k iters each, A30, fixed get_outputs_for_lidar)

| PandaSet seq | median range err (cm) | ray-drop acc (%) | intensity RMSE | scans | step |
|---|---|---|---|---|---|
| 001 | 1.31 | 96.7 | 0.083 | 10 | 30000 |
| 011 | 1.46 | 96.6 | 0.064 | 10 | 30000 |
| 016 | 0.92 | 97.0 | 0.057 | 10 | 30000 |
| 028 | 0.24 | 98.4 | 0.055 | 10 | 30000 |
| 053 | 1.15 | 97.1 | 0.065 | 10 | 30000 |
| 063 | 2.11 | 95.9 | 0.062 | 10 | 30000 |
| 084 | 2.16 | 95.4 | 0.057 | 10 | 30000 |
| 106 | 1.13 | 96.3 | 0.060 | 10 | 30000 |
| 123 | 0.92 | 97.0 | 0.057 | 10 | 30000 |
| 158 | 1.44 | 96.4 | 0.050 | 10 | 30000 |
| **median / mean** | **1.23** | **96.7** | **0.061** | 100 | |

Paper (SplatAD, fully trained): median depth error 1-2 cm, ray-drop acc 92.6-96.7%, intensity RMSE 0.036-0.059.
