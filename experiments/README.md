# Experiments (not part of the demo)

Approaches we tried and did not deploy. They are kept because they explain the choices in the main pipeline,
and some of them hold negative results.

| Folder | What | Outcome |
|---|---|---|
| `early_v1/` | Tim's first early-detection SNN (fires 300–500 ms before the swing peak), external data (DailyDVS, HMDB via v2e), viewer | precise on spliced clips, too many false alarms on whole recordings |
| `early_ctx_eval/` | Noah's evaluation of early detection on whole recordings, stateful, with a chance baseline | gate failed (see `REPORT.md`); led to training on whole recordings (`training/early_ctx`) |
| `robust_release/` | throw vs feint classification (did the ball leave the hand?) | near chance across sessions: the yarn ball is barely visible to the sensor |
| `deploy_noah/` | Noah's earlier two-stage reflex (wind-up arms, throw dashes) on the real Go2 from a laptop | superseded by `dodge/`, which runs on the Kria itself |
| `tiny_snn/` | Laurenz's first tiny all-convolutional SNN and its Kria export | first end-to-end export check |

The low-level dodge controller and its simulation stay in [`sim/`](../sim/README.md): the deployed dodge uses the
Go2 sport API instead, but the sim results justify its speed and duration.
