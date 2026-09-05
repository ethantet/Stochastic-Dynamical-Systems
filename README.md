# Natural Measure and ESN Climate Learning

**Ethan Tan Ee Teng — NTU Mathematical Sciences, May 2026**
**Supervisor: Prof. Florian Rossmannek**

---

## What this project is about

Chaotic dynamical systems like the Lorenz system are unpredictable over long horizons because trajectories diverge rapidly from any small perturbation in initial conditions. That said, their long-run statistical behaviour is actually quite stable. The distribution describing where trajectories spend their time on average is called the **natural measure** u*, and it characterises the system's climate rather than any specific trajectory.

A key result from Chan & Tong (2001) says the natural measure can be recovered as the limit of stationary distributions of the system under vanishing noise, i.e. pi_sigma -> u* as sigma -> 0.

This project has two parts:

**Part 1: Natural measure experiments**

Verifies numerically that the noise-limit result holds across five chaotic systems: Lorenz, Chen, Lu (continuous 3D ODEs), and the logistic map and tent map (discrete 1D maps). Also tests whether this holds under correlated noise (AR(1) and fractional Brownian motion), since ML model prediction errors tend to be correlated rather than independent.

**Part 2: ESN climate learning**

Replicates and extends the experiments from Louw & Ortega (2025), which show that Echo State Networks can learn the long-run statistical behaviour of a chaotic system even though point predictions break down due to chaos. The original paper only tests this on Lorenz. This project runs the same experiment on Chen, Lu, the logistic map, and the tent map.

---

## Repository structure

```
natural_measure_experiments/
  natural_measure_experiments.py   - run this: experiments 1 and 2
esn_climate_experiments/
  esn_climate_experiments.py       - run this: ESN climate learning
```

Running either script generates figures (and, for the ESN script, a
`results/` folder and validation `summary.txt`) next to it. These outputs
are gitignored and kept locally rather than committed — see the full
write-up for the figures and findings.

## Files

`natural_measure_experiments/natural_measure_experiments.py` - runs experiments 1 and 2 (noise convergence and correlated noise). Select a system and experiment from the console prompt when running.

`esn_climate_experiments/esn_climate_experiments.py` - ESN climate learning replication. Use `--systems` to pick a system, `--seed` to change the random initialisation, and `--quick` for a fast low-scale test.

## Setup

```bash
pip install -r requirements.txt
```

---

## Main findings

The noise-limit result holds across all five systems under i.i.d. noise, and stays robust under correlated noise at all tested levels. This is relevant because it suggests the natural measure remains the right long-run target even when noise has memory, which is typical of ML model errors.

For the ESN experiments, climate learning works for Lorenz, Chen, Lu, and the logistic map. The tent map fails consistently across all seeds, which comes down to its non-differentiable kink at x = 0.5 being incompatible with the smooth ESN architecture. Worth investigating further.

### Latest validation summary (ESN climate learning)

| System | V1 (init MMD² high) | V2a/b (stable MMD² low) | V2c (ratio ≫1) | V3 (error saturates) |
|---|---|---|---|---|
| Lorenz | PASS | PASS | PASS (714x) | PASS |
| Chen | PASS | PASS | PASS (493x) | PASS |
| Lu | PASS | PASS | PASS (638x) | PASS |
| Logistic map | PASS | PASS | PASS (1414x) | PASS |
| Tent map | PASS | **FAIL** | **FAIL (27x)** | PASS |

---

## References

Louw, J. & Ortega, J.P. (2025). Learning the climate of dynamical systems with state-space systems. arXiv:2512.15530.

Chan, K.S. & Tong, H. (2001). Chaos: A Statistical Perspective. Springer.

Lorenz, E.N. (1963). Deterministic nonperiodic flow. Journal of Atmospheric Sciences, 20(2), 130-141.

Leonov, G.A. & Kuznetsov, N.V. (2014). Differences and similarities in the analysis of Lorenz, Chen, and Lu systems. arXiv:1409.8649.
