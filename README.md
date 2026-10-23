# sfdojo: Deterministic Frame Data & Game Theory Engine

A competitive data analytics and game theory framework for Street Fighter 6.

## Overview

This package implements mathematical modeling for fighting game decision trees, focusing on:
1. **Deterministic Meaty Timing**: Exact active frame connection math, bonus frame advantages, and automated frame kill sequence derivation.
2. **Game Theoretic Payoff Modeling**: 2-player zero-sum payoff matrices incorporating damage, corner proximity, meter state, and burnout.
3. **Nash Equilibrium Solving**: Linear programming (via `scipy.optimize.linprog` with HiGHS) to find optimal mixed strategies and game value.
4. **Exploitative Best Response**: Counter-strategy calculation against biased human tendencies (e.g., delay-tech spammers, wakeup mashers).

---

## Directory Structure

```
sfdojo/
├── __init__.py           # Package exports
├── frame_data.py         # Move data models, meaty timing & frame kill search
├── payoff_matrix.py      # SF6 interaction rules & situational modifiers
├── nash_solver.py        # LP minimax equilibrium & best response solver
├── oki_engine.py         # Unified engine with character presets (Ken, Luke)
├── demo_oki_analysis.py  # Executable CLI simulation runner
├── test_oki.py           # Unit tests
└── README.md             # This document
```

---

## Quickstart

Run the unit tests:
```bash
PYTHONPATH=/home/muse/Documents/code python3 -m unittest sfdojo/test_oki.py
```

Run the complete simulation:
```bash
PYTHONPATH=/home/muse/Documents/code python3 sfdojo/demo_oki_analysis.py
```

---

## Roadmap

- **Phase 1 (Complete)**: Static Mathematical & Game Theory Oki Model.
- **Phase 2 (Next)**: Replay Telemetry Pipeline using REFramework Lua hooks to extract match frames.
- **Phase 3**: Empirical Behavioral Modeling linking match telemetry to the Nash solver.
- **Phase 4**: Neutral & Footsies Analytics (whiff probability maps and walk sequence mining).
