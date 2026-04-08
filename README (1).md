---
title: Healthcare RL Env
sdk: docker
colorFrom: blue
colorTo: green
pinned: false
---

# AI Healthcare Assistant — Reinforcement Learning Environment

A simulated episodic RL environment for medical triage built on a canonical
**state–action–reward (SAR)** framework. An agent observes structured patient
states and selects from 8 discrete clinical guidance actions. Rewards are
normalised to `[0, 1]` and shaped to balance safety and clinical appropriateness.

The app runs inference automatically on startup and displays results on a
web UI served at port 7860.

---

## Project Structure

```
app.py                 # Flask server — HF Spaces entry point (port 7860)
healthcare_rl_env.py   # RL environment, Pydantic models, agent, reward fn, inference
openenv.yaml           # OpenEnv-compatible metadata
requirements.txt       # pydantic, flask
Dockerfile             # python:3.10-slim, exposes 7860, CMD app.py
README.md              # This file
```

---

## Three Tasks with Increasing Difficulty

The environment exposes three named tasks via `reset(difficulty=...)`. Each
task is a separate, fully deterministic episode with a fixed initial state and
a clear correct action. All three tasks share the same action space and reward
function — only the clinical scenario changes.

| Task | Severity | Patient Profile | Correct Action | Max Reward |
|---|---|---|---|---|
| `easy` | mild | Single symptom, healthy adult | `home_care` | 1.0 |
| `medium` | moderate | Multi-symptom, no chronic condition | `gp_visit` / `telehealth` | 0.875 |
| `hard` | severe | Elderly patient with chronic condition | `emergency` | 1.0 |

Each difficulty level acts as an independent graded task. Rewards are
deterministic and continuous in `[0, 1]` — partial credit for suboptimal-but-safe
actions, close to 0 for dangerous decisions.

---

## Reward Design

- **Range:** `[0.0, 1.0]` normalised via `(raw + 2) / 4` from raw range `[-2, 2]`
- `1.0` — clinically correct and optimal action
- `0.5` — partially correct or acceptable but suboptimal
- `0.0` — dangerous or inappropriate (unsafe under- or over-reaction)

**Anti-reward-hacking:** penalties for both under- and over-reaction prevent
the agent from exploiting a one-sided signal.

---

## Pydantic Models

All data structures are Pydantic `BaseModel` classes — full type validation,
`.model_dump()`, and JSON schema generation included.

| Model | Fields |
|---|---|
| `State` | `symptoms`, `severity`, `duration_days`, `age_group`, `has_chronic_condition`, `step` |
| `Action` | `action_id`, `label`, `description` |
| `StepResult` | `next_state`, `reward`, `done`, `info` |

---

## Deployment

### HuggingFace Spaces (Docker SDK)

This Space uses `sdk: docker`. The container:
1. Installs `pydantic` and `flask`
2. Starts `app.py` which launches a Flask server on port **7860**
3. Runs inference in a background thread on startup
4. Serves results as a web UI at `/`, JSON at `/api/results`, health at `/health`

### Local Run

```bash
pip install pydantic flask
python app.py          # web UI at http://localhost:7860

# or run inference directly in the terminal:
python healthcare_rl_env.py
```

---

## OpenEnv Interface

Compatible with any Gym-style training loop:

```python
env   = HealthcareEnv(max_steps=5, seed=42)
state = env.reset(difficulty="medium")

done = False
while not done:
    action_id = agent.select_action(state)
    result    = env.step(action_id)
    done      = result.done
    state     = result.next_state or state
```

---

## Extending the Agent

```python
class MyAgent:
    def select_action(self, state: State) -> int:
        ...  # Q-table, DQN, PPO — any policy returning int in [0, 7]
```
