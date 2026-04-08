---
title: Healthcare RL Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# AI Healthcare Assistant — Reinforcement Learning Environment

A simulated episodic RL environment for medical triage built on a canonical
**state–action–reward (SAR)** framework. An LLM agent observes structured
patient states and selects from 8 discrete clinical guidance actions. Rewards
are normalised to `[0, 1]`.

---

## Project Structure

```
inference.py           # MANDATORY — hackathon grader entry point
app.py                 # Flask server — HF Spaces web UI (port 7860)
healthcare_rl_env.py   # RL environment, Pydantic models, reward function
openenv.yaml           # OpenEnv-compatible metadata
requirements.txt       # pydantic, flask, openai
Dockerfile             # python:3.10-slim, exposes 7860, CMD app.py
README.md              # This file
```

---

## Inference Script

`inference.py` is the mandatory evaluation entry point. It uses the OpenAI
client to query an LLM, which selects triage actions based on patient state.

### Required Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace / API key | *(required)* |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `TASK_NAME` | Task difficulty: `easy`, `medium`, `hard` | `easy` |

### Stdout Format

```
[START] task=easy env=healthcare-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=home_care reward=0.88 done=false error=null
[STEP] step=2 action=home_care reward=0.88 done=true error=null
[END] success=true steps=2 score=0.88 rewards=0.88,0.88
```

### Run Locally

```bash
export HF_TOKEN=your_token_here
export TASK_NAME=easy        # easy | medium | hard
python inference.py
```

---

## Three Tasks with Increasing Difficulty

| Task | Severity | Patient Profile | Correct Action | Max Reward |
|---|---|---|---|---|
| `easy` | mild | Single symptom, healthy adult | `home_care` | 1.0 |
| `medium` | moderate | Multi-symptom, no chronic condition | `gp_visit` / `telehealth` | 0.875 |
| `hard` | severe | Elderly patient with chronic condition | `emergency` | 1.0 |

Score = mean reward per episode, always in `[0, 1]`. Success threshold: `≥ 0.7`.

---

## Reward Design

- **Range:** `[0.0, 1.0]` via `(raw + 2) / 4`, raw clamped to `[-2, 2]`
- `1.0` — clinically correct and optimal
- `0.5` — partially correct / suboptimal
- `0.0` — dangerous under- or over-reaction
- **Anti-reward-hacking:** penalises both under- and over-reaction symmetrically

---

## Deployment (HF Spaces Docker SDK)

The Space starts `app.py` (Flask on port 7860) as the long-running process.
`inference.py` is invoked separately by the hackathon grader.

```bash
# Local web UI
pip install pydantic flask openai
python app.py    # → http://localhost:7860
```
