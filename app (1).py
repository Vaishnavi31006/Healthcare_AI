"""
app.py — HuggingFace Spaces / Docker entry point
=================================================
Runs the RL inference pipeline on startup, captures all output,
and serves it as a clean HTML page on port 7860.
The Flask server stays alive so the container remains healthy.
"""

import io
import sys
import threading
from flask import Flask, Response

# Import the RL environment and inference logic
from healthcare_rl_env import (
    HealthcareEnv, RuleBasedAgent, ACTIONS,
    compute_reward, inference
)

app = Flask(__name__)

# ── Run inference once at startup and capture output ─────────────────────────

_results: dict = {"text": "Running inference, please wait...", "summaries": {}}

def _run_inference() -> None:
    """Execute inference(), capture stdout, store results globally."""
    buf = io.StringIO()
    sys.stdout = buf
    try:
        env   = HealthcareEnv(max_steps=5, seed=42)
        agent = RuleBasedAgent()
        difficulties = ["easy", "medium", "hard"]
        summaries = {}

        for difficulty in difficulties:
            state = env.reset(difficulty=difficulty)
            done  = False
            while not done:
                action_id = agent.select_action(state)
                result    = env.step(action_id)
                done      = result.done
                state     = result.next_state or state

            total = round(sum(env.episode_rewards), 2)
            mean  = round(total / len(env.episode_rewards), 2)
            steps = len(env.episode_rewards)
            summaries[difficulty] = {"steps": steps, "total": total, "mean": mean}

        overall_mean = round(
            sum(s["mean"] for s in summaries.values()) / len(summaries), 2
        )
        _results["summaries"]    = summaries
        _results["overall_mean"] = overall_mean
        _results["text"]         = "Inference complete."
    finally:
        sys.stdout = sys.__stdout__

# Run inference in background so Flask starts immediately
threading.Thread(target=_run_inference, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index() -> str:
    summaries    = _results.get("summaries", {})
    overall_mean = _results.get("overall_mean", "—")
    status       = _results.get("text", "")

    # Build rows for the summary table
    rows = ""
    for diff, s in summaries.items():
        badge_color = {"easy": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}.get(diff, "#6b7280")
        rows += f"""
        <tr>
          <td><span class="badge" style="background:{badge_color}">{diff.upper()}</span></td>
          <td>{s['steps']}</td>
          <td>{s['total']}</td>
          <td><strong>{s['mean']}</strong></td>
        </tr>"""

    overall_row = f"""
        <tr class="overall">
          <td colspan="3"><strong>Overall Mean Reward</strong></td>
          <td><strong>{overall_mean}</strong></td>
        </tr>""" if summaries else ""

    action_rows = "".join(
        f"<tr><td>{a.action_id}</td><td><code>{a.label}</code></td><td>{a.description}</td></tr>"
        for a in ACTIONS
    )

    waiting = "" if summaries else '<p class="waiting">⏳ Inference running — refresh in a moment…</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Healthcare Assistant — RL Environment</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }}
    h1   {{ font-size: 1.75rem; font-weight: 700; color: #38bdf8; margin-bottom: .25rem; }}
    h2   {{ font-size: 1.1rem; font-weight: 600; color: #94a3b8; margin: 2rem 0 .75rem; text-transform: uppercase; letter-spacing: .05em; }}
    p    {{ color: #94a3b8; margin-bottom: 1rem; line-height: 1.6; }}
    .hero  {{ max-width: 860px; margin: 0 auto; }}
    .subtitle {{ color: #64748b; font-size: .95rem; margin-bottom: 2rem; }}
    .card  {{ background: #1e293b; border: 1px solid #334155; border-radius: .75rem; padding: 1.5rem; margin-bottom: 1.5rem; }}
    table  {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
    th     {{ background: #0f172a; color: #38bdf8; padding: .6rem .9rem; text-align: left; font-weight: 600; }}
    td     {{ padding: .55rem .9rem; border-bottom: 1px solid #1e293b; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr.overall td {{ background: #0f172a; color: #38bdf8; }}
    .badge {{ display: inline-block; padding: .15rem .55rem; border-radius: 9999px; font-size: .75rem; font-weight: 700; color: #fff; }}
    code   {{ background: #0f172a; padding: .1rem .35rem; border-radius: .25rem; font-size: .85rem; color: #a5f3fc; }}
    .tag   {{ display: inline-block; background: #1e3a5f; color: #38bdf8; border-radius: .35rem; padding: .1rem .5rem; font-size: .78rem; margin-right: .3rem; }}
    .waiting {{ color: #f59e0b; font-style: italic; }}
    .reward-bar {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-top: .5rem; }}
    .reward-item {{ background: #0f172a; border-radius: .5rem; padding: .6rem 1rem; flex: 1; min-width: 140px; }}
    .reward-item .val {{ font-size: 1.4rem; font-weight: 700; color: #38bdf8; }}
    .reward-item .lbl {{ font-size: .75rem; color: #64748b; margin-top: .15rem; }}
  </style>
</head>
<body>
<div class="hero">
  <h1>🏥 AI Healthcare Assistant</h1>
  <p class="subtitle">Reinforcement Learning Triage Environment &nbsp;·&nbsp;
    <span class="tag">OpenEnv</span>
    <span class="tag">Pydantic</span>
    <span class="tag">Reward ∈ [0, 1]</span>
    <span class="tag">3 Difficulty Tasks</span>
  </p>

  <div class="card">
    <h2>Reward Scale</h2>
    <div class="reward-bar">
      <div class="reward-item"><div class="val">1.0</div><div class="lbl">Clinically correct &amp; optimal</div></div>
      <div class="reward-item"><div class="val">0.5</div><div class="lbl">Partially correct / suboptimal</div></div>
      <div class="reward-item"><div class="val">0.0</div><div class="lbl">Unsafe under- or over-reaction</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Inference Results — All Difficulty Tasks</h2>
    {waiting}
    <table>
      <thead><tr><th>Task</th><th>Steps</th><th>Total Reward</th><th>Mean Reward</th></tr></thead>
      <tbody>{rows}{overall_row}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Action Space (8 Discrete Actions)</h2>
    <table>
      <thead><tr><th>ID</th><th>Label</th><th>Description</th></tr></thead>
      <tbody>{action_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Environment Contract</h2>
    <p><strong>State</strong> — symptoms · severity (mild/moderate/severe) · duration_days · age_group · has_chronic_condition</p>
    <p><strong>Actions</strong> — 8 discrete clinical guidance options (home care → emergency)</p>
    <p><strong>Reward</strong> — Normalised to [0, 1] via <code>(raw + 2) / 4</code>. Penalises both under- and over-reaction to prevent reward hacking.</p>
    <p><strong>Episodes</strong> — End on terminal action (urgent_care / emergency) or after max 5 steps.</p>
    <p><strong>Tasks</strong> — <span class="badge" style="background:#22c55e">EASY</span> mild · <span class="badge" style="background:#f59e0b">MEDIUM</span> moderate · <span class="badge" style="background:#ef4444">HARD</span> severe/elderly/chronic</p>
  </div>
</div>
</body>
</html>"""


@app.route("/health")
def health() -> Response:
    """Health-check endpoint for container orchestration."""
    return Response("ok", status=200, mimetype="text/plain")


@app.route("/api/results")
def api_results() -> Response:
    """JSON endpoint exposing inference results for programmatic consumers."""
    import json
    return Response(
        json.dumps({
            "summaries":    _results.get("summaries", {}),
            "overall_mean": _results.get("overall_mean", None),
            "status":       _results.get("text", ""),
        }),
        status=200,
        mimetype="application/json",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Port 7860 is required by HuggingFace Spaces Docker SDK
    app.run(host="0.0.0.0", port=7860, debug=False)
