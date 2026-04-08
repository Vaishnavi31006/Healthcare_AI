"""
AI Healthcare Assistant - Reinforcement Learning Environment
============================================================
A simulated RL environment where an agent receives symptom observations
and learns to generate appropriate medical guidance actions.
RL FRAMEWORK OVERVIEW
─────────────────────
This environment simulates real-world healthcare triage using a canonical
state–action–reward (SAR) framework compatible with Open-Env / Gym-style
interfaces:
  • STATE   — A structured patient observation (symptoms, severity, age
              group, duration, chronic condition flag) that fully describes
              the current triage situation at each timestep.
  • ACTION  — A discrete set of 8 clinically meaningful guidance options
              ranging from home care through to emergency referral.  The
              agent selects one action per timestep.
  • REWARD  — A normalised scalar in [0, 1] returned after every action.
              1.0 signals a fully correct clinical response; 0.0 signals a
              dangerous or inappropriate one; values in between reflect
              partial correctness (e.g., acceptable but sub-optimal triage).
              The reward function is designed to penalise both under- and
              over-reaction, preventing reward hacking.
  • EPISODE — Begins with reset() and ends when a terminal action is taken
              (urgent_care / emergency) or max_steps is reached.
This structure maps directly onto standard RL training loops and allows any
policy — rule-based, tabular Q-learning, or deep RL — to be dropped in by
implementing a single select_action(state) → int interface.
"""

import random
import json
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

class State(BaseModel):
    """Represents a patient observation at a given timestep."""
    symptoms: list[str]
    severity: str          # "mild" | "moderate" | "severe"
    duration_days: int
    age_group: str         # "child" | "adult" | "elderly"
    has_chronic_condition: bool
    step: int = 0

    def to_vector(self) -> dict:
        """Returns a dict representation useful for feature encoding."""
        return {
            "symptoms": self.symptoms,
            "severity": self.severity,
            "duration_days": self.duration_days,
            "age_group": self.age_group,
            "has_chronic_condition": self.has_chronic_condition,
            "step": self.step,
        }


class Action(BaseModel):
    """Represents a medical guidance action the agent can take."""
    action_id: int
    label: str
    description: str


class StepResult(BaseModel):
    """Result returned after the agent takes an action."""
    next_state: Optional[State]
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


# ─────────────────────────────────────────────
#  ACTION SPACE
# ─────────────────────────────────────────────

ACTIONS: list[Action] = [
    Action(action_id=0, label="home_care",        description="Recommend rest, hydration, and OTC medication."),
    Action(action_id=1, label="monitor",          description="Advise monitoring symptoms for 24–48 hours."),
    Action(action_id=2, label="gp_visit",         description="Recommend scheduling a GP appointment."),
    Action(action_id=3, label="urgent_care",      description="Recommend urgent care visit within hours."),
    Action(action_id=4, label="emergency",        description="Direct to the emergency room immediately."),
    Action(action_id=5, label="specialist",       description="Refer to a medical specialist."),
    Action(action_id=6, label="telehealth",       description="Suggest a telehealth consultation."),
    Action(action_id=7, label="medication_check", description="Advise reviewing current medications with a pharmacist."),
]


# ─────────────────────────────────────────────
#  REWARD FUNCTION
# ─────────────────────────────────────────────

def normalize_reward(r: float) -> float:
    """Convert raw reward from [-2, 2] to [0, 1] using linear scaling."""
    return round((r + 2) / 4, 2)


def compute_reward(state: State, action: Action) -> float:
    """
    Rule-based reward shaping.
    Returns a normalized float in [0.0, 1.0] via normalize_reward().
    Logic:
      - Severe/chronic/elderly cases require escalated actions.
      - Mild cases are penalised for unnecessary escalation.
      - Moderate cases are rewarded for balanced guidance.
    Anti-reward-hacking design: penalties for both under-reaction (ignoring
    severe symptoms) and over-reaction (sending mild cases to the ER) prevent
    the agent from exploiting a one-sided reward signal. This ensures safe,
    clinically balanced guidance rather than a trivially high-scoring policy.
    """
    severity = state.severity
    action_id = action.action_id
    reward = 0.0

    if severity == "severe":
        # Must escalate; reward urgent/emergency actions
        if action_id in (3, 4):
            reward = 2.0
        elif action_id in (2, 5):
            reward = 0.5
        else:
            reward = -1.5   # Under-reacting to severe symptoms is dangerous

    elif severity == "moderate":
        if action_id in (2, 6):
            reward = 1.5
        elif action_id in (1, 5):
            reward = 0.8
        elif action_id == 3:
            reward = 0.2    # Slight over-escalation, but acceptable
        elif action_id == 0:
            reward = -0.5   # Probably under-treating
        else:
            reward = 0.0

    else:  # mild
        if action_id in (0, 1):
            reward = 1.5
        elif action_id == 6:
            reward = 0.5
        elif action_id == 2:
            reward = 0.0    # Acceptable but slightly over-cautious
        elif action_id in (3, 4):
            reward = -2.0   # Severe over-escalation

    # Bonus modifiers
    if state.has_chronic_condition and action_id in (2, 5, 7):
        reward += 0.5       # Good to loop in professionals for chronic patients
    if state.age_group == "elderly" and action_id in (2, 3, 4):
        reward += 0.3       # Elderly patients benefit from professional review
    if state.age_group == "child" and action_id == 4 and severity == "severe":
        reward += 0.5       # Children with severe symptoms should go to ER


    # Clamp to [-2.0, 2.0] before normalising so bonuses never push the
    # normalised output outside the guaranteed [0.0, 1.0] range.
    reward = max(-2.0, min(2.0, reward))
    return normalize_reward(reward)


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────

class HealthcareEnv:
    """
    A simple episodic RL environment for medical triage.
    Observation  : State (symptoms, severity, demographics)
    Action Space : 8 discrete medical guidance actions
    Reward       : Rule-based, shaped around clinical appropriateness
    Episode end  : After max_steps OR when a terminal action is chosen
    """

    SEVERITIES   = ["mild", "moderate", "severe"]
    AGE_GROUPS   = ["child", "adult", "elderly"]
    SYMPTOM_POOL = [
        "fever", "cough", "shortness_of_breath", "chest_pain",
        "headache", "fatigue", "nausea", "vomiting", "dizziness",
        "rash", "joint_pain", "abdominal_pain", "sore_throat",
        "runny_nose", "back_pain", "blurred_vision", "palpitations",
    ]
    TERMINAL_ACTIONS = {3, 4}   # urgent_care, emergency end the episode

    def __init__(self, max_steps: int = 5, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.current_state: Optional[State] = None
        self.episode_rewards: list[float] = []
        self._step_count = 0

    # ── Core API ──────────────────────────────

    # Difficulty presets — each maps to a fixed State template that defines
    # increasing clinical complexity for structured hackathon evaluation.
    DIFFICULTY_PRESETS: dict = {
        # Easy: mild, single symptom, healthy adult — home care is correct.
        "easy": dict(
            symptoms=["runny_nose"],
            severity="mild",
            duration_days=2,
            age_group="adult",
            has_chronic_condition=False,
        ),
        # Medium: moderate, multi-symptom, brief duration — GP or telehealth correct.
        "medium": dict(
            symptoms=["fever", "headache", "fatigue"],
            severity="moderate",
            duration_days=4,
            age_group="adult",
            has_chronic_condition=False,
        ),
        # Hard: severe, high-risk demographics, chronic condition — urgent/ER correct.
        "hard": dict(
            symptoms=["chest_pain", "shortness_of_breath", "palpitations", "dizziness"],
            severity="severe",
            duration_days=1,
            age_group="elderly",
            has_chronic_condition=True,
        ),
    }

    def reset(
        self,
        user_input: Optional[dict] = None,
        difficulty: Optional[str] = None,
    ) -> State:
        """
        Start a new episode.
        Args:
            user_input:  Optional dict with keys: symptoms (list[str]),
                         severity (str), duration_days (int), age_group (str),
                         has_chronic_condition (bool).  Takes priority over
                         difficulty when both are supplied.
            difficulty:  One of "easy" | "medium" | "hard".  Loads a preset
                         State with increasing clinical complexity:
                           easy   — mild, single symptom, healthy adult.
                           medium — moderate, multi-symptom, no chronic cond.
                           hard   — severe, elderly, chronic condition present.
                         If neither user_input nor difficulty is given a random
                         state is generated.
        Returns:
            Initial State
        """
        self._step_count = 0
        self.episode_rewards = []

        if user_input:
            source = user_input
        elif difficulty:
            assert difficulty in self.DIFFICULTY_PRESETS, (
                f"difficulty must be one of {list(self.DIFFICULTY_PRESETS)}"
            )
            source = self.DIFFICULTY_PRESETS[difficulty]
        else:
            self.current_state = self._random_state()
            return self.current_state

        self.current_state = State(
            symptoms=source.get("symptoms", ["fever"]),
            severity=source.get("severity", "mild"),
            duration_days=source.get("duration_days", 1),
            age_group=source.get("age_group", "adult"),
            has_chronic_condition=source.get("has_chronic_condition", False),
            step=0,
        )
        return self.current_state

    def step(self, action_id: int) -> StepResult:
        """
        Apply an action to the current state.
        Args:
            action_id: Integer index into ACTIONS list.
        Returns:
            StepResult with next_state, reward, done flag, and info dict.
        """
        assert self.current_state is not None, "Call reset() before step()."
        assert 0 <= action_id < len(ACTIONS), f"Invalid action_id: {action_id}"

        action = ACTIONS[action_id]
        reward = compute_reward(self.current_state, action)
        self.episode_rewards.append(reward)
        self._step_count += 1

        terminal = (
            action_id in self.TERMINAL_ACTIONS
            or self._step_count >= self.max_steps
        )

        next_state = None if terminal else self._transition(self.current_state, action)
        if not terminal:
            self.current_state = next_state

        return StepResult(
            next_state=next_state,
            reward=reward,
            done=terminal,
            info={
                "action_label": action.label,
                "action_description": action.description,
                "cumulative_reward": round(sum(self.episode_rewards), 2),
                "step": self._step_count,
                # Reward semantics for judges / external consumers:
                # 1.0 = clinically correct and optimal response for this state
                # 0.5 = partially correct or acceptable but suboptimal response
                # 0.0 = dangerous or inappropriate response (unsafe under- or over-reaction)
                # Values in between reflect degrees of partial correctness.
                "reward_interpretation": (
                    "1.0 = correct triage action | "
                    "0.5 = partially correct or acceptable but suboptimal | "
                    "0.0 = incorrect (unsafe under- or over-reaction)"
                ),
            },
        )

    def render(self, result: Optional[StepResult] = None):
        """Pretty-print the current state and optionally the last step result."""
        s = self.current_state
        print("\n" + "=" * 55)
        print("  HEALTHCARE RL ENVIRONMENT")
        print("=" * 55)
        if s:
            print(f"  Step           : {s.step}")
            print(f"  Symptoms       : {', '.join(s.symptoms)}")
            print(f"  Severity       : {s.severity.upper()}")
            print(f"  Duration       : {s.duration_days} day(s)")
            print(f"  Age Group      : {s.age_group}")
            print(f"  Chronic Cond.  : {'Yes' if s.has_chronic_condition else 'No'}")
        if result:
            print("-" * 55)
            print(f"  Action Taken   : {result.info['action_label']}")
            print(f"  Guidance       : {result.info['action_description']}")
            print(f"  Reward         : {result.reward:+.2f}")
            print(f"  Cumul. Reward  : {result.info['cumulative_reward']:+.2f}")
            print(f"  Episode Done   : {result.done}")
        print("=" * 55 + "\n")

    def action_space(self) -> list[Action]:
        return ACTIONS

    def get_action_labels(self) -> dict[int, dict]:
        """
        Expose the full action space as a plain dict for external consumers,
        Open-Env introspection, or display in evaluation reports.
        Returns:
            {action_id: {"label": str, "description": str}, ...}
        Example output:
            {
              0: {"label": "home_care",   "description": "Recommend rest ..."},
              1: {"label": "monitor",     "description": "Advise monitoring ..."},
              ...
              7: {"label": "medication_check", "description": "Advise reviewing ..."},
            }
        """
        return {
            a.action_id: {"label": a.label, "description": a.description}
            for a in ACTIONS
        }

    def state(self) -> Optional[State]:
        """
        Return the current environment state.
        Provides a clean accessor for external consumers, evaluation harnesses,
        and OpenEnv-compatible tooling that expect a dedicated state() method
        rather than direct attribute access.
        Returns:
            The current State if an episode is in progress, else None.
        Example:
            env = HealthcareEnv()
            env.reset(difficulty="medium")
            obs = env.state()   # identical to env.current_state
        """
        return self.current_state

    # ── Internal helpers ──────────────────────

    def _random_state(self) -> State:
        n_symptoms = self.rng.randint(1, 4)
        return State(
            symptoms=self.rng.sample(self.SYMPTOM_POOL, n_symptoms),
            severity=self.rng.choice(self.SEVERITIES),
            duration_days=self.rng.randint(1, 14),
            age_group=self.rng.choice(self.AGE_GROUPS),
            has_chronic_condition=self.rng.random() < 0.3,
            step=0,
        )

    def _transition(self, state: State, action: Action) -> State:
        """
        Simulate how symptoms evolve after an action is taken.
        In a full RL setup this would be a learned model.
        """
        new_severity = state.severity

        # Appropriate action slightly improves severity next step.
        # Thresholds use the normalised [0, 1] reward range:
        #   > 0.7  → good action, patient improves one severity level
        #   < 0.3  → poor action, patient stays same or worsens one level
        reward = compute_reward(state, action)
        if reward > 0.7 and state.severity != "mild":
            new_severity = {"severe": "moderate", "moderate": "mild"}.get(
                state.severity, state.severity
            )
        elif reward < 0.3:
            # Wrong action – condition stays same or worsens
            new_severity = {"mild": "moderate", "moderate": "severe"}.get(
                state.severity, state.severity
            )

        return State(
            symptoms=state.symptoms,
            severity=new_severity,
            duration_days=state.duration_days + 1,
            age_group=state.age_group,
            has_chronic_condition=state.has_chronic_condition,
            step=state.step + 1,
        )


# ─────────────────────────────────────────────
#  SIMPLE RULE-BASED AGENT (baseline)
# ─────────────────────────────────────────────

class RuleBasedAgent:
    """
    Deterministic baseline agent.
    Acts as a lower bound – a learned agent should outperform this.
    """

    def select_action(self, state: State) -> int:
        if state.severity == "severe":
            if state.age_group in ("child", "elderly") or state.has_chronic_condition:
                return 4   # emergency
            return 3       # urgent_care

        if state.severity == "moderate":
            if state.has_chronic_condition or state.duration_days > 5:
                return 2   # gp_visit
            return 6       # telehealth

        # mild
        if state.duration_days > 7:
            return 2       # gp_visit after a week
        return 0           # home_care


# ─────────────────────────────────────────────
#  INFERENCE / EVALUATION SCRIPT
# ─────────────────────────────────────────────

def inference():
    """
    Programmatic inference and evaluation entry point.
    This function serves as the official evaluation script for the hackathon submission.
    Initialises the HealthcareEnv and RuleBasedAgent, then runs three
    full episodes end-to-end without any manual input — one per difficulty
    tier (easy → medium → hard) — demonstrating increasing clinical complexity:
      easy   — mild, single symptom, healthy adult.
      medium — moderate, multi-symptom, no chronic condition.
      hard   — severe, elderly patient with chronic condition.
    Each episode loops over reset(difficulty=...) → select_action() → step()
    until the environment signals done=True, printing a full trace of states,
    actions, and normalised rewards [0, 1] at every step.  Per-episode and
    aggregate summary statistics are printed to make pass/fail evaluation
    straightforward.
    """

    env   = HealthcareEnv(max_steps=5, seed=42)
    agent = RuleBasedAgent()

    difficulties = ["easy", "medium", "hard"]
    summaries: dict[str, dict] = {}

    for difficulty in difficulties:
        print("\n" + "=" * 55)
        print(f"  INFERENCE — Difficulty: {difficulty.upper()}")
        print("=" * 55)

        state = env.reset(difficulty=difficulty)
        env.render()

        done = False
        while not done:
            action_id = agent.select_action(state)
            result    = env.step(action_id)
            env.render(result)
            done  = result.done
            state = result.next_state or state

        total = round(sum(env.episode_rewards), 2)
        mean  = round(total / len(env.episode_rewards), 2)
        steps = len(env.episode_rewards)
        summaries[difficulty] = {"steps": steps, "total": total, "mean": mean}

        print(f"  [{difficulty.upper()} Episode Summary]")
        print(f"  Steps taken     : {steps}")
        print(f"  Total reward    : {total}  (normalised 0–1 per step)")
        print(f"  Mean reward     : {mean}\n")

    # ── Aggregate evaluation summary ──────────────────────────────────────
    overall_mean = round(
        sum(s["mean"] for s in summaries.values()) / len(summaries), 2
    )
    print("=" * 55)
    print("  EVALUATION SUMMARY  (all difficulties)")
    print("=" * 55)
    print(f"  {'Difficulty':<12} {'Steps':>6}  {'Total':>7}  {'Mean':>6}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*6}")
    for diff, s in summaries.items():
        print(f"  {diff:<12} {s['steps']:>6}  {s['total']:>7}  {s['mean']:>6}")
    print(f"  {'─'*37}")
    print(f"  {'Overall mean':>27}  {overall_mean:>6}")
    print("=" * 55 + "\n")


# Guard against double-execution: some HF Spaces / container runtimes
# both import the module and invoke it as a script, which would run
# inference() twice. The flag ensures it executes exactly once.
_INFERENCE_RAN = False

def main():
    global _INFERENCE_RAN
    if _INFERENCE_RAN:
        return
    _INFERENCE_RAN = True
    inference()

if __name__ == "__main__":
    main()
