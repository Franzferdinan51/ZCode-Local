"""systemone.loop — See > Decide > Act, the one agent loop.

Every agentic use of SystemOne (demos, game benchmarks, operator tools)
runs through :class:`DecisionLoop`. One tick is:

    SEE     env.observe()  -> text (+ optional images) of the world now
    DECIDE  one batched judge call -> action + gates in a single pass
    ACT     env.act(action) -> progress signal; StallGuard trips on stalls

The loop bakes in the patterns every hand-rolled loop needs and none
should reimplement:

- byte-identical state prefixes (``system_prompt`` first, memory second,
  fresh observation last) so server-side prefix caches hit;
- compressed memory: only the latest observation travels raw, older turns
  collapse to "I saw / I thought / I did" one-liners;
- an uncertainty gate: low confidence (or low ``label_mass`` when the
  judge reports it) falls back to a safe action instead of acting on a
  guess;
- System 1 -> System 2 escalation: below ``escalate_below`` confidence
  (default 0.70, the JEV-27B-VL model card's operating point) a System 2
  advisor may override the action; without one the step is recorded as
  un-escalated and the loop continues on System 1;
- :class:`StallGuard` progress tracking with early stop.

Judges are backend-agnostic callables
``(state_text, questions, images, videos) -> answers`` returning
api-shaped answers (see ``systemone/examples/demo_decision_loop.py`` for the stub
and engine judges). Environments implement :class:`Env`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol, Sequence


@dataclass
class Observation:
    """SEE: what the world looks like this tick."""

    text: str
    images: Sequence[Any] = ()
    videos: Sequence[Any] = ()
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActResult:
    """ACT: what happened after the action."""

    progressed: bool
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Env(Protocol):
    """World the loop sees and acts in."""

    def observe(self) -> Observation: ...
    def act(self, action: str) -> ActResult: ...


@dataclass
class Step:
    """One recorded See > Decide > Act tick."""

    tick: int
    observation: str
    action: str
    confidence: float
    progressed: bool
    status: str
    latency_ms: float | None = None
    backend: str | None = None
    gated: bool = False
    escalated: bool = False
    escalation: str | None = None


@dataclass
class LoopResult:
    """Outcome of :meth:`DecisionLoop.run`."""

    outcome: str  # "done" | "stalled" | "budget"
    steps: List[Step] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)

    @property
    def n_ticks(self) -> int:
        return len(self.steps)


JudgeFn = Callable[..., Dict[str, Any]]
System2Fn = Callable[[str, List[str]], Any]


class DecisionLoop:
    """See > Decide > Act over any :class:`Env` with any judge.

    Args:
        judge: ``(state_text, questions, images, videos) -> answers`` in
            api shapes (``{name: {choice/confidence/...}, "_meta":
            {...}}``).
        action_name: which question's choice is the action.
        uncertainty_action: safe fallback when the gate trips.
        confidence_floor: below this, act ``uncertainty_action``.
        label_mass_floor: same gate for judges reporting label_mass.
        escalate_below: System 1 confidence below this asks ``system2``.
        system2: ``(state_text, options) -> action | None`` advisor.
        budget: max ticks. memory_lines: compressed turns carried.
        system_prompt: byte-identical prefix (cache-friendly).
    """

    def __init__(
        self,
        judge: JudgeFn,
        action_name: str = "action",
        uncertainty_action: str = "wait",
        confidence_floor: float = 0.35,
        label_mass_floor: float = 0.5,
        escalate_below: float = 0.70,
        system2: System2Fn | None = None,
        budget: int = 40,
        memory_lines: int = 6,
        system_prompt: str = "",
        max_stalls: int = 3,
    ) -> None:
        from .patterns import StallGuard

        self.judge = judge
        self.action_name = action_name
        self.uncertainty_action = uncertainty_action
        self.confidence_floor = confidence_floor
        self.label_mass_floor = label_mass_floor
        self.escalate_below = escalate_below
        self.system2 = system2
        self.budget = budget
        self.memory_lines = memory_lines
        self.system_prompt = system_prompt
        self.guard = StallGuard(max_stalls=max_stalls)

    def build_state_text(
        self, memory: Sequence[str], obs_text: str, tick: int
    ) -> str:
        """Cache-friendly state: fixed prefix, memory, fresh observation."""
        tail = list(memory[-self.memory_lines:] if self.memory_lines > 0 else [])
        mem = "\n".join(tail) if tail else "(no history yet)"
        prefix = f"{self.system_prompt}\n\n" if self.system_prompt else ""
        return (
            f"{prefix}"
            f"Memory (compressed turns):\n{mem}\n\n"
            f"Tick {tick} — current observation:\n{obs_text}"
        )

    def run(self, env: Env, questions: Sequence[Dict[str, Any]]) -> LoopResult:
        """Run See > Decide > Act until done, stalled, or budget."""
        self.guard.reset()
        memory: List[str] = []
        steps: List[Step] = []
        for tick in range(self.budget):
            obs = env.observe()  # SEE
            state_text = self.build_state_text(memory, obs.text, tick)
            answers = self.judge(
                state_text, list(questions), list(obs.images or []),
                list(obs.videos or []),
            )  # DECIDE
            meta = answers.get("_meta", {}) if isinstance(answers, dict) else {}

            action_answer = answers.get(self.action_name, {})
            action = str(action_answer.get("choice", self.uncertainty_action))
            try:
                conf = float(action_answer.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            try:
                mass = action_answer.get("label_mass")
                mass = None if mass is None else float(mass)
            except (TypeError, ValueError):
                mass = None

            gated = bool(
                (mass is not None and mass < self.label_mass_floor)
                or conf < self.confidence_floor
            )
            if gated:
                action = self.uncertainty_action

            escalated = False
            escalation: str | None = None
            if not gated and conf < self.escalate_below:
                if self.system2 is not None:
                    try:
                        override = self.system2(state_text, [action])
                    except Exception as exc:
                        override = None
                        escalation = f"system2 failed: {type(exc).__name__}"
                    if override:
                        action = str(override)
                        escalated = True
                        escalation = f"system2 override -> {action}"
                    elif escalation is None:
                        escalation = "system2 abstained; kept system1 action"
                else:
                    escalation = (
                        f"below {self.escalate_below:.2f}; no system2 configured"
                    )

            result = env.act(action)  # ACT
            status = self.guard.observe(bool(result.progressed))
            memory.append(
                f"tick {tick}: I saw {obs.text[:80].strip() or 'nothing new'}; "
                f"I thought action={action} (conf {conf:.2f}); "
                f"I did act and {'progressed' if result.progressed else 'stalled'}."
            )
            try:
                latency = meta.get("latency_ms")
                latency_ms = None if latency is None else float(latency)
            except (TypeError, ValueError):
                latency_ms = None
            steps.append(
                Step(
                    tick=tick,
                    observation=obs.text,
                    action=action,
                    confidence=conf,
                    progressed=bool(result.progressed),
                    status=status,
                    latency_ms=latency_ms,
                    backend=meta.get("backend"),
                    gated=gated,
                    escalated=escalated,
                    escalation=escalation,
                )
            )
            if bool(result.done) or bool(obs.done):
                return LoopResult(outcome="done", steps=steps, memory=memory)
            if status == "stalled":
                return LoopResult(outcome="stalled", steps=steps, memory=memory)
        return LoopResult(outcome="budget", steps=steps, memory=memory)


def run_loop(
    env: Env,
    judge: JudgeFn,
    questions: Sequence[Dict[str, Any]],
    **kwargs: Any,
) -> LoopResult:
    """One-shot See > Decide > Act: build a loop and run it."""
    return DecisionLoop(judge, **kwargs).run(env, questions)
