"""The distress investigator: a real ReAct loop (SPEC 5, PROMPT hard rule 2).

The loop hypothesises, calls a typed tool, observes the structured result,
decides the next call *from that result*, and terminates on its own judgment --
including at *insufficient evidence*. It is not one large "analyse this" prompt;
the model chooses each step after seeing the previous observation, and the
branching is what makes the system agentic rather than a fixed pipeline.

Control flow the loop owns (not the model):

* Tool dispatch and argument validation.
* The step budget, and forced abstention when it is exhausted.
* The deterministic critic and the bounded retry (<=2), which terminates in
  abstention rather than in a guess.

Control flow the model owns:

* Which tool to call next, and with what arguments.
* When it has enough evidence to stop.
* The signal, the confidence, and the residual.

The model never computes. Every number in the output must have come from a tool
call, and the critic rejects any that did not.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from agents.critic import CriticReport, review
from agents.llm import LLMClient, _dispatch_complete_call, extract_json
from agents.schemas import (
    SIGNAL_INSUFFICIENT,
    Evidence,
    InvestigatorOutput,
)
from agents.tools import ToolBox, tool_schemas

#: Each step resends the whole conversation plus the tool schemas, so the token
#: cost is roughly quadratic in step count and linear in prompt size. Observed
#: successful runs used 8-10 tool calls, so 8 is a real budget rather than a
#: token-saving guess -- but it does trade some depth for throughput under a
#: daily token cap.
#: Raised to 14 after measurement: gemini-3.1-flash-lite used 8 investigative
#: calls and then needed room to conclude, exhausting a budget of 8 without
#: ever reaching finish. A budget that truncates the investigation turns a
#: capable model into an abstention, which would then be misread as caution.
MAX_STEPS = 14
MAX_RETRIES = 2

#: Steps remaining when the loop asks for a verdict. Two, so the model has a
#: turn to comply after being told -- warning it on the very last step leaves
#: no room to act on the warning.
FINISH_WARNING_STEPS = 2

TERMINATED_MODEL = "model_finished"
TERMINATED_BUDGET = "step_budget_exhausted"
TERMINATED_RETRIES = "retries_exhausted"
TERMINATED_MALFORMED = "unparseable_response"

# Every token here is resent on every step of every case. Measured: fixed
# per-call overhead was 62% of total token spend, and tokens-per-minute is the
# binding rate limit -- so prompt length, not model latency, sets the wall clock.
SYSTEM_PROMPT = """\
You are a credit-distress investigator. From SEC filings, judge how likely one \
company is to file for bankruptcy within 12 months, for a human analyst.

RULES
1. Never do arithmetic. Every number must come from a tool.
2. Assess risk; never recommend an action.
3. Tools already exclude anything filed after the prediction date.
4. If evidence is genuinely insufficient, finish with insufficient_evidence. \
A confident wrong answer is far worse than an honest abstention.

METHOD
Form a hypothesis, call ONE tool, read it, let the result choose your next call. \
Chase what looks wrong. An uncomputable metric is itself a finding -- distress \
removes tags. Most discriminating: quick_ratio, liabilities_to_assets, \
ohlson_o_score, return_on_assets, interest_coverage.

Finish when you have enough, or know you cannot get it. risk_score is 0-100 and \
is what you are ranked on: be granular, use the whole range, and never give two \
different companies the same score. confidence is 0.0-1.0, max 0.6 if you report \
a residual.
"""

USER_TEMPLATE = """\
Assess credit distress risk for CIK {cik} as of {as_of}.

Question: how likely is this company to file for bankruptcy protection within \
the next 12 months, judged only from filings public as of {as_of}?

Begin by finding out what periods are available."""


class DistressInvestigator:
    """A ReAct loop over the typed tools, wrapped in a deterministic critic."""

    def __init__(
        self,
        client: LLMClient,
        with_baseline: bool = False,
        max_steps: int = MAX_STEPS,
        max_retries: int = MAX_RETRIES,
        on_step: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client
        self._schemas = tool_schemas(with_baseline)
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.on_step = on_step

    # ---- tool dispatch -------------------------------------------------

    def _dispatch(self, tools: ToolBox, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "available_periods": tools.available_periods,
            "get_metric": tools.get_metric,
            "get_trend": tools.get_trend,
            "get_line_item": tools.get_line_item,
            "get_peer_comparison": tools.get_peer_comparison,
            "check_threshold": tools.check_threshold,
            "get_prior_distress_events": tools.get_prior_distress_events,
            "get_model_score": tools.get_model_score,
        }
        handler = handlers.get(name)
        if handler is None:
            return {
                "error": f"unknown tool '{name}'; available: {', '.join(sorted(handlers))}",
            }
        try:
            return handler(**(arguments or {}))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}

    # ---- the loop ------------------------------------------------------

    def _run_once(
        self, tools: ToolBox, messages: list[dict[str, str]]
    ) -> tuple[dict[str, Any] | None, str, int]:
        """Drive the loop until the model finishes or the budget runs out."""
        steps = 0
        warned = False
        for remaining in range(self.max_steps, 0, -1):
            # Ask for a verdict before the budget runs out rather than after.
            #
            # Measured: gemma-4-31b-it averaged 13.3 of 14 steps and failed to
            # conclude on 75% of cases. Those became step-budget abstentions --
            # recorded as protocol failures, which made a truncated
            # investigation look like the model declining to answer. The model
            # was not incapable, it was cut off mid-thought.
            if remaining <= FINISH_WARNING_STEPS and not warned:
                warned = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have one step left. Call finish now with your best "
                            "assessment from the evidence gathered so far. If that "
                            "evidence genuinely does not support a conclusion, finish "
                            "with signal 'insufficient_evidence' -- but do so as a "
                            "judgment, not by running out of steps."
                        ),
                    }
                )

            completion = _dispatch_complete_call(self.client, messages, tools=self._schemas)

            # Native tool calls take precedence. Several models emit them
            # whatever the prompt says -- gpt-oss-20b called a tool natively
            # and the provider rejected the request outright because none had
            # been declared. Accepting both shapes is more robust than
            # insisting on one.
            if completion.has_tool_calls:
                # Replay the provider's own message when we have it: some
                # providers attach fields to function calls that must come back
                # unchanged, and a reconstructed message loses them.
                messages.append(
                    completion.raw_message
                    or {
                        "role": "assistant",
                        "content": completion.content or None,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["name"],
                                    "arguments": json.dumps(c["arguments"]),
                                },
                            }
                            for c in completion.tool_calls
                        ],
                    }
                )
                finished = None
                for call in completion.tool_calls:
                    if call["name"] == "finish":
                        finished = call["arguments"]
                        observation: dict[str, Any] = {"ok": True}
                    else:
                        steps += 1
                        observation = self._dispatch(tools, call["name"], call["arguments"])
                        if self.on_step:
                            self.on_step(steps, {"tool": call["name"], "arguments": call["arguments"]})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(observation, default=str),
                        }
                    )
                if finished is not None:
                    return finished, TERMINATED_MODEL, steps
                continue

            reply = completion.content
            messages.append({"role": "assistant", "content": reply})
            parsed = extract_json(reply)

            if parsed is None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON. Reply with a single JSON object "
                            "using the documented format."
                        ),
                    }
                )
                steps += 1
                continue

            action = parsed.get("action")
            if action == "finish":
                return parsed, TERMINATED_MODEL, steps

            if action == "call_tool":
                steps += 1
                name = parsed.get("tool", "")
                arguments = parsed.get("arguments") or {}
                observation = self._dispatch(tools, name, arguments)
                if self.on_step:
                    self.on_step(steps, {"tool": name, "arguments": arguments})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation:\n{json.dumps(observation, default=str)}",
                    }
                )
                continue

            steps += 1
            messages.append(
                {
                    "role": "user",
                    "content": (
                        'Unknown "action". Use "call_tool" to investigate or '
                        '"finish" to conclude.'
                    ),
                }
            )

        return None, TERMINATED_BUDGET, steps

    def _build_output(
        self,
        parsed: dict[str, Any],
        tools: ToolBox,
        cik: int,
        as_of: date,
        steps: int,
        reason: str,
        retries: int,
        critic: CriticReport | None = None,
    ) -> InvestigatorOutput:
        evidence = []
        for item in parsed.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            raw = item.get("value")
            evidence.append(
                Evidence(
                    metric=str(item.get("metric", "")),
                    value=float(raw) if isinstance(raw, (int, float)) else None,
                    note=str(item.get("note", "")),
                )
            )
        raw_score = parsed.get("risk_score")
        try:
            risk_score = (
                max(0.0, min(100.0, float(raw_score))) if raw_score is not None else None
            )
        except (TypeError, ValueError):
            risk_score = None
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        return InvestigatorOutput(
            cik=cik,
            as_of=as_of,
            signal=str(parsed.get("signal", SIGNAL_INSUFFICIENT)),
            confidence=confidence,
            risk_score=risk_score,
            rationale=str(parsed.get("rationale", "")),
            evidence=evidence,
            residual=str(parsed.get("residual", "")),
            steps_taken=steps,
            terminated_because=reason,
            audit_trail=tools.audit_trail(),
            verification_passed=critic.passed if critic else True,
            verification_defects=(
                "; ".join(d.detail for d in critic.defects) if critic and critic.defects else ""
            ),
            retries=retries,
        )

    def _abstain(
        self, tools: ToolBox, cik: int, as_of: date, steps: int, reason: str, retries: int, note: str
    ) -> InvestigatorOutput:
        return InvestigatorOutput(
            cik=cik,
            as_of=as_of,
            signal=SIGNAL_INSUFFICIENT,
            confidence=0.0,
            rationale=note,
            residual=note,
            steps_taken=steps,
            terminated_because=reason,
            audit_trail=tools.audit_trail(),
            verification_passed=False,
            verification_defects=note,
            retries=retries,
        )

    def run(
        self,
        cik: int,
        as_of: date,
        facts: Sequence[Any],
        peer_facts: dict[int, Sequence[Any]] | None = None,
        sic_by_cik: dict[int, str] | None = None,
        events: Sequence[Any] = (),
        model_score: float | None = None,
    ) -> InvestigatorOutput:
        """Investigate one filer at one prediction date."""
        tools = ToolBox(
            cik=cik,
            as_of=as_of,
            facts=facts,
            peer_facts=peer_facts,
            sic_by_cik=sic_by_cik,
            events=events,
            model_score=model_score,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(cik=cik, as_of=as_of.isoformat()),
            },
        ]

        total_steps = 0
        for attempt in range(self.max_retries + 1):
            parsed, reason, steps = self._run_once(tools, messages)
            total_steps += steps

            if parsed is None:
                return self._abstain(
                    tools,
                    cik,
                    as_of,
                    total_steps,
                    reason,
                    attempt,
                    "the investigation did not reach a conclusion within its step budget",
                )

            candidate = self._build_output(
                parsed, tools, cik, as_of, total_steps, reason, attempt
            )
            critic = review(candidate, tools.cited, as_of, tools.cited_line_items)
            if critic.passed:
                return self._build_output(
                    parsed, tools, cik, as_of, total_steps, reason, attempt, critic
                )

            if attempt >= self.max_retries:
                # Bounded retries exhausted: abstain rather than ship a
                # response that failed a hard check.
                return self._abstain(
                    tools,
                    cik,
                    as_of,
                    total_steps,
                    TERMINATED_RETRIES,
                    attempt,
                    f"failed deterministic checks after {attempt + 1} attempts: "
                    + "; ".join(d.detail for d in critic.defects),
                )

            messages.append({"role": "user", "content": critic.feedback()})

        return self._abstain(
            tools, cik, as_of, total_steps, TERMINATED_RETRIES, self.max_retries, "unreachable"
        )
