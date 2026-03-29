from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents import build_agents
from config import Settings
from run_modes import PipelineRunConfig, get_pipeline_run_config
from schemas import DialogueState


class MindBridgePipeline:
    def __init__(
        self,
        settings: Settings,
        run_config: PipelineRunConfig | None = None,
    ) -> None:
        self.settings = settings
        self.run_config = run_config or get_pipeline_run_config("full")
        self.agents = build_agents(settings)

    def _skipped_output(self, reason: str, **payload: object) -> dict[str, object]:
        result = {"_skipped": True, "reason": reason}
        result.update(payload)
        return result

    def _run_support_agents(
        self,
        user_input: str,
        state: DialogueState,
    ) -> None:
        shared_context = {
            "user_input": user_input,
            "analysis": state.analyzer,
        }
        max_workers = 3 if self.run_config.use_safety else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            empathy_future = executor.submit(
                self.agents["empathy"].run, shared_context
            )
            strategy_future = executor.submit(
                self.agents["strategy"].run, shared_context
            )
            safety_future = None
            if self.run_config.use_safety:
                safety_future = executor.submit(
                    self.agents["safety"].run, shared_context
                )

            state.empathy = empathy_future.result()
            state.strategy = strategy_future.result()
            if safety_future is None:
                state.safety = self._skipped_output(
                    "Safety ablation mode disabled this stage.",
                    risk_detected=False,
                    risk_level="unknown",
                    flags=[],
                    constraints=[],
                    required_actions=[],
                )
            else:
                state.safety = safety_future.result()

    def _run_revision_stage(self, user_input: str, state: DialogueState) -> str:
        if self.run_config.use_critic:
            state.critic = self.agents["critic"].run(
                {
                    "user_input": user_input,
                    "analysis": state.analyzer,
                    "draft_response": state.coordinator.get("draft_response", ""),
                    "safety": state.safety,
                }
            )
        else:
            state.critic = self._skipped_output(
                "Critic ablation mode disabled this stage.",
                issues=[],
                revision_goals=[],
            )

        if self.run_config.use_reviser:
            state.reviser = self.agents["reviser"].run(
                {
                    "user_input": user_input,
                    "analysis": state.analyzer,
                    "draft_response": state.coordinator.get("draft_response", ""),
                    "critic_feedback": state.critic,
                    "safety": state.safety,
                }
            )
            return state.reviser.get(
                "final_response",
                state.coordinator.get("draft_response", ""),
            )

        state.reviser = self._skipped_output(
            "Reviser ablation mode disabled this stage.",
            final_response="",
        )
        return state.coordinator.get("draft_response", "")

    def run(self, user_input: str) -> DialogueState:
        state = DialogueState(
            user_input=user_input,
            system_name="pipeline",
            run_mode=self.run_config.name,
        )

        state.analyzer = self.agents["analyzer"].run({"user_input": user_input})
        self._run_support_agents(user_input, state)

        state.coordinator = self.agents["coordinator"].run(
            {
                "user_input": user_input,
                "analysis": state.analyzer,
                "empathy": state.empathy,
                "strategy": state.strategy,
                "safety": state.safety,
            }
        )

        candidate_response = self._run_revision_stage(user_input, state)

        if self.run_config.use_safety:
            state.final_check = self.agents["final_checker"].run(
                {
                    "user_input": user_input,
                    "candidate_response": candidate_response,
                    "safety": state.safety,
                }
            )
            state.final_response = state.final_check.get(
                "final_response",
                candidate_response,
            )
        else:
            state.final_check = self._skipped_output(
                "Safety ablation mode skipped the final safety check.",
                approved=True,
                final_response=candidate_response,
                notes=[],
            )
            state.final_response = candidate_response

        return state
