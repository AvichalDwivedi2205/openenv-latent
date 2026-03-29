"""Main environment implementation for LatentGoalOps."""

from __future__ import annotations

import json
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from latentgoalops.experiment import ExperimentConfig
from latentgoalops.models import (
    FeedbackLabel,
    GoalArchetype,
    GraderResult,
    ItemLabelAssignment,
    ItemPriorityAssignment,
    LatentGoalOpsAction,
    LatentGoalOpsObservation,
    LatentGoalOpsState,
    MessagingAction,
    SupportPolicy,
    TaskDescriptor,
    TaskId,
)
from latentgoalops.server.config import load_config
from latentgoalops.server.grader import (
    grade_task1,
    grade_task2,
    grade_task3,
    grade_task4,
    grade_task5,
    latent_score_from_dashboard,
    trajectory_coherence,
)
from latentgoalops.server.hidden_goals import (
    HiddenGoal,
    active_goal_name,
    active_weights,
    compute_utility,
    sample_hidden_goal,
)
from latentgoalops.server.rewards import compute_proxy_reward, strategy_embedding
from latentgoalops.server.tasks.task1_feedback import build_task1_episode
from latentgoalops.server.tasks.task2_prioritization import (
    build_task2_episode,
    selection_value,
)
from latentgoalops.server.tasks.task4_capital_allocation import (
    allocation_value,
    build_task4_episode,
)
from latentgoalops.server.tasks.task3_startup_week import (
    apply_task3_action,
    build_task3_episode,
    build_task3_view,
    evaluate_task3_action_value,
    solve_task3_oracle_action,
)
from latentgoalops.server.tasks.task5_crisis_response import (
    build_task5_episode,
    evaluate_task5_action_value,
    solve_task5_oracle_action,
)
from latentgoalops.server.world import build_world


class LatentGoalOpsEnvironment(
    Environment[LatentGoalOpsAction, LatentGoalOpsObservation, LatentGoalOpsState]
):
    """OpenEnv environment for latent-goal startup operations."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, experiment_config: ExperimentConfig | None = None) -> None:
        super().__init__()
        self._config = experiment_config or ExperimentConfig()
        self._rng = random.Random(0)
        self._seed = 0
        self._state = LatentGoalOpsState(episode_id=str(uuid4()), step_count=0)
        self._hidden_goal: HiddenGoal | None = None
        self._task_id: TaskId | None = None
        self._episode: dict | None = None
        self._last_grade: GraderResult | None = None
        self._previous_action: LatentGoalOpsAction | None = None

    @staticmethod
    def describe_tasks() -> list[TaskDescriptor]:
        """Static task catalog for docs and API exposure."""
        return [
            TaskDescriptor(
                task_id=TaskId.TASK1,
                difficulty="easy",
                objective="Label, prioritize, and escalate customer feedback while inferring the hidden business objective.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK2,
                difficulty="medium",
                objective="Select the best roadmap slice under budget when stakeholder clues only indirectly reveal the objective.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK3,
                difficulty="hard",
                objective="Operate a startup across a dated calendar, reason about delayed effects, detect a silent goal shift, and re-align policy without direct supervision.",
                horizon=10,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK4,
                difficulty="medium",
                objective="Allocate capital across visible programs with diminishing returns, interactions, and indirect stakeholder clues about what outcome matters most.",
                horizon=1,
            ),
            TaskDescriptor(
                task_id=TaskId.TASK5,
                difficulty="hard",
                objective="Assemble one crisis-response package from initiatives plus pricing, messaging, and support policy while inferring the hidden business objective.",
                horizon=1,
            ),
        ]

    def get_metadata(self) -> EnvironmentMetadata:
        """Expose environment metadata and README content."""
        readme_path = Path(__file__).resolve().parents[3] / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
        return EnvironmentMetadata(
            name="LatentGoalOps",
            description="A realistic startup-operations benchmark for latent objective inference and non-stationary goal adaptation.",
            readme_content=readme_content,
            version="0.1.0",
            author="Avichal Dwivedi",
        )

    @staticmethod
    def _infer_goal_hint_from_evidence(evidence: str) -> str:
        """Infer a rough visible objective from narrative evidence only."""
        lowered = evidence.lower()
        if any(keyword in lowered for keyword in ("renewal", "account health", "trust", "support backlog", "fragile")):
            return "retention"
        if any(keyword in lowered for keyword in ("pricing", "monetization", "board", "commercial", "pipeline")):
            return "revenue"
        if any(keyword in lowered for keyword in ("margin", "latency", "cost-to-serve", "ops", "runway discipline")):
            return "efficiency"
        return "growth"

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: object,
    ) -> LatentGoalOpsObservation:
        """Reset the environment to a clean episode."""
        self._seed = 0 if seed is None else int(seed)
        self._rng = random.Random(self._seed)
        self._task_id = TaskId(task_id or TaskId.TASK1.value)
        self._hidden_goal = sample_hidden_goal(
            self._rng,
            allow_shift=self._task_id == TaskId.TASK3 and self._config.enable_hidden_shift,
        )
        self._last_grade = None
        self._previous_action = None
        world = build_world(random.Random(self._seed + 10_003))

        task_config = load_config("tasks.yaml")
        if self._task_id == TaskId.TASK1:
            low, high = task_config["task1"]["item_count_range"]
            self._episode = build_task1_episode(
                self._rng,
                self._hidden_goal,
                self._rng.randint(low, high),
                world,
            )
            max_steps = 1
            budget_remaining = 0.0
            capacity_remaining = 0.0
        elif self._task_id == TaskId.TASK2:
            budget = int(task_config["task2"]["sprint_budget"])
            self._episode = build_task2_episode(
                self._rng,
                self._hidden_goal,
                budget,
                world,
                split=self._config.scenario_split,
            )
            max_steps = 1
            budget_remaining = float(budget)
            capacity_remaining = 0.0
        elif self._task_id == TaskId.TASK3:
            horizon = int(self._config.task3_horizon_override or task_config["task3"]["horizon"])
            budget = float(task_config["task3"]["budget_per_episode"])
            capacity = float(task_config["task3"]["capacity_per_episode"])
            self._episode = build_task3_episode(
                self._rng,
                self._hidden_goal,
                horizon,
                budget,
                capacity,
                world,
                split=self._config.scenario_split,
            )
            self._episode["initial_budget"] = budget
            self._episode["initial_capacity"] = capacity
            self._episode["seed"] = self._seed
            self._episode["enable_delayed_effects"] = self._config.enable_delayed_effects
            max_steps = horizon
            budget_remaining = budget
            capacity_remaining = capacity
        elif self._task_id == TaskId.TASK4:
            budget = int(task_config["task4"]["allocation_budget"])
            self._episode = build_task4_episode(
                self._rng,
                self._hidden_goal,
                budget,
                world,
                split=self._config.scenario_split,
            )
            max_steps = 1
            budget_remaining = float(budget)
            capacity_remaining = 0.0
        else:
            budget = float(task_config["task5"]["budget"])
            capacity = int(task_config["task5"]["capacity"])
            self._episode = build_task5_episode(
                self._rng,
                self._hidden_goal,
                budget,
                capacity,
                world,
                split=self._config.scenario_split,
            )
            max_steps = 1
            budget_remaining = budget
            capacity_remaining = float(capacity)

        self._state = LatentGoalOpsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task_id,
            max_steps=max_steps,
            sim_date=self._episode.get("start_date") if self._task_id == TaskId.TASK3 else None,
            sim_day_label="Day 1" if self._task_id == TaskId.TASK3 else None,
            budget_remaining=budget_remaining,
            capacity_remaining=capacity_remaining,
            decision_count=0,
            pending_effect_count=0,
            shift_active=False,
            completed=False,
            cumulative_reward=0.0,
            last_score=None,
        )
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: LatentGoalOpsAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> LatentGoalOpsObservation:
        """Take a step in the environment."""
        del timeout_s
        if self._episode is None or self._task_id is None or self._hidden_goal is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if action.task_id != self._task_id:
            raise ValueError(f"Action task_id={action.task_id.value} does not match active task {self._task_id.value}.")
        if self._state.completed:
            return self._build_observation(reward=0.0, done=True)

        if self._task_id == TaskId.TASK1:
            observation = self._step_task1(action)
        elif self._task_id == TaskId.TASK2:
            observation = self._step_task2(action)
        elif self._task_id == TaskId.TASK3:
            observation = self._step_task3(action)
        elif self._task_id == TaskId.TASK4:
            observation = self._step_task4(action)
        else:
            observation = self._step_task5(action)

        self._previous_action = action
        return observation

    def _step_task1(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        grade = grade_task1(
            true_labels=self._episode["true_labels"],
            true_priorities=self._episode["true_priorities"],
            oracle_escalations=self._episode["oracle_escalations"],
            action_payload=action.model_dump(mode="json"),
            hidden_goal=self._hidden_goal,
        )
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = grade.score
        return self._build_observation(
            reward=grade.score,
            done=True,
            metadata={"grader": grade.model_dump(mode="json")},
        )

    def _step_task2(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        backlog = self._episode["backlog"]
        valid_ids = {item.item_id for item in backlog}
        selected_ids = [item_id for item_id in action.selected_item_ids if item_id in valid_ids]
        spent_budget = sum(item.cost for item in backlog if item.item_id in selected_ids)
        budget = float(self._episode["sprint_budget"])
        invalid = spent_budget > budget + 1e-9
        if invalid:
            selected_ids = []
            spent_budget = 0.0
        agent_value = selection_value(selected_ids, backlog, self._hidden_goal.weights)
        unused_budget_ratio = (budget - spent_budget) / max(budget, 1.0)
        grade = grade_task2(
            agent_value=agent_value,
            random_baseline=float(self._episode["random_value"]),
            oracle_value=float(self._episode["oracle_value"]),
            unused_budget_ratio=unused_budget_ratio,
        )
        reward = max(0.0, grade.score - (0.15 if invalid else 0.0))
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = reward
        self._state.budget_remaining = max(0.0, budget - spent_budget)
        return self._build_observation(
            reward=reward,
            done=True,
            metadata={
                "grader": grade.model_dump(mode="json"),
                "spent_budget": spent_budget,
                "invalid_budget": invalid,
            },
        )

    def _step_task3(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        current_step_index = self._episode["step_index"]
        current_weights = active_weights(self._hidden_goal, current_step_index)
        oracle_action, oracle_decision_value = solve_task3_oracle_action(self._episode, current_weights)
        del oracle_action
        agent_decision_value = evaluate_task3_action_value(self._episode, action, current_weights)
        decision_quality = (
            1.0
            if oracle_decision_value <= 1e-9
            else max(0.0, min(1.0, agent_decision_value / oracle_decision_value))
        )
        previous_metrics = self._episode["dashboard"].to_metric_vector()
        step_result = apply_task3_action(self._rng, self._hidden_goal, self._episode, action)
        self._state.step_count += 1

        current_step_index = self._episode["step_index"]
        shift_active = (
            self._hidden_goal.shift_step is not None and current_step_index >= self._hidden_goal.shift_step
        )
        self._state.shift_active = shift_active
        self._state.budget_remaining = float(self._episode["budget_remaining"])
        self._state.capacity_remaining = float(self._episode["capacity_remaining"])

        new_metrics = self._episode["dashboard"].to_metric_vector()
        reward = compute_proxy_reward(
            previous_metrics=previous_metrics,
            new_metrics=new_metrics,
            previous_action=self._previous_action,
            current_action=action,
            invalid=step_result["invalid"],
            unused_budget_ratio=self._episode["budget_remaining"] / max(self._episode["initial_budget"], 1.0),
        )
        reward -= 0.05 * len(step_result["governance_flags"])
        if self._config.reward_mode == "sparse":
            reward = 0.0
        self._state.cumulative_reward += reward
        self._episode["decision_quality_history"].append(
            {
                "step_index": current_step_index - 1,
                "quality": round(float(decision_quality), 4),
            }
        )

        done = current_step_index >= self._episode["horizon"]
        active_goal = active_goal_name(self._hidden_goal, current_step_index)
        metadata = {
            "invalid": step_result["invalid"],
            "events": [event["name"] for event in step_result["step_events"]],
            "scheduled_effect_ids": [effect.effect_id for effect in step_result["scheduled_effects"]],
            "realized_effect_ids": [effect.effect_id for effect in step_result["realized_effects"]],
            "decision_id": step_result["decision_id"],
            "shift_active": shift_active,
            "active_goal": active_goal,
            "governance_flags": step_result["governance_flags"],
            "decision_quality": round(float(decision_quality), 4),
        }
        if done:
            weights = active_weights(self._hidden_goal, current_step_index)
            final_utility = compute_utility(
                self._episode["dashboard"].to_metric_vector(),
                HiddenGoal(
                    archetype=GoalArchetype(active_goal),
                    weights=weights,
                    alpha=self._hidden_goal.alpha,
                ),
            )
            if self._hidden_goal.shift_step is None:
                adaptation_score = 1.0
            else:
                post_shift_qualities = [
                    row["quality"]
                    for row in self._episode["decision_quality_history"]
                    if row["step_index"] >= self._hidden_goal.shift_step
                ]
                if not post_shift_qualities:
                    adaptation_score = 0.0
                else:
                    quality_weights = list(range(len(post_shift_qualities), 0, -1))
                    adaptation_score = sum(
                        quality * weight
                        for quality, weight in zip(post_shift_qualities, quality_weights)
                    ) / sum(quality_weights)
            coherence_score = trajectory_coherence(
                self._episode["action_history"],
                split_index=self._hidden_goal.shift_step,
            )
            constraint_score = max(
                0.0,
                1.0 - (
                    self._episode["invalid_actions"] + 0.5 * self._episode["policy_violations"]
                ) / max(self._episode["horizon"], 1),
            )
            grade = grade_task3(
                latent_utility=final_utility,
                adaptation_score=adaptation_score,
                coherence_score=coherence_score,
                constraint_score=constraint_score,
            )
            self._last_grade = grade
            self._state.last_score = grade.score
            self._state.completed = True
            if self._config.reward_mode == "sparse":
                reward = grade.score
                self._state.cumulative_reward += grade.score
            metadata["grader"] = grade.model_dump(mode="json")

        return self._build_observation(reward=reward, done=done, metadata=metadata)

    def _step_task4(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        backlog = self._episode["backlog"]
        valid_ids = {item.item_id for item in backlog}
        raw_allocations = {
            item_id: float(amount)
            for item_id, amount in action.budget_allocations.items()
            if item_id in valid_ids and float(amount) > 0.0
        }
        invalid = any(float(amount) < 0.0 for amount in action.budget_allocations.values())
        allocations = {item_id: float(int(round(amount))) for item_id, amount in raw_allocations.items()}
        spent_budget = sum(allocations.values())
        budget = float(self._episode["sprint_budget"])
        if spent_budget > budget + 1e-9:
            invalid = True
            allocations = {}
            spent_budget = 0.0
        agent_value = allocation_value(allocations, backlog, self._hidden_goal.weights)
        budget_use_ratio = spent_budget / max(budget, 1.0)
        grade = grade_task4(
            agent_value=agent_value,
            random_baseline=float(self._episode["random_value"]),
            oracle_value=float(self._episode["oracle_value"]),
            budget_use_ratio=budget_use_ratio,
        )
        reward = max(0.0, grade.score - (0.15 if invalid else 0.0))
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = reward
        self._state.budget_remaining = max(0.0, budget - spent_budget)
        return self._build_observation(
            reward=reward,
            done=True,
            metadata={
                "grader": grade.model_dump(mode="json"),
                "spent_budget": spent_budget,
                "invalid_budget": invalid,
                "resolved_allocations": allocations,
            },
        )

    def _step_task5(self, action: LatentGoalOpsAction) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._hidden_goal is not None
        self._state.step_count += 1
        backlog = self._episode["backlog"]
        valid_ids = {item.item_id for item in backlog}
        chosen_ids = [item_id for item_id in action.chosen_initiatives if item_id in valid_ids]
        spent_budget = sum(item.cost for item in backlog if item.item_id in chosen_ids)
        capacity_used = len(chosen_ids)
        invalid = spent_budget > float(self._episode["budget_remaining"]) + 1e-9 or capacity_used > int(
            self._episode["capacity_remaining"]
        )
        scored_action = action
        if invalid:
            chosen_ids = []
            spent_budget = 0.0
            capacity_used = 0
            scored_action = action.model_copy(update={"chosen_initiatives": []})
        else:
            scored_action = action.model_copy(update={"chosen_initiatives": chosen_ids})
        raw_value, constraint_score, flags = evaluate_task5_action_value(
            self._episode,
            scored_action,
            self._hidden_goal.weights,
        )
        composite_value = raw_value + 0.12 * constraint_score
        grade = grade_task5(
            agent_value=composite_value,
            random_baseline=float(self._episode["random_value"]),
            oracle_value=float(self._episode["oracle_value"]),
            constraint_score=constraint_score,
        )
        reward = max(0.0, grade.score - (0.15 if invalid else 0.0))
        self._last_grade = grade
        self._state.completed = True
        self._state.last_score = grade.score
        self._state.cumulative_reward = reward
        self._state.budget_remaining = max(0.0, float(self._episode["budget_remaining"]) - spent_budget)
        self._state.capacity_remaining = max(0.0, float(self._episode["capacity_remaining"]) - capacity_used)
        return self._build_observation(
            reward=reward,
            done=True,
            metadata={
                "grader": grade.model_dump(mode="json"),
                "spent_budget": spent_budget,
                "capacity_used": capacity_used,
                "invalid_selection": invalid,
                "governance_flags": flags,
            },
        )

    def _available_actions(self) -> list[str]:
        if self._task_id == TaskId.TASK1:
            return [
                "Assign labels for every feedback item.",
                "Set priorities from 1-5.",
                "Escalate up to three item IDs.",
            ]
        if self._task_id == TaskId.TASK2:
            return [
                "Select a subset of roadmap item IDs within budget.",
                "Optionally provide a short rationale_summary.",
            ]
        if self._task_id == TaskId.TASK4:
            return [
                "Allocate discrete budget points across visible program IDs using budget_allocations.",
                "Respect each program's allocation_max and the overall sprint_budget.",
                "Account for diminishing returns, dependencies, conflicts, and stakeholder pressure.",
            ]
        if self._task_id == TaskId.TASK5:
            return [
                "Choose a small crisis-response package from the visible initiative IDs.",
                "Optionally adjust pricing, messaging, and support policy in the same decision.",
                "Respect visible governance constraints around pricing, SLA handling, and margin stress.",
            ]
        return [
            "Choose initiatives for the current day.",
            "Optionally change pricing, messaging, and support policy.",
            "Reason over delayed effects, calendar events, account renewals, and prior decisions.",
            "Respect visible governance constraints around pricing, SLA handling, and margin pressure.",
            "Stay coherent across steps and adapt if the objective silently shifts.",
        ]

    def _build_observation(
        self,
        reward: float,
        done: bool,
        metadata: dict | None = None,
    ) -> LatentGoalOpsObservation:
        assert self._episode is not None
        assert self._task_id is not None
        if self._task_id == TaskId.TASK1:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                dashboard=self._episode["dashboard"],
                inbox=self._episode["inbox"],
                backlog=[],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=[],
                budget_remaining=0.0,
                capacity_remaining=0.0,
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )
        if self._task_id == TaskId.TASK2:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                dashboard=self._episode["dashboard"],
                inbox=[],
                backlog=self._episode["backlog"],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=[],
                budget_remaining=self._state.budget_remaining,
                capacity_remaining=0.0,
                sprint_budget=self._episode["sprint_budget"],
                stakeholder_notes=self._episode["stakeholder_notes"],
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )
        if self._task_id == TaskId.TASK4:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                dashboard=self._episode["dashboard"],
                inbox=[],
                backlog=self._episode["backlog"],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=[],
                budget_remaining=self._state.budget_remaining,
                capacity_remaining=0.0,
                sprint_budget=self._episode["sprint_budget"],
                stakeholder_notes=self._episode["stakeholder_notes"],
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )
        if self._task_id == TaskId.TASK5:
            return LatentGoalOpsObservation(
                task_id=self._task_id,
                step_index=self._state.step_count,
                horizon=1,
                task_summary=self._episode["task_summary"],
                narrative=self._episode["narrative"],
                dashboard=self._episode["dashboard"],
                inbox=self._episode["inbox"],
                backlog=self._episode["backlog"],
                accounts=self._episode["accounts"],
                stakeholders=self._episode["stakeholders"],
                teams=self._episode["teams"],
                market_context=self._episode["market_context"],
                governance_constraints=self._episode["governance_constraints"],
                alerts=self._episode["alerts"],
                budget_remaining=self._state.budget_remaining,
                capacity_remaining=self._state.capacity_remaining,
                available_actions=self._available_actions(),
                done=done,
                reward=reward,
                metadata=metadata or {},
            )

        view_rng = random.Random(self._seed + self._episode["step_index"] * 9973)
        task3_view = build_task3_view(view_rng, self._hidden_goal, self._episode)
        if not self._config.expose_decision_ledger:
            task3_view["decision_ledger"] = []
            task3_view["pending_effects"] = []
            task3_view["realized_effects"] = []
            task3_view["memory_summary"] = None
        self._state.sim_date = task3_view["sim_date"]
        self._state.sim_day_label = task3_view["sim_day_label"]
        self._state.decision_count = len(task3_view["decision_ledger"])
        self._state.pending_effect_count = len(task3_view["pending_effects"])
        return LatentGoalOpsObservation(
            task_id=self._task_id,
            step_index=task3_view["step_index"],
            horizon=task3_view["horizon"],
            sim_date=task3_view["sim_date"],
            sim_day_label=task3_view["sim_day_label"],
            task_summary=task3_view["task_summary"],
            narrative=task3_view["narrative"],
            dashboard=task3_view["dashboard"],
            inbox=task3_view["inbox"],
            backlog=task3_view["backlog"],
            accounts=task3_view["accounts"],
            stakeholders=task3_view["stakeholders"],
            teams=task3_view["teams"],
            market_context=task3_view["market_context"],
            governance_constraints=task3_view["governance_constraints"],
            alerts=task3_view["alerts"],
            calendar_events=task3_view["calendar_events"],
            decision_ledger=task3_view["decision_ledger"],
            pending_effects=task3_view["pending_effects"],
            realized_effects=task3_view["realized_effects"],
            budget_remaining=task3_view["budget_remaining"],
            capacity_remaining=task3_view["capacity_remaining"],
            memory_summary=task3_view["memory_summary"],
            available_actions=self._available_actions(),
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    def sample_random_action(self) -> LatentGoalOpsAction:
        """Sample a deterministic random action for the active state."""
        if self._episode is None or self._task_id is None:
            raise RuntimeError("Reset before sampling actions.")
        rng = random.Random(self._seed + 17 * (self._state.step_count + 1))
        if self._task_id == TaskId.TASK1:
            labels = []
            priorities = []
            for item in self._episode["inbox"]:
                labels.append(ItemLabelAssignment(item_id=item.item_id, label=rng.choice(list(FeedbackLabel))))
                priorities.append(ItemPriorityAssignment(item_id=item.item_id, priority=rng.randint(1, 5)))
            escalate_count = min(3, len(self._episode["inbox"]))
            escalate_ids = [item.item_id for item in rng.sample(self._episode["inbox"], k=escalate_count)]
            return LatentGoalOpsAction(task_id=self._task_id, labels=labels, priorities=priorities, escalate_ids=escalate_ids)
        if self._task_id == TaskId.TASK2:
            shuffled = self._episode["backlog"][:]
            rng.shuffle(shuffled)
            remaining = self._episode["sprint_budget"]
            selected_ids = []
            for item in shuffled:
                if item.cost <= remaining and rng.random() > 0.5:
                    selected_ids.append(item.item_id)
                    remaining -= item.cost
            return LatentGoalOpsAction(task_id=self._task_id, selected_item_ids=selected_ids)
        if self._task_id == TaskId.TASK4:
            remaining = int(round(float(self._episode["sprint_budget"])))
            allocations: dict[str, float] = {}
            shuffled = self._episode["backlog"][:]
            rng.shuffle(shuffled)
            for item in shuffled:
                if remaining <= 0:
                    break
                max_amount = min(remaining, int(round(float(item.allocation_max or 0.0))))
                amount = rng.randint(0, max_amount)
                if amount > 0:
                    allocations[item.item_id] = float(amount)
                    remaining -= amount
            return LatentGoalOpsAction(task_id=self._task_id, budget_allocations=allocations)
        if self._task_id == TaskId.TASK5:
            visible = self._episode["backlog"][:]
            rng.shuffle(visible)
            chosen: list[str] = []
            remaining_budget = float(self._episode["budget_remaining"])
            for item in visible:
                if len(chosen) >= int(self._episode["capacity_remaining"]):
                    break
                if item.cost > remaining_budget:
                    continue
                if rng.random() > 0.5:
                    chosen.append(item.item_id)
                    remaining_budget -= item.cost
            return LatentGoalOpsAction(
                task_id=self._task_id,
                chosen_initiatives=chosen,
                messaging_action=rng.choice(list(MessagingAction)),
                pricing_change_pct=rng.choice([-0.06, -0.03, 0.0, 0.03, 0.06]),
                support_policy=rng.choice(list(SupportPolicy)),
            )
        visible = [item for item in self._episode["backlog"] if item.item_id not in self._episode["completed_ids"]]
        rng.shuffle(visible)
        chosen = [item.item_id for item in visible[: rng.randint(0, min(2, len(visible)))]]
        return LatentGoalOpsAction(
            task_id=self._task_id,
            chosen_initiatives=chosen,
            messaging_action=rng.choice(list(MessagingAction)),
            pricing_change_pct=round(rng.uniform(-0.08, 0.08), 3),
            support_policy=rng.choice(list(SupportPolicy)),
        )

    def sample_heuristic_action(self) -> LatentGoalOpsAction:
        """Generate a simple heuristic baseline action from visible context only."""
        if self._episode is None or self._task_id is None:
            raise RuntimeError("Reset before sampling actions.")
        if self._task_id == TaskId.TASK1:
            keyword_map = {
                "charged": FeedbackLabel.BILLING_ISSUE,
                "invoice": FeedbackLabel.BILLING_ISSUE,
                "cancel": FeedbackLabel.CHURN_RISK,
                "renew": FeedbackLabel.CHURN_RISK,
                "slow": FeedbackLabel.LATENCY_COMPLAINT,
                "latency": FeedbackLabel.LATENCY_COMPLAINT,
                "crash": FeedbackLabel.BUG,
                "blank screen": FeedbackLabel.BUG,
                "please add": FeedbackLabel.FEATURE_REQUEST,
                "want": FeedbackLabel.FEATURE_REQUEST,
            }
            labels = []
            priorities = []
            scored = []
            for item in self._episode["inbox"]:
                lowered = item.text.lower()
                predicted = FeedbackLabel.PRAISE
                for keyword, label in keyword_map.items():
                    if keyword in lowered:
                        predicted = label
                        break
                severity = int(item.metadata.get("severity", 3))
                tier = str(item.metadata.get("user_tier", "pro"))
                priority = min(5, max(1, severity + (1 if tier == "enterprise" else 0)))
                labels.append(ItemLabelAssignment(item_id=item.item_id, label=predicted))
                priorities.append(ItemPriorityAssignment(item_id=item.item_id, priority=priority))
                scored.append((item.item_id, priority))
            scored.sort(key=lambda row: row[1], reverse=True)
            return LatentGoalOpsAction(
                task_id=self._task_id,
                labels=labels,
                priorities=priorities,
                escalate_ids=[item_id for item_id, _ in scored[:3]],
            )
        if self._task_id == TaskId.TASK2:
            ranked = sorted(
                self._episode["backlog"],
                key=lambda item: (sum(max(0.0, value) for value in item.kpi_deltas.values()) / max(item.cost, 1.0)),
                reverse=True,
            )
            remaining = self._episode["sprint_budget"]
            selected_ids = []
            for item in ranked:
                if item.cost <= remaining:
                    selected_ids.append(item.item_id)
                    remaining -= item.cost
            return LatentGoalOpsAction(task_id=self._task_id, selected_item_ids=selected_ids, rationale_summary="Prioritized high visible impact per budget.")
        if self._task_id == TaskId.TASK4:
            ranked = sorted(
                self._episode["backlog"],
                key=lambda item: (
                    sum(max(0.0, value) for value in item.kpi_deltas.values()),
                    -float(item.implementation_risk),
                ),
                reverse=True,
            )
            remaining = int(round(float(self._episode["sprint_budget"])))
            allocations: dict[str, float] = {}
            for item in ranked:
                if remaining <= 0:
                    break
                preferred = min(
                    remaining,
                    int(round(float(item.saturation_point or item.allocation_max or 0.0))),
                )
                if preferred <= 0:
                    continue
                allocations[item.item_id] = float(preferred)
                remaining -= preferred
            return LatentGoalOpsAction(
                task_id=self._task_id,
                budget_allocations=allocations,
                rationale_summary="Allocated budget toward the strongest visible programs until returns visibly started tapering.",
            )
        if self._task_id == TaskId.TASK5:
            evidence = " ".join(
                [self._episode.get("narrative", "")]
                + [message.text for message in self._episode.get("inbox", [])]
                + list(self._episode.get("alerts", []))
            )
            goal_hint = self._infer_goal_hint_from_evidence(evidence)
            inferred_weights = {
                "growth": {"growth": 0.55, "retention": 0.20, "revenue": 0.15, "efficiency": 0.10},
                "retention": {"growth": 0.15, "retention": 0.55, "revenue": 0.15, "efficiency": 0.15},
                "revenue": {"growth": 0.10, "retention": 0.20, "revenue": 0.55, "efficiency": 0.15},
                "efficiency": {"growth": 0.10, "retention": 0.15, "revenue": 0.15, "efficiency": 0.60},
            }[goal_hint]
            heuristic_action, _, _ = solve_task5_oracle_action(self._episode, inferred_weights)
            return heuristic_action.model_copy(
                update={"rationale": "Heuristic crisis-response package based on visible signals."}
            )

        view_rng = random.Random(self._seed + self._episode["step_index"] * 9973)
        view = build_task3_view(view_rng, self._hidden_goal, self._episode)
        evidence = " ".join(
            [view.get("narrative", "")]
            + [message.text for message in view["inbox"]]
            + list(view["alerts"])
        )
        goal_hint = self._infer_goal_hint_from_evidence(evidence)
        ranked = sorted(
            [item for item in self._episode["backlog"] if item.item_id not in self._episode["completed_ids"]],
            key=lambda item: (
                1 if item.kind == goal_hint else 0,
                sum(max(0.0, value) for value in item.kpi_deltas.values()) / max(item.cost, 1.0),
            ),
            reverse=True,
        )
        chosen_ids = [item.item_id for item in ranked[:2]]
        message_map = {
            "growth": MessagingAction.GROWTH_PUSH,
            "retention": MessagingAction.RETENTION_CAMPAIGN,
            "revenue": MessagingAction.REVENUE_UPSELL,
            "efficiency": MessagingAction.COST_COMMS,
        }
        support_map = {
            "growth": SupportPolicy.BALANCED_TRIAGE,
            "retention": SupportPolicy.PREMIUM_SLA,
            "revenue": SupportPolicy.BALANCED_TRIAGE,
            "efficiency": SupportPolicy.AUTOMATION_FIRST,
        }
        return LatentGoalOpsAction(
            task_id=self._task_id,
            chosen_initiatives=chosen_ids,
            messaging_action=message_map[goal_hint],
            pricing_change_pct=0.04 if goal_hint == "revenue" else (-0.03 if goal_hint == "growth" else 0.0),
            support_policy=support_map[goal_hint],
            rationale="Heuristic policy based on visible backlog and alert signals.",
        )

    def sample_oracle_action(self) -> LatentGoalOpsAction:
        """Construct an oracle action using hidden episode state."""
        if self._episode is None or self._task_id is None or self._hidden_goal is None:
            raise RuntimeError("Reset before sampling actions.")
        if self._task_id == TaskId.TASK1:
            return LatentGoalOpsAction(
                task_id=self._task_id,
                labels=[
                    ItemLabelAssignment(item_id=item_id, label=label)
                    for item_id, label in self._episode["true_labels"].items()
                ],
                priorities=[
                    ItemPriorityAssignment(item_id=item_id, priority=priority)
                    for item_id, priority in self._episode["true_priorities"].items()
                ],
                escalate_ids=self._episode["oracle_escalations"],
            )
        if self._task_id == TaskId.TASK2:
            return LatentGoalOpsAction(
                task_id=self._task_id,
                selected_item_ids=self._episode["oracle_selection"],
                rationale_summary="Oracle roadmap plan.",
            )
        if self._task_id == TaskId.TASK4:
            return LatentGoalOpsAction(
                task_id=self._task_id,
                budget_allocations=self._episode["oracle_allocations"],
                rationale_summary="Oracle capital allocation.",
            )
        if self._task_id == TaskId.TASK5:
            return self._episode["oracle_action"].model_copy(
                update={"rationale": "Oracle crisis-response package."}
            )

        step_index = self._episode["step_index"]
        active_goal = active_goal_name(self._hidden_goal, step_index)
        weights = active_weights(self._hidden_goal, step_index)
        planning_weights = dict(weights)
        if (
            self._hidden_goal.shift_step is not None
            and self._hidden_goal.shift_weights is not None
            and step_index < self._hidden_goal.shift_step
        ):
            current_window = max(self._hidden_goal.shift_step - step_index, 1)
            future_window = max(self._episode["horizon"] - self._hidden_goal.shift_step, 1)
            total_window = current_window + future_window
            planning_weights = {
                channel: (
                    float(weights[channel]) * current_window
                    + float(self._hidden_goal.shift_weights[channel]) * future_window
                )
                / total_window
                for channel in weights
            }
        oracle_action, _ = solve_task3_oracle_action(self._episode, planning_weights)
        return oracle_action.model_copy(
            update={"rationale": f"Oracle action using latent goal knowledge for {active_goal}."}
        )

    @property
    def state(self) -> LatentGoalOpsState:
        """Return current public environment state."""
        return self._state

    @property
    def last_grade(self) -> GraderResult | None:
        """Return last terminal grade."""
        return self._last_grade
