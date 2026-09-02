"""
Curriculum learning management system.
Controls progressive stage advancement from endgame sub-problems to full 8x8 chess.
"""

from typing import List, Optional, Dict, Any
from chess_rl.config.schema import CurriculumConfig, CurriculumStageConfig
from chess_rl.utils.logging import get_logger


class CurriculumManager:
    """
    Orchestrates progressive curriculum stages based on game budgets and performance thresholds.
    """

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.enabled = config.enabled
        self.stages: List[CurriculumStageConfig] = config.stages if self.enabled else []
        self.current_stage_idx: int = 0
        self.games_in_current_stage: int = 0
        self.total_curriculum_games: int = 0
        self.stage_history: List[Dict[str, Any]] = []
        self.logger = get_logger("curriculum_manager")

    @property
    def current_stage(self) -> Optional[CurriculumStageConfig]:
        if not self.enabled or not self.stages:
            return None
        return self.stages[self.current_stage_idx]

    @property
    def is_final_stage(self) -> bool:
        if not self.enabled or not self.stages:
            return True
        return self.current_stage_idx >= len(self.stages) - 1

    def record_games(self, num_games: int) -> None:
        """Increment played game count."""
        self.games_in_current_stage += num_games
        self.total_curriculum_games += num_games

    def check_stage_progression(self, current_win_rate: Optional[float] = None) -> bool:
        """
        Evaluate if current stage completion criteria are satisfied.

        Args:
            current_win_rate: Optional evaluation win-rate achieved on current stage.

        Returns:
            True if stage was advanced, False otherwise.
        """
        if not self.enabled or self.is_final_stage:
            return False

        stage = self.current_stage
        if stage is None:
            return False

        advance = False
        reason = ""

        # Check maximum games upper limit
        if stage.max_games is not None and self.games_in_current_stage >= stage.max_games:
            advance = True
            reason = f"Reached max_games limit ({stage.max_games})"

        # Check performance triggered threshold
        elif (
            stage.target_win_rate is not None
            and current_win_rate is not None
            and self.games_in_current_stage >= stage.min_games
        ):
            if current_win_rate >= stage.target_win_rate:
                advance = True
                reason = f"Met target win rate {current_win_rate:.2f} >= {stage.target_win_rate:.2f}"

        if advance:
            self._advance_stage(reason, current_win_rate)
            return True

        return False

    def _advance_stage(self, reason: str, final_win_rate: Optional[float]) -> None:
        """Execute stage advancement."""
        old_stage = self.current_stage
        self.stage_history.append(
            {
                "stage_id": old_stage.stage_id,
                "stage_name": old_stage.name,
                "games_played": self.games_in_current_stage,
                "final_win_rate": final_win_rate,
                "transition_reason": reason,
            }
        )
        self.logger.info(
            f"Advancing curriculum from stage {old_stage.stage_id} ({old_stage.name}) "
            f"to stage {self.current_stage_idx + 1}. Reason: {reason}"
        )
        self.current_stage_idx += 1
        self.games_in_current_stage = 0

    def get_status(self) -> Dict[str, Any]:
        """Return curriculum status snapshot for telemetry logging."""
        stage = self.current_stage
        return {
            "curriculum_enabled": self.enabled,
            "stage_index": self.current_stage_idx if self.enabled else 0,
            "stage_name": stage.name if stage else "none",
            "is_final_stage": self.is_final_stage,
            "games_in_stage": self.games_in_current_stage,
            "total_games": self.total_curriculum_games,
        }
