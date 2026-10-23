"""
Data structures for SF6 moves, frame data, knockdowns, and meaty timing calculations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple


class AttackLevel(Enum):
    HIGH = "High"       # Can be blocked standing or crouching
    MID = "Mid"         # Must be blocked standing (Overhead)
    LOW = "Low"         # Must be blocked crouching
    THROW = "Throw"     # Cannot be blocked
    AIR = "Air"         # Jump-in attack


class CancelType(Enum):
    NONE = "None"
    SPECIAL = "Special"
    SUPER = "Super"
    DRIVE_RUSH = "Drive Rush"
    TARGET_COMBO = "Target Combo"


@dataclass
class MoveData:
    name: str
    startup: int             # Frame number on which the first active frame begins (e.g. 4 for 4f jab)
    active_frames: int       # Duration of active hit window in frames (e.g. 3)
    recovery: int            # Recovery frames after active window
    on_block: int            # Frame advantage on block (hitting on 1st active frame)
    on_hit: int              # Frame advantage on hit (hitting on 1st active frame)
    damage: int              # Base damage
    attack_level: AttackLevel = AttackLevel.HIGH
    cancels: List[CancelType] = field(default_factory=lambda: [CancelType.NONE])
    punish_counter_combo_damage: Optional[int] = None
    counter_hit_combo_damage: Optional[int] = None

    @property
    def total_frames(self) -> int:
        """Total duration of move if it whiffs."""
        return (self.startup - 1) + self.active_frames + self.recovery

    def get_meaty_advantage(self, active_frame_index: int) -> Tuple[int, int]:
        """
        Calculate frame advantage on block and hit when connecting on a specific active frame.
        active_frame_index: 1-indexed (1 = first active frame, active_frames = last active frame)
        """
        if not (1 <= active_frame_index <= self.active_frames):
            raise ValueError(f"active_frame_index must be between 1 and {self.active_frames}")
        
        bonus = active_frame_index - 1
        return self.on_block + bonus, self.on_hit + bonus


@dataclass
class KnockdownState:
    name: str
    frame_advantage: int     # Number of frames attacker can act before defender can act
    position: str = "midscreen"  # "corner" or "midscreen"
    is_hard_knockdown: bool = False
    drive_advantage: float = 0.0  # Drive gauge diff


@dataclass
class MeatyResult:
    move_name: str
    connecting_active_frame: int  # 1 to active_frames
    is_valid_meaty: bool
    frames_late_or_early: int    # 0 = exact, negative = early (whiffs before wakeup), positive = late (opponent woke up and acted)
    advantage_on_block: int
    advantage_on_hit: int
    frame_kill_sequence: List[str]
    total_frames_killed: int
    manual_delay_frames: int


class MeatyCalculator:
    """Calculates exact meaty timings, frame kills, and frame advantages."""

    @staticmethod
    def calculate_meaty(
        knockdown: KnockdownState,
        move: MoveData,
        frame_kill_duration: int = 0,
        manual_delay: int = 0
    ) -> MeatyResult:
        """
        Calculates if a move hits meaty after a knockdown with given frame kills and delay.
        
        In SF6:
        - Opponent is actionable on relative frame 0 (meaning attacker has `knockdown.frame_advantage` frames
          of free time prior to opponent wakeup frame 1).
        - If attacker performs actions taking total `delay = frame_kill_duration + manual_delay` frames,
          then initiates `move`:
          The move's active frames will occupy relative frames:
          [delay + move.startup, delay + move.startup + move.active_frames - 1] relative to knockdown start.
        - Opponent wakes up on frame `knockdown.frame_advantage + 1`.
        """
        total_delay = frame_kill_duration + manual_delay
        # The frame on which the move starts active frames, relative to opponent wakeup (1 = frame 1 of wakeup):
        active_start_rel = (total_delay + move.startup) - knockdown.frame_advantage

        # An ideal meaty hits on active frame 1 to active_frames exactly when active_rel == 1
        # That is, active frame i satisfies: active_start_rel + (i - 1) == 1
        # => i = 2 - active_start_rel
        active_index = 2 - active_start_rel

        if 1 <= active_index <= move.active_frames:
            on_block, on_hit = move.get_meaty_advantage(active_index)
            return MeatyResult(
                move_name=move.name,
                connecting_active_frame=active_index,
                is_valid_meaty=True,
                frames_late_or_early=0,
                advantage_on_block=on_block,
                advantage_on_hit=on_hit,
                frame_kill_sequence=[],
                total_frames_killed=frame_kill_duration,
                manual_delay_frames=manual_delay
            )
        elif active_index > move.active_frames:
            # Active frames ended before opponent woke up (too early / whiffed)
            frames_early = active_index - move.active_frames
            return MeatyResult(
                move_name=move.name,
                connecting_active_frame=0,
                is_valid_meaty=False,
                frames_late_or_early=-frames_early,
                advantage_on_block=move.on_block,
                advantage_on_hit=move.on_hit,
                frame_kill_sequence=[],
                total_frames_killed=frame_kill_duration,
                manual_delay_frames=manual_delay
            )
        else:
            # Move hit after opponent woke up (too late, opponent can press 4f or jump)
            frames_late = 1 - active_index
            return MeatyResult(
                move_name=move.name,
                connecting_active_frame=0,
                is_valid_meaty=False,
                frames_late_or_early=frames_late,
                advantage_on_block=move.on_block,
                advantage_on_hit=move.on_hit,
                frame_kill_sequence=[],
                total_frames_killed=frame_kill_duration,
                manual_delay_frames=manual_delay
            )

    @staticmethod
    def find_frame_kills(
        knockdown_advantage: int,
        target_move: MoveData,
        whiff_moves: Dict[str, int],
        target_active_frame: int = 1,
        max_moves: int = 2
    ) -> List[Tuple[List[str], int]]:
        """
        Find sequences of whiffed moves that kill the exact required number of frames
        to make target_move connect on target_active_frame meaty.
        """
        # We need: total_killed + target_move.startup + (target_active_frame - 1) == knockdown_advantage + 1
        # => total_killed = (knockdown_advantage + 1) - (target_move.startup + target_active_frame - 1)
        required_kill = (knockdown_advantage + 1) - (target_move.startup + target_active_frame - 1)
        if required_kill < 0:
            return []

        solutions: List[Tuple[List[str], int]] = []

        def search(current_seq: List[str], current_frames: int):
            if current_frames == required_kill:
                solutions.append((list(current_seq), current_frames))
                return
            if len(current_seq) >= max_moves or current_frames > required_kill:
                return

            for name, dur in whiff_moves.items():
                if current_frames + dur <= required_kill:
                    current_seq.append(name)
                    search(current_seq, current_frames + dur)
                    current_seq.pop()

        search([], 0)
        return solutions
