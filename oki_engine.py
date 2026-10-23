"""
High-level Oki Engine combining deterministic frame data calculations,
Nash equilibrium solving, and situational strategic recommendations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from sfdojo.frame_data import (
    MoveData, KnockdownState, MeatyCalculator, MeatyResult, AttackLevel, CancelType
)
from sfdojo.payoff_matrix import (
    OkiScenario, PayoffMatrixBuilder, AttackerOption, DefenderOption
)
from sfdojo.nash_solver import NashSolver, EquilibriumResult


class OkiEngine:
    """End-to-end analyzer for SF6 Oki, Meaty Timings, and Interaction Distributions."""

    @staticmethod
    def get_character_presets() -> Dict[str, Dict]:
        """Provides realistic frame data presets for popular SF6 characters."""
        return {
            "Ken": {
                "moves": {
                    "5LP (Stand Light Punch)": MoveData(
                        name="5LP", startup=4, active_frames=3, recovery=7,
                        on_block=-1, on_hit=4, damage=300,
                        cancels=[CancelType.SPECIAL, CancelType.SUPER],
                        counter_hit_combo_damage=1800, punish_counter_combo_damage=2200
                    ),
                    "2LP (Crouch Light Punch)": MoveData(
                        name="2LP", startup=4, active_frames=3, recovery=6,
                        on_block=-1, on_hit=5, damage=300,
                        cancels=[CancelType.SPECIAL, CancelType.SUPER],
                        counter_hit_combo_damage=1800, punish_counter_combo_damage=2200
                    ),
                    "2MP (Crouch Medium Punch)": MoveData(
                        name="2MP", startup=6, active_frames=3, recovery=12,
                        on_block=-1, on_hit=4, damage=600,
                        cancels=[CancelType.SPECIAL, CancelType.DRIVE_RUSH],
                        counter_hit_combo_damage=2600, punish_counter_combo_damage=3400
                    ),
                    "5HP (Stand Heavy Punch)": MoveData(
                        name="5HP", startup=9, active_frames=3, recovery=19,
                        on_block=-3, on_hit=1, damage=800,
                        cancels=[CancelType.SPECIAL, CancelType.DRIVE_RUSH],
                        counter_hit_combo_damage=3200, punish_counter_combo_damage=4600
                    ),
                    "5HK (Stand Heavy Kick)": MoveData(
                        name="5HK", startup=12, active_frames=3, recovery=20,
                        on_block=-5, on_hit=3, damage=900,
                        counter_hit_combo_damage=3400, punish_counter_combo_damage=5100
                    ),
                },
                "whiff_durations": {
                    "whiff_2LP": 12,
                    "whiff_5LP": 13,
                    "whiff_2MP": 20,
                    "whiff_5MP": 22,
                    "forward_dash": 19,
                    "back_dash": 23
                },
                "knockdowns": {
                    "Forward Throw (Corner)": KnockdownState(
                        name="Forward Throw (Corner)", frame_advantage=23, position="corner"
                    ),
                    "Heavy Tatsu (Corner)": KnockdownState(
                        name="Heavy Tatsu (Corner)", frame_advantage=35, position="corner"
                    ),
                    "Safe Jump Setup (e.g. Jinrai Ender)": KnockdownState(
                        name="Safe Jump Setup", frame_advantage=42, position="midscreen"
                    ),
                    "Midscreen Forward Throw": KnockdownState(
                        name="Midscreen Forward Throw", frame_advantage=23, position="midscreen"
                    )
                }
            },
            "Luke": {
                "moves": {
                    "2LP (Crouch Light Punch)": MoveData(
                        name="2LP", startup=4, active_frames=3, recovery=6,
                        on_block=-1, on_hit=5, damage=300,
                        cancels=[CancelType.SPECIAL, CancelType.SUPER],
                        counter_hit_combo_damage=1900, punish_counter_combo_damage=2300
                    ),
                    "2MP (Crouch Medium Punch)": MoveData(
                        name="2MP", startup=6, active_frames=4, recovery=13,
                        on_block=-1, on_hit=4, damage=700,
                        cancels=[CancelType.SPECIAL, CancelType.DRIVE_RUSH],
                        counter_hit_combo_damage=2800, punish_counter_combo_damage=3600
                    ),
                    "5HP (Stand Heavy Punch)": MoveData(
                        name="5HP", startup=10, active_frames=4, recovery=17,
                        on_block=-3, on_hit=4, damage=850,
                        cancels=[CancelType.SPECIAL, CancelType.DRIVE_RUSH],
                        counter_hit_combo_damage=3300, punish_counter_combo_damage=4800
                    )
                },
                "whiff_durations": {
                    "whiff_2LP": 12,
                    "whiff_5LP": 13,
                    "whiff_2MP": 22,
                    "forward_dash": 20,
                    "back_dash": 23
                },
                "knockdowns": {
                    "Medium Flash Knuckle (Corner)": KnockdownState(
                        name="Medium Flash Knuckle (Corner)", frame_advantage=38, position="corner"
                    ),
                    "Light Knuckle Ender": KnockdownState(
                        name="Light Knuckle Ender", frame_advantage=42, position="midscreen"
                    )
                }
            }
        }

    @classmethod
    def solve_oki_game(cls, scenario: OkiScenario) -> Tuple[EquilibriumResult, np.ndarray, List[str], List[str]]:
        """
        Builds the payoff matrix and solves for the mixed-strategy Nash equilibrium.
        """
        matrix, att_ops, def_ops = PayoffMatrixBuilder.build(scenario)
        att_labels = [op.value for op in att_ops]
        def_labels = [op.value for op in def_ops]
        eq = NashSolver.solve_zero_sum(matrix, att_labels, def_labels)
        return eq, matrix, att_labels, def_labels

    @classmethod
    def evaluate_meaty_setups(
        cls,
        knockdown: KnockdownState,
        moves: Dict[str, MoveData],
        whiff_moves: Dict[str, int]
    ) -> List[Dict]:
        """
        Evaluates moves and finds auto-timed frame kill setups for exact meaty timings.
        """
        results = []
        for name, move in moves.items():
            # Check frame kills for active frame 1 (standard) and active frame A (max plus frames)
            for target_active in range(1, move.active_frames + 1):
                kills = MeatyCalculator.find_frame_kills(
                    knockdown_advantage=knockdown.frame_advantage,
                    target_move=move,
                    whiff_moves=whiff_moves,
                    target_active_frame=target_active,
                    max_moves=2
                )
                for seq, dur in kills:
                    res = MeatyCalculator.calculate_meaty(
                        knockdown=knockdown,
                        move=move,
                        frame_kill_duration=dur,
                        manual_delay=0
                    )
                    results.append({
                        "move": move.name,
                        "target_active_frame": target_active,
                        "frame_kill_sequence": seq,
                        "frame_kill_duration": dur,
                        "advantage_on_block": res.advantage_on_block,
                        "advantage_on_hit": res.advantage_on_hit,
                        "bonus_advantage": target_active - 1
                    })
        return results

    @classmethod
    def simulate_exploitative_response(
        cls,
        scenario: OkiScenario,
        defender_observed_freqs: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Calculates Attacker's best response against an observed/biased opponent distribution.
        """
        matrix, att_ops, def_ops = PayoffMatrixBuilder.build(scenario)
        att_labels = [op.value for op in att_ops]
        def_labels = [op.value for op in def_ops]

        # Convert observed frequencies into array matching def_labels
        q = np.zeros(len(def_labels))
        for j, label in enumerate(def_labels):
            q[j] = defender_observed_freqs.get(label, 0.0)

        total = np.sum(q)
        if total > 0:
            q = q / total
        else:
            q = np.ones(len(def_labels)) / len(def_labels)

        return NashSolver.calculate_best_response(matrix, att_labels, q)
