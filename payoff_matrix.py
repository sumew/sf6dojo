"""
Game theoretic payoff matrix definitions for SF6 Oki scenarios.
Supports situational modifiers: Midscreen vs Corner, Resource states (Burnout, Super Meter).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


class AttackerOption(Enum):
    MEATY_STRIKE = "Meaty Strike"        # Timed frame 1 meaty button
    MEATY_THROW = "Meaty Throw"          # 5f throw timed frame 1
    SHIMMY = "Shimmy"                    # Walk back to bait delay tech / DP
    BLOCK_BAIT = "Block / Bait"          # Block in place to bait DP
    DELAY_STRIKE = "Delay Strike (Trap)" # Slight delay frame trap (beats mash/jump)


class DefenderOption(Enum):
    BLOCK = "Block"                      # Block high/low
    MASH_4F = "Wakeup Mash (4f)"         # 4-frame normal
    INVINCIBLE_REVERSAL = "OD DP / Reversal" # Frame 1 fully invincible reversal
    DELAY_TECH = "Delay Tech"            # Fuzzy guard: block 2-3f then throw tech
    WAKEUP_JUMP = "Wakeup Jump"          # Jump up or back (throw invincible frame 1)
    WAKEUP_BACKDASH = "Wakeup Backdash"  # Backdash (throw invincible frame 1)
    PERFECT_PARRY = "Perfect Parry"      # Tap parry frame 1


@dataclass
class OkiScenario:
    """Configurable environment parameters for Oki payoff calculation."""
    position: str = "midscreen"          # "midscreen" or "corner"
    defender_has_reversal: bool = True   # Has OD meter or Super meter for reversal
    defender_in_burnout: bool = False    # Burnout disables OD & Parry, increases blockstun
    defender_has_super: bool = True      # Can use Level 1/2/3 if in burnout or no OD
    attacker_strike_damage: int = 2800   # Counter-hit / confirm combo from meaty strike
    attacker_throw_damage: int = 1200    # Standard forward/back throw damage
    attacker_shimmy_punish: int = 4200   # Punish counter combo from baited whiff throw
    attacker_dp_punish: int = 4800       # Punish counter combo from blocked DP
    defender_dp_damage: int = 1500       # Damage dealt by defender's DP
    defender_jump_punish: int = 3400     # Damage defender gets by jumping over whiffed throw
    defender_parry_punish: int = 1600    # Scaled (50%) damage from Perfect Parry punish


class PayoffMatrixBuilder:
    """
    Constructs a 2-player zero-sum payoff matrix where values represent
    Attacker's Net Expected Value (positive = favorable to Attacker, negative = favorable to Defender).
    """

    @classmethod
    def build(cls, scenario: OkiScenario) -> Tuple[np.ndarray, List[AttackerOption], List[DefenderOption]]:
        # Filter available options based on game state
        attacker_ops = [
            AttackerOption.MEATY_STRIKE,
            AttackerOption.MEATY_THROW,
            AttackerOption.SHIMMY,
            AttackerOption.BLOCK_BAIT,
            AttackerOption.DELAY_STRIKE
        ]

        defender_ops = [
            DefenderOption.BLOCK,
            DefenderOption.MASH_4F,
            DefenderOption.DELAY_TECH,
            DefenderOption.WAKEUP_JUMP,
            DefenderOption.WAKEUP_BACKDASH
        ]

        # In SF6, reversal requires either OD meter (not in burnout) or Super Art
        can_reversal = (not scenario.defender_in_burnout and scenario.defender_has_reversal) or \
                       (scenario.defender_in_burnout and scenario.defender_has_super)
        if can_reversal:
            defender_ops.insert(2, DefenderOption.INVINCIBLE_REVERSAL)

        # In Burnout, Drive Parry / Perfect Parry is unavailable
        if not scenario.defender_in_burnout:
            defender_ops.append(DefenderOption.PERFECT_PARRY)

        matrix = np.zeros((len(attacker_ops), len(defender_ops)), dtype=float)

        for i, a in enumerate(attacker_ops):
            for j, d in enumerate(defender_ops):
                matrix[i, j] = cls._calculate_cell_payoff(a, d, scenario)

        return matrix, attacker_ops, defender_ops

    @classmethod
    def _calculate_cell_payoff(
        cls,
        attacker: AttackerOption,
        defender: DefenderOption,
        scenario: OkiScenario
    ) -> float:
        """
        Determines the net utility for Attacker given both choices.
        Utility reflects expected health differential + positional value.
        """
        is_corner = (scenario.position.lower() == "corner")
        in_burnout = scenario.defender_in_burnout

        # Base payoffs by interaction
        if attacker == AttackerOption.MEATY_STRIKE:
            if defender == DefenderOption.BLOCK:
                # Block: Attacker gains frame/drive chip advantage (+50 dmg equivalent; +150 in burnout)
                return 150.0 if in_burnout else 50.0
            elif defender == DefenderOption.MASH_4F:
                # Strike crushes 4f mash on counter-hit
                return float(scenario.attacker_strike_damage)
            elif defender == DefenderOption.INVINCIBLE_REVERSAL:
                # Reversal cleanly beats meaty strike
                return -float(scenario.defender_dp_damage)
            elif defender == DefenderOption.DELAY_TECH:
                # Delay tech blocks the strike cleanly
                return 150.0 if in_burnout else 50.0
            elif defender == DefenderOption.WAKEUP_JUMP:
                # Strike hits prejump frames (cannot block in prejump)
                return float(scenario.attacker_strike_damage * 0.85)
            elif defender == DefenderOption.WAKEUP_BACKDASH:
                # In corner, backdash cannot escape and gets hit; midscreen might get clipped
                return float(scenario.attacker_strike_damage * (0.9 if is_corner else 0.5))
            elif defender == DefenderOption.PERFECT_PARRY:
                # Defender gets 50% damage scaled punish
                return -float(scenario.defender_parry_punish)

        elif attacker == AttackerOption.MEATY_THROW:
            if defender == DefenderOption.BLOCK:
                # Throw beats block cleanly
                return float(scenario.attacker_throw_damage + (300 if is_corner else 100))
            elif defender == DefenderOption.MASH_4F:
                # Frame 1 meaty throw in SF6 beats 4f startup normal
                return float(scenario.attacker_throw_damage)
            elif defender == DefenderOption.INVINCIBLE_REVERSAL:
                # Reversal beats throw
                return -float(scenario.defender_dp_damage)
            elif defender == DefenderOption.DELAY_TECH:
                # Throw is teched (neutral reset, 0 damage)
                return 0.0
            elif defender == DefenderOption.WAKEUP_JUMP:
                # Throw whiffs completely; defender gets full airborne punish combo
                return -float(scenario.defender_jump_punish)
            elif defender == DefenderOption.WAKEUP_BACKDASH:
                # Throw whiffs against backdash; defender gets punish counter combo
                punish = scenario.defender_jump_punish * 0.8
                return -float(punish if not is_corner else punish * 0.6)
            elif defender == DefenderOption.PERFECT_PARRY:
                # Parry CANNOT tech throw! Triggers Punish Counter Throw (+70% damage in SF6)
                return float(scenario.attacker_throw_damage * 1.7)

        elif attacker == AttackerOption.SHIMMY:
            if defender == DefenderOption.BLOCK:
                # Defender holds block: small neutral reset
                return 0.0
            elif defender == DefenderOption.MASH_4F:
                # If 4f reaches attacker walking back, defender gets small poke (e.g. cr.MK/jab);
                # If it whiffs, attacker whiffs punishes. Average slight defender favor or neutral.
                return -300.0 if is_corner else -200.0
            elif defender == DefenderOption.INVINCIBLE_REVERSAL:
                # Attacker is blocking out of range: Max Punish Counter Combo!
                bonus = 600 if is_corner else 0
                return float(scenario.attacker_dp_punish + bonus)
            elif defender == DefenderOption.DELAY_TECH:
                # Defender whiffs throw -> Attacker lands huge Punish Counter Combo!
                bonus = 700 if is_corner else 0
                return float(scenario.attacker_shimmy_punish + bonus)
            elif defender == DefenderOption.WAKEUP_JUMP:
                # Defender jumps: in corner, attacker anti-airs; midscreen defender escapes
                return 1200.0 if is_corner else -400.0
            elif defender == DefenderOption.WAKEUP_BACKDASH:
                # In corner backdash is trapped; midscreen distance reset
                return 1800.0 if is_corner else 0.0
            elif defender == DefenderOption.PERFECT_PARRY:
                # Parry attempt whiffs / does nothing
                return 0.0

        elif attacker == AttackerOption.BLOCK_BAIT:
            if defender == DefenderOption.BLOCK:
                return 0.0
            elif defender == DefenderOption.MASH_4F:
                # Defender presses jab, attacker blocks -> Defender is plus or neutral (+1 on jab block)
                return -200.0
            elif defender == DefenderOption.INVINCIBLE_REVERSAL:
                # DP blocked -> Full Punish Counter Combo
                bonus = 600 if is_corner else 0
                return float(scenario.attacker_dp_punish + bonus)
            elif defender == DefenderOption.DELAY_TECH:
                return 0.0
            elif defender == DefenderOption.WAKEUP_JUMP:
                # Defender jumps away
                return 1000.0 if is_corner else -500.0
            elif defender == DefenderOption.WAKEUP_BACKDASH:
                return 1200.0 if is_corner else -200.0
            elif defender == DefenderOption.PERFECT_PARRY:
                return 0.0

        elif attacker == AttackerOption.DELAY_STRIKE:
            # Attacker delays strike by 2-3 frames (frame trap / shimmy poke)
            if defender == DefenderOption.BLOCK:
                return 50.0
            elif defender == DefenderOption.MASH_4F:
                # Opponent's 4f jab starts up on frame 1 and hits on frame 4;
                # Attacker's delayed button gets counter-hit cleanly
                return -float(scenario.defender_dp_damage * 0.9)
            elif defender == DefenderOption.INVINCIBLE_REVERSAL:
                # Reversal hits attacker during delay
                return -float(scenario.defender_dp_damage)
            elif defender == DefenderOption.DELAY_TECH:
                # Catches opponent's delayed throw startup -> Counter Hit combo!
                return float(scenario.attacker_strike_damage * 0.85)
            elif defender == DefenderOption.WAKEUP_JUMP:
                # In corner, catches jump or resets; midscreen opponent might escape
                return 1200.0 if is_corner else -400.0
            elif defender == DefenderOption.WAKEUP_BACKDASH:
                return 1400.0 if is_corner else -200.0
            elif defender == DefenderOption.PERFECT_PARRY:
                # Timing variation makes frame 1 perfect parry whiff or become normal parry
                return 100.0

        return 0.0
