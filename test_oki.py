"""
Unit tests for SF6 Oki, Frame Data, Meaty Calculator, and Nash Equilibrium Solver.
"""

import unittest
import numpy as np
from sfdojo.frame_data import MoveData, KnockdownState, MeatyCalculator
from sfdojo.payoff_matrix import OkiScenario, PayoffMatrixBuilder, AttackerOption, DefenderOption
from sfdojo.nash_solver import NashSolver
from sfdojo.oki_engine import OkiEngine


class TestSFDojo(unittest.TestCase):

    def test_meaty_calculation(self):
        # Forward throw in corner gives +23f advantage
        # Move: 2MP with startup 6, active 3. Total startup delay needed = 23 + 1 - 6 = 18 frames
        # If we whiff a move taking 18 frames (or 19 for active 2), check meaty
        move = MoveData(
            name="2MP", startup=6, active_frames=3, recovery=12,
            on_block=-1, on_hit=4, damage=600
        )
        kd = KnockdownState(name="Forward Throw", frame_advantage=23)

        # Frame kill 18f:
        # relative active frame 1 starts at 18 + 6 = 24. Wakeup is on frame 24.
        # So active frame 1 hits on wakeup frame 1!
        res1 = MeatyCalculator.calculate_meaty(kd, move, frame_kill_duration=18)
        self.assertTrue(res1.is_valid_meaty)
        self.assertEqual(res1.connecting_active_frame, 1)
        self.assertEqual(res1.advantage_on_block, -1)
        self.assertEqual(res1.advantage_on_hit, 4)

        # Frame kill 17f:
        # Move starts 1 frame earlier: 17 + 6 = 23 (frame before wakeup).
        # Active 1 hits on 23, active 2 hits on 24 (wakeup)!
        # So active frame 2 connects on wakeup!
        # Bonus advantage = 2 - 1 = +1
        res2 = MeatyCalculator.calculate_meaty(kd, move, frame_kill_duration=17)
        self.assertTrue(res2.is_valid_meaty)
        self.assertEqual(res2.connecting_active_frame, 2)
        self.assertEqual(res2.advantage_on_block, 0)
        self.assertEqual(res2.advantage_on_hit, 5)

        # Frame kill 16f:
        # Active 3 hits on wakeup frame 24!
        # Bonus advantage = +2
        res3 = MeatyCalculator.calculate_meaty(kd, move, frame_kill_duration=16)
        self.assertTrue(res3.is_valid_meaty)
        self.assertEqual(res3.connecting_active_frame, 3)
        self.assertEqual(res3.advantage_on_block, 1)  # -1 + 2 = +1 on block!
        self.assertEqual(res3.advantage_on_hit, 6)

    def test_frame_kill_finder(self):
        move = MoveData(
            name="2MP", startup=6, active_frames=3, recovery=12,
            on_block=-1, on_hit=4, damage=600
        )
        whiff_moves = {"whiff_2LP": 12, "whiff_5LP": 13, "dash": 19}
        # Kd 35f (e.g. Heavy Tatsu corner)
        # Required kill for active frame 1: (35 + 1) - (6 + 0) = 30
        # Whiff 2LP (12) + dash (19) = 31 (for active 2: (35+1) - (6+1) = 29)
        kills = MeatyCalculator.find_frame_kills(
            knockdown_advantage=37, target_move=move, whiff_moves=whiff_moves, target_active_frame=1
        )
        # Required kill: 38 - 6 = 32. 13 + 19 = 32!
        self.assertTrue(any("whiff_5LP" in seq and "dash" in seq for seq, _ in kills))

    def test_nash_equilibrium_properties(self):
        # Rock Paper Scissors check
        rps_matrix = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ], dtype=float)
        eq = NashSolver.solve_zero_sum(rps_matrix, ["R", "P", "S"], ["R", "P", "S"])
        self.assertAlmostEqual(eq.game_value, 0.0, places=4)
        for val in eq.attacker_strategy.values():
            self.assertAlmostEqual(val, 1/3, places=3)
        for val in eq.defender_strategy.values():
            self.assertAlmostEqual(val, 1/3, places=3)

    def test_sf6_oki_equilibrium(self):
        scenario = OkiScenario(position="corner", defender_in_burnout=False)
        eq, matrix, att_labels, def_labels = OkiEngine.solve_oki_game(scenario)

        # In the corner, Attacker has positive expected value (advantage)
        self.assertGreater(eq.game_value, 0.0)

        # Probabilities sum to 1.0
        self.assertAlmostEqual(sum(eq.attacker_strategy.values()), 1.0, places=5)
        self.assertAlmostEqual(sum(eq.defender_strategy.values()), 1.0, places=5)

        # In corner, Shimmy, Strike, and Throw should have non-zero presence
        self.assertGreater(eq.attacker_strategy[AttackerOption.MEATY_STRIKE.value], 0.0)
        self.assertGreater(eq.attacker_strategy[AttackerOption.MEATY_THROW.value], 0.0)
        self.assertGreater(eq.attacker_strategy[AttackerOption.SHIMMY.value], 0.0)

    def test_burnout_scenario(self):
        # When defender is in burnout (no OD DP, no Parry)
        scenario = OkiScenario(position="corner", defender_in_burnout=True, defender_has_super=False)
        eq, matrix, att_labels, def_labels = OkiEngine.solve_oki_game(scenario)

        # OD DP and Parry should not even be in defender options
        self.assertNotIn(DefenderOption.INVINCIBLE_REVERSAL.value, eq.defender_strategy)
        self.assertNotIn(DefenderOption.PERFECT_PARRY.value, eq.defender_strategy)

        # Value of game should be significantly higher for attacker
        self.assertGreater(eq.game_value, 300.0)


if __name__ == "__main__":
    unittest.main()
