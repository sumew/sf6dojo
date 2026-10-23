"""
Linear programming solver for two-player zero-sum Nash Equilibrium
and exploitative best-response analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linprog


@dataclass
class EquilibriumResult:
    game_value: float
    attacker_strategy: Dict[str, float]
    defender_strategy: Dict[str, float]
    attacker_expected_payoffs: Dict[str, float]
    defender_expected_payoffs: Dict[str, float]


class NashSolver:
    """Solves for mixed-strategy Nash Equilibrium in zero-sum matrix games."""

    @staticmethod
    def solve_zero_sum(
        payoff_matrix: np.ndarray,
        attacker_labels: List[str],
        defender_labels: List[str]
    ) -> EquilibriumResult:
        """
        Solves for the optimal minimax mixed strategies for both players.
        payoff_matrix: Shape (m, n) where m = num attacker actions, n = num defender actions.
        """
        m, n = payoff_matrix.shape

        # --- 1. Attacker (Maximizer) LP ---
        # Variables: [p_0, p_1, ..., p_{m-1}, v]
        # Maximize v  <=>  Minimize -v
        # c = [0, ..., 0, -1] of length m + 1
        c_att = np.zeros(m + 1)
        c_att[-1] = -1.0

        # Constraints: for each defender action j:
        #   sum_i (M_ij * p_i) >= v  <=>  - sum_i (M_ij * p_i) + v <= 0
        A_ub_att = np.zeros((n, m + 1))
        for j in range(n):
            A_ub_att[j, :m] = -payoff_matrix[:, j]
            A_ub_att[j, -1] = 1.0
        b_ub_att = np.zeros(n)

        # Equality constraint: sum_i p_i = 1
        A_eq_att = np.zeros((1, m + 1))
        A_eq_att[0, :m] = 1.0
        b_eq_att = np.array([1.0])

        # Bounds: p_i >= 0, v is unconstrained (-inf, inf)
        bounds_att = [(0.0, 1.0) for _ in range(m)] + [(None, None)]

        res_att = linprog(
            c_att,
            A_ub=A_ub_att,
            b_ub=b_ub_att,
            A_eq=A_eq_att,
            b_eq=b_eq_att,
            bounds=bounds_att,
            method="highs"
        )

        if not res_att.success:
            raise RuntimeError(f"Attacker LP failed to converge: {res_att.message}")

        p_opt = np.maximum(0.0, res_att.x[:m])
        p_opt = p_opt / np.sum(p_opt)  # Normalize for numerical precision
        game_val = float(res_att.x[-1])

        # --- 2. Defender (Minimizer) LP ---
        # Variables: [q_0, q_1, ..., q_{n-1}, u]
        # Minimize u
        # c = [0, ..., 0, 1] of length n + 1
        c_def = np.zeros(n + 1)
        c_def[-1] = 1.0

        # Constraints: for each attacker action i:
        #   sum_j (M_ij * q_j) <= u  <=>  sum_j (M_ij * q_j) - u <= 0
        A_ub_def = np.zeros((m, n + 1))
        for i in range(m):
            A_ub_def[i, :n] = payoff_matrix[i, :]
            A_ub_def[i, -1] = -1.0
        b_ub_def = np.zeros(m)

        # Equality constraint: sum_j q_j = 1
        A_eq_def = np.zeros((1, n + 1))
        A_eq_def[0, :n] = 1.0
        b_eq_def = np.array([1.0])

        bounds_def = [(0.0, 1.0) for _ in range(n)] + [(None, None)]

        res_def = linprog(
            c_def,
            A_ub=A_ub_def,
            b_ub=b_ub_def,
            A_eq=A_eq_def,
            b_eq=b_eq_def,
            bounds=bounds_def,
            method="highs"
        )

        if not res_def.success:
            raise RuntimeError(f"Defender LP failed to converge: {res_def.message}")

        q_opt = np.maximum(0.0, res_def.x[:n])
        q_opt = q_opt / np.sum(q_opt)

        # Expected payoffs for each pure strategy
        att_ev_per_action = payoff_matrix @ q_opt
        def_ev_per_action = p_opt @ payoff_matrix

        attacker_strat = {attacker_labels[i]: float(p_opt[i]) for i in range(m)}
        defender_strat = {defender_labels[j]: float(q_opt[j]) for j in range(n)}
        att_evs = {attacker_labels[i]: float(att_ev_per_action[i]) for i in range(m)}
        def_evs = {defender_labels[j]: float(def_ev_per_action[j]) for j in range(n)}

        return EquilibriumResult(
            game_value=game_val,
            attacker_strategy=attacker_strat,
            defender_strategy=defender_strat,
            attacker_expected_payoffs=att_evs,
            defender_expected_payoffs=def_evs
        )

    @staticmethod
    def calculate_best_response(
        payoff_matrix: np.ndarray,
        attacker_labels: List[str],
        opponent_distribution: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Given a non-optimal / observed opponent distribution q,
        calculate the Attacker's expected payoff for each option and best response.
        """
        evs = payoff_matrix @ opponent_distribution
        ev_dict = {attacker_labels[i]: float(evs[i]) for i in range(len(attacker_labels))}
        best_idx = int(np.argmax(evs))
        best_action = attacker_labels[best_idx]
        best_ev = float(evs[best_idx])
        return best_action, best_ev, ev_dict
