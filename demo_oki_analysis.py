#!/usr/bin/env python3
"""
Comprehensive SF6 Oki, Frame Data & Game Theory Simulation Runner.
Demonstrates:
1. Meaty timing formulas and auto-timed frame kill setups.
2. Zero-sum payoff matrix generation across positions (Corner vs Midscreen, Burnout).
3. Mixed-strategy Nash Equilibrium solving.
4. Exploitative Best Response analysis for common human playstyles.
"""

import sys
from sfdojo.frame_data import MeatyCalculator
from sfdojo.payoff_matrix import OkiScenario, AttackerOption, DefenderOption
from sfdojo.oki_engine import OkiEngine


def print_separator(title: str = "", char: str = "="):
    if title:
        print(f"\n{char * 5} {title} {char * (75 - len(title))}")
    else:
        print(char * 80)


def demo_meaty_timings():
    print_separator("PART 1: DETERMINISTIC MEATY TIMINGS & FRAME KILLS")
    presets = OkiEngine.get_character_presets()
    ken = presets["Ken"]

    print("\n[Scenario]: Ken in Corner after Forward Throw (+23 Advantage)")
    kd_corner = ken["knockdowns"]["Forward Throw (Corner)"]
    setups = OkiEngine.evaluate_meaty_setups(kd_corner, ken["moves"], ken["whiff_durations"])

    print(f"{'Move':<10} | {'Active Hit':<10} | {'Frame Kill Sequence':<25} | {'Adv (Block)':<12} | {'Adv (Hit)':<10}")
    print("-" * 75)
    for s in setups:
        seq_str = " + ".join(s["frame_kill_sequence"]) if s["frame_kill_sequence"] else "Manual Timing"
        print(f"{s['move']:<10} | {s['target_active_frame']:<10} | {seq_str:<25} | {s['advantage_on_block']:<+12} | {s['advantage_on_hit']:<+10}")

    print("\n[Scenario]: Ken after Heavy Tatsu Knockdown (+35 Advantage)")
    kd_tatsu = ken["knockdowns"]["Heavy Tatsu (Corner)"]
    tatsu_setups = OkiEngine.evaluate_meaty_setups(kd_tatsu, ken["moves"], ken["whiff_durations"])

    print(f"{'Move':<10} | {'Active Hit':<10} | {'Frame Kill Sequence':<25} | {'Adv (Block)':<12} | {'Adv (Hit)':<10}")
    print("-" * 75)
    for s in tatsu_setups[:8]:
        seq_str = " + ".join(s["frame_kill_sequence"]) if s["frame_kill_sequence"] else "Manual Timing"
        print(f"{s['move']:<10} | {s['target_active_frame']:<10} | {seq_str:<25} | {s['advantage_on_block']:<+12} | {s['advantage_on_hit']:<+10}")


def demo_nash_equilibria():
    print_separator("PART 2: GAME THEORY & NASH EQUILIBRIUM DISTRIBUTIONS")

    scenarios = [
        ("Midscreen (Both with Meter)", OkiScenario(position="midscreen")),
        ("Corner (Both with Meter)", OkiScenario(position="corner")),
        ("Corner (Defender in Burnout, No Meter)", OkiScenario(position="corner", defender_in_burnout=True, defender_has_super=False))
    ]

    for title, sc in scenarios:
        eq, matrix, att_labels, def_labels = OkiEngine.solve_oki_game(sc)
        print(f"\n--- {title} ---")
        print(f"Expected Game Value (Attacker Net EV): {eq.game_value:+.2f} HP per oki knockdown\n")

        print("Attacker Optimal Strategy Distribution:")
        for action, prob in eq.attacker_strategy.items():
            bar = "#" * int(prob * 40)
            print(f"  {action:<22} : {prob * 100:5.1f}%  {bar}")

        print("\nDefender Optimal Strategy Distribution:")
        for action, prob in eq.defender_strategy.items():
            bar = "#" * int(prob * 40)
            print(f"  {action:<22} : {prob * 100:5.1f}%  {bar}")


def demo_exploitative_responses():
    print_separator("PART 3: EXPLOITATIVE BEST RESPONSE (ADAPTING TO OPPONENT BIAS)")
    corner_scenario = OkiScenario(position="corner")

    playstyles = [
        ("Ranked 'Delay Tech' Habit (Common Platinum/Diamond)", {
            "Delay Tech": 0.55,
            "Block": 0.25,
            "OD DP / Reversal": 0.10,
            "Wakeup Mash (4f)": 0.10
        }),
        ("Impatient Masher (Wakeup 4f Button)", {
            "Wakeup Mash (4f)": 0.60,
            "Block": 0.20,
            "OD DP / Reversal": 0.20
        }),
        ("DP Gambler (Wakeup OD DP / Level 3)", {
            "OD DP / Reversal": 0.50,
            "Block": 0.30,
            "Wakeup Mash (4f)": 0.20
        }),
        ("Parry Heavy Defense", {
            "Perfect Parry": 0.50,
            "Block": 0.30,
            "Delay Tech": 0.20
        })
    ]

    for name, bias in playstyles:
        best_act, best_ev, evs = OkiEngine.simulate_exploitative_response(corner_scenario, bias)
        print(f"\nOpponent Profile: {name}")
        print("  Observed tendencies:", ", ".join(f"{k}: {v*100:.0f}%" for k, v in bias.items()))
        print(f"  => Optimal Counter: [{best_act}] (Expected Payoff: {best_ev:+.1f} damage)")
        print("  Payoffs for all attacker choices:")
        for act, val in sorted(evs.items(), key=lambda x: x[1], reverse=True):
            print(f"     {act:<22}: {val:+.1f}")


if __name__ == "__main__":
    demo_meaty_timings()
    demo_nash_equilibria()
    demo_exploitative_responses()
    print_separator("SIMULATION COMPLETE")
