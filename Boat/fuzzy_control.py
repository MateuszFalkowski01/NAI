"""Fuzzy controller for the boat simulator.

This module provides:
- create_fuzzy_system() -> ControlSystemSimulation
- compute_controls(sim, distance, speed, angle_deg) -> (engine_power, rudder)

Module contains the fuzzy variables, rule set and a small helper for
computing outputs from inputs.
"""
from typing import Tuple
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_system() -> ctrl.ControlSystemSimulation:
    """Create and return a configured skfuzzy ControlSystemSimulation.

    Returns
    -------
    ctrl.ControlSystemSimulation
        Simulation object ready to accept inputs: 'distance', 'speed',
        'target_angle'.
    """
    # 1. Variable Definitions (Inputs)
    distance = ctrl.Antecedent(np.arange(0, 2001, 1), 'distance')
    distance['close'] = fuzz.trimf(distance.universe, [0, 0, 15])
    distance['medium'] = fuzz.trimf(distance.universe, [15, 100, 200])
    distance['far'] = fuzz.trapmf(distance.universe, [150, 300, 2000, 2000])

    speed = ctrl.Antecedent(np.arange(-5, 11, 0.1), 'speed')
    speed['reverse'] = fuzz.trapmf(speed.universe, [-5, -5, -1, 0])
    speed['zero'] = fuzz.trimf(speed.universe, [-1, 0, 1])
    speed['forward'] = fuzz.trapmf(speed.universe, [0, 1, 10, 10])

    target_angle = ctrl.Antecedent(np.arange(0, 361, 1), 'target_angle')
    target_angle['left'] = fuzz.trapmf(target_angle.universe, [0, 0, 135, 180])
    target_angle['straight'] = fuzz.trimf(target_angle.universe, [150, 180, 210])
    target_angle['right'] = fuzz.trapmf(target_angle.universe, [180, 225, 360, 360])

    # 2. Variable Definitions (Outputs)
    engine_power = ctrl.Consequent(np.arange(-100, 101, 1), 'engine_power')
    engine_power['full_reverse'] = fuzz.trimf(engine_power.universe, [-100, -100, -50])
    engine_power['brake'] = fuzz.trimf(engine_power.universe, [-60, -30, -5])
    engine_power['zero'] = fuzz.trimf(engine_power.universe, [-40, 0, 40])
    engine_power['slow_forward'] = fuzz.trimf(engine_power.universe, [10, 30, 60])
    engine_power['full_forward'] = fuzz.trimf(engine_power.universe, [20, 80, 80])

    rudder = ctrl.Consequent(np.arange(-45, 46, 1), 'rudder')
    rudder['max_left'] = fuzz.trimf(rudder.universe, [-45, -45, -20])
    rudder['slight_left'] = fuzz.trimf(rudder.universe, [-30, -15, 0])
    rudder['zero'] = fuzz.trimf(rudder.universe, [-10, 0, 10])
    rudder['slight_right'] = fuzz.trimf(rudder.universe, [0, 15, 30])
    rudder['max_right'] = fuzz.trimf(rudder.universe, [20, 45, 45])

    # 3. Fuzzy Rules
    rule_s1 = ctrl.Rule(target_angle['left'], rudder['max_left'])
    rule_s2 = ctrl.Rule(target_angle['straight'], rudder['zero'])
    rule_s3 = ctrl.Rule(target_angle['right'], rudder['max_right'])
    rule_s4 = ctrl.Rule(distance['close'], rudder['zero'])

    rule_m1 = ctrl.Rule(distance['far'] & target_angle['straight'], engine_power['full_forward'])
    rule_m2 = ctrl.Rule(distance['far'] & (target_angle['left'] | target_angle['right']), engine_power['slow_forward'])
    rule_m3 = ctrl.Rule(distance['medium'] & target_angle['straight'], engine_power['slow_forward'])
    rule_m4 = ctrl.Rule(distance['close'] & speed['forward'], engine_power['full_reverse'])

    rule_m5a = ctrl.Rule(distance['close'] & speed['forward'] & target_angle['straight'], engine_power['brake'])
    rule_m5b = ctrl.Rule(distance['close'] & speed['reverse'] & target_angle['straight'], engine_power['slow_forward'])
    rule_m5c = ctrl.Rule(distance['close'] & speed['zero'] & target_angle['straight'], engine_power['zero'])
    rule_m5d = ctrl.Rule(distance['close'] & (target_angle['straight'] | target_angle['left'] | target_angle['right']), engine_power['zero'])

    rule_m6 = ctrl.Rule(distance['close'] & (target_angle['left'] | target_angle['right']), engine_power['brake'])
    rule_m7 = ctrl.Rule(speed['reverse'], engine_power['slow_forward'])
    rule_m8 = ctrl.Rule(distance['medium'] & (target_angle['left'] | target_angle['right']), engine_power['slow_forward'])
    rule_m9 = ctrl.Rule(distance['medium'] & speed['forward'], engine_power['brake'])

    rule_unstick = ctrl.Rule(distance['close'] & speed['zero'] & (target_angle['left'] | target_angle['right']), engine_power['slow_forward'])

    all_rules = [
        rule_s1, rule_s2, rule_s3, rule_s4,
        rule_m1, rule_m2, rule_m3, rule_m4, rule_m5a, rule_m5b, rule_m5c, rule_m5d,
        rule_m6, rule_m7, rule_m8, rule_m9,
        rule_unstick
    ]

    control_system = ctrl.ControlSystem(all_rules)
    simulator = ctrl.ControlSystemSimulation(control_system)

    return simulator


def compute_controls(sim: ctrl.ControlSystemSimulation, distance: float, speed: float, angle_deg: float) -> Tuple[float, float]:
    """Compute engine power and rudder from fuzzy simulation.

    Parameters
    ----------
    sim : ctrl.ControlSystemSimulation
        Fuzzy simulation object returned by create_fuzzy_system().
    distance : float
        Distance to the target (units as used by the simulator).
    speed : float
        Current forward speed of the boat.
    angle_deg : float
        Relative angle to the target in degrees ([-180, 180]).

    Returns
    -------
    (float, float)
        (engine_power, rudder_angle_degrees)
    """
    # Convert angle [-180,180] -> [0,360] measured from stern
    angle_from_stern = (angle_deg + 180.0) % 360.0
    sim.input['distance'] = float(distance)
    sim.input['speed'] = float(speed)
    sim.input['target_angle'] = float(angle_from_stern)
    sim.compute()
    power = float(sim.output.get('engine_power', 0.0))
    rudder = float(sim.output.get('rudder', 0.0))
    return power, rudder
