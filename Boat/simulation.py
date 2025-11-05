"""Simulation helpers for the boat simulator.

This module provides a minimal 2D physics model for a boat and a
helper to generate random start/target positions for demonstrations.

"""

import math
import random


class Boat:
    """Lightweight boat physics for the simplified simulation.

    Attributes
    ----------
    x, y : float
        Position of the boat in screen coordinates.
    angle_rad : float
        Heading in radians.
    speed : float
        Current forward speed.

    Methods
    -------
    get_inputs_for_fuzzy(target_x, target_y) -> dict
        Returns the fuzzy inputs: distance, speed, target_angle.
    update_physics(engine_power, rudder_rad)
        Update position/heading from control inputs.
    """
    def __init__(self, x: float, y: float, angle_deg: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.angle_rad = math.radians(angle_deg)
        self.speed = 0.0
        self.max_speed = 5.0
        self.max_reverse_speed = -5.0

        # Boat shape used by renderer
        self.shape = [
            (0, -16),
            (6, -8),
            (8, 0),
            (6, 8),
            (4, 12),
            (-4, 12),
            (-6, 8),
            (-8, 0),
            (-6, -8),
        ]

    def get_inputs_for_fuzzy(self, target_x: float, target_y: float) -> dict:
        """Compute and return inputs for the fuzzy controller.

        Returns a dict with keys: 'distance', 'speed', 'target_angle'
        where 'target_angle' is in degrees in range [-180,180].
        """
        vx = target_x - self.x
        vy = target_y - self.y
        distance = math.hypot(vx, vy)

        # angle from boat heading to target, normalized to degrees in [-180,180]
        target_angle = math.atan2(vy, vx)
        # convert to same reference as original: 0 = up screen
        target_angle_pygame = target_angle + math.pi / 2.0
        delta = target_angle_pygame - self.angle_rad
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        delta_deg = math.degrees(delta)

        return {
            "distance": distance,
            "speed": self.speed,
            "target_angle": delta_deg,
        }
 
    def update_physics(self, engine_power: float, rudder_rad: float) -> None:
        """Update the boat physics using the controller outputs.

        Parameters
        ----------
        engine_power : float
            Control signal for engine power (range roughly -100..100).
        rudder_rad : float
            Rudder angle in radians (signed).
        """
        # acceleration proportional to engine power
        acc = float(engine_power) / 1000.0
        self.speed += acc
        # simple drag
        self.speed *= 0.985

        # clamp speeds
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < self.max_reverse_speed:
            self.speed = self.max_reverse_speed

        # steering
        if abs(self.speed) > 0.05:
            steer_effect = max(abs(self.speed) / self.max_speed, 0.05)
            self.angle_rad += rudder_rad * steer_effect * 0.18

        # normalize angle
        self.angle_rad = (self.angle_rad + math.pi) % (2 * math.pi) - math.pi

        # move
        self.x += math.sin(self.angle_rad) * self.speed
        self.y -= math.cos(self.angle_rad) * self.speed


def random_positions(screen_w: int, screen_h: int):
    """Return random (target_pos, start_pos, start_angle).

    Used by the demo to spawn random scenarios.
    """
    tgt_x = random.randint(80, screen_w - 80)
    tgt_y = random.randint(80, screen_h // 2)
    start_x = random.randint(80, screen_w - 80)
    start_y = random.randint(screen_h - 180, screen_h - 60)
    start_angle = random.uniform(-180.0, 180.0)
    return (tgt_x, tgt_y), (start_x, start_y), start_angle
