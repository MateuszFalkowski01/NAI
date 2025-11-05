"""Rendering utilities for the boat simulator.

This module provides simple pygame-based rendering helpers. It keeps
visuals minimal and intended for demonstration and debugging.
"""

import pygame
import math
from typing import Tuple


COLOR_WATER = (20, 90, 160)
COLOR_BOAT = (139, 69, 19)
COLOR_TARGET = (255, 0, 0)
COLOR_TEXT = (255, 255, 255)


def init_display(width: int, height: int, title: str = "Simulator"):
    """Initialize pygame display and return (screen, clock, font).

    Parameters
    ----------
    width, height : int
        Window size in pixels.
    title : str
        Window title.

    Returns
    -------
    (screen, clock, font)
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    return screen, clock, font


def draw(screen, font, boat, target_pos: Tuple[int, int], inputs: dict, outputs: dict):
    """Draw the scene: target, boat and debug text.

    Parameters
    ----------
    screen : pygame.Surface
    font : pygame.font.Font
    boat : Boat
        Instance providing position, heading and shape.
    target_pos : (int,int)
    inputs : dict
        Debug inputs (distance, speed, target_angle).
    outputs : dict
        Debug outputs (engine_power, rudder).
    """
    screen.fill(COLOR_WATER)

    # draw target
    pygame.draw.circle(screen, COLOR_TARGET, (int(target_pos[0]), int(target_pos[1])), 12)
    pygame.draw.circle(screen, (255, 255, 255), (int(target_pos[0]), int(target_pos[1])), 8, 2)

    # draw boat
    rotated = []
    for x, y in boat.shape:
        rx = x * math.cos(boat.angle_rad) - y * math.sin(boat.angle_rad)
        ry = x * math.sin(boat.angle_rad) + y * math.cos(boat.angle_rad)
        rotated.append((boat.x + rx, boat.y + ry))
    pygame.draw.polygon(screen, COLOR_BOAT, rotated)
    
    # Draw front sail
    sail1_left = (boat.x + (-6 * math.cos(boat.angle_rad) - (-4) * math.sin(boat.angle_rad)),
                 boat.y + (-6 * math.sin(boat.angle_rad) + (-4) * math.cos(boat.angle_rad)))
    sail1_right = (boat.x + (6 * math.cos(boat.angle_rad) - (-4) * math.sin(boat.angle_rad)),
                  boat.y + (6 * math.sin(boat.angle_rad) + (-4) * math.cos(boat.angle_rad)))
    pygame.draw.line(screen, (255, 255, 255), sail1_left, sail1_right, 2)

    # Draw rear sail
    sail2_left = (boat.x + (-5 * math.cos(boat.angle_rad) - (4) * math.sin(boat.angle_rad)),
                 boat.y + (-5 * math.sin(boat.angle_rad) + (4) * math.cos(boat.angle_rad)))
    sail2_right = (boat.x + (5 * math.cos(boat.angle_rad) - (4) * math.sin(boat.angle_rad)),
                  boat.y + (5 * math.sin(boat.angle_rad) + (4) * math.cos(boat.angle_rad)))
    pygame.draw.line(screen, (255, 255, 255), sail2_left, sail2_right, 2)

    # debug text
    lines = [
        f"X: {boat.x:.1f} Y: {boat.y:.1f}",
        f"Speed: {boat.speed:.2f}",
        f"Angle: {math.degrees(boat.angle_rad):.1f} deg",
        "--- INPUTS ---",
        f"Dist: {inputs.get('distance', 0):.1f}",
        f"AngleToTarget: {inputs.get('target_angle', 0):.1f}",
        "--- OUTPUTS ---",
        f"Motor: {outputs.get('engine_power', 0):.2f}",
        f"Rudder: {outputs.get('rudder', 0):.2f}",
    ]

    for i, line in enumerate(lines):
        surf = font.render(line, True, COLOR_TEXT)
        screen.blit(surf, (8, 8 + i * 18))

    pygame.display.flip()
