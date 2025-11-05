"""Fuzzy Boat Simulator

Problem
-------
This program uses a fuzzy logic controller to autonomously steer a
simulated boat so it navigates toward a docking target and stops
inside the docking area. The fuzzy controller consumes three inputs
(distance, speed, relative target angle) and produces two outputs
(engine power and rudder angle) which are applied to a lightweight
physics model.

Authors
-------
Krzysztof Cieślik s27115
Mateusz Falkowski s27426

run instructions:
     Python: Works with Python 3.12.x
     Tested on Linux 6.14.0-33-generic
     Linux/macOS:
        1) Create venv
            python3 -m venv venv && source venv/bin/activate
        2) Install dependencies
            pip install -r requirements.txt
        3) Run
            python src/main.py

     Windows:
        1) Create venv
            py -m venv venv
            .\\venv\\Scripts\\Activate.ps1
        2) Install dependencies
            pip install -r requirements.txt
        3) Run
            python .\\src\\main.py
"""

from simulation import Boat, random_positions
from fuzzy_control import create_fuzzy_system, compute_controls
from render import init_display, draw
import math
import time

SCREEN_WIDTH = 960
SCREEN_HEIGHT = 600


def main():
    """Main runtime loop: initialize subsystems and run simulation.

    The top-level module docstring contains problem description, authors
    and setup instructions (per project requirement). This function
    wires together the simulation, fuzzy controller and renderer.
    """
    screen, clock, font = init_display(SCREEN_WIDTH, SCREEN_HEIGHT, "Fuzzy Boat - Simplified")

    fuzzy_sim = create_fuzzy_system()

    target_pos, start_pos, start_angle = random_positions(SCREEN_WIDTH, SCREEN_HEIGHT)
    boat = Boat(start_pos[0], start_pos[1], start_angle)

    running = True
    while running:
        # event handling
        for ev in __import__('pygame').event.get():
            if ev.type == __import__('pygame').QUIT:
                running = False
            if ev.type == __import__('pygame').KEYDOWN:
                if ev.key == __import__('pygame').K_ESCAPE:
                    running = False
                if ev.key == __import__('pygame').K_r:
                    target_pos, start_pos, start_angle = random_positions(SCREEN_WIDTH, SCREEN_HEIGHT)
                    boat = Boat(start_pos[0], start_pos[1], start_angle)

        # sense
        inputs = boat.get_inputs_for_fuzzy(target_pos[0], target_pos[1])

        # compute fuzzy controls
        try:
            power, rudder_deg = compute_controls(fuzzy_sim, inputs['distance'], inputs['speed'], inputs['target_angle'])
            rudder_rad = math.radians(rudder_deg)
        except Exception:
            power, rudder_rad = 0.0, 0.0

        outputs = {'engine_power': power, 'rudder': rudder_deg}

        # apply physics
        boat.update_physics(power, rudder_rad)

        # render
        draw(screen, font, boat, target_pos, inputs, outputs)

        # docking check
        if inputs.get('distance', 999) < 15 and abs(boat.speed) < 0.12:
            print("Docked!")
            time.sleep(0.6)
            target_pos, start_pos, start_angle = random_positions(SCREEN_WIDTH, SCREEN_HEIGHT)
            boat = Boat(start_pos[0], start_pos[1], start_angle)

        clock.tick(60)

    # cleanup
    __import__('pygame').quit()


if __name__ == "__main__":
    main()
