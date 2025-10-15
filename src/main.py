r"""Command-line two-player misère Nim.

Players alternate removing stones from a single pile. In misère Nim,
the player who takes the last remaining stone loses.
Configuration is read from config.yaml (YAML) where you also pick the play mode. Logging goes to run.log.

https://www.hackerrank.com/challenges/misere-nim-1/problem

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
# only for random initial state
import random
import sys
import yaml
import logging
from pathlib import Path

# Visible config and log paths
ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"
LOG_PATH = ROOT / "run.log"

from state import GameState, Player
from ai_agent import MisereNimAI


def load_config():
    """Load YAML config from CONFIG_PATH and return it as a dict.

    Exits the program with a message if the file is missing or malformed.
    """
    try:
        with open(CONFIG_PATH, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return config
    except FileNotFoundError:
        print(f"Error: missing config file at {CONFIG_PATH}")
        sys.exit(1)

def initialize_game():
    """Initialize a misère Nim game from config.

    Returns a tuple: (GameState, mode_description, ai_enabled, ai_depth).
    """
    config = load_config()

    piles_number = config.get("piles_number", 3)
    min_stones = config.get("min_stones_per_pile", 1)
    max_stones = config.get("max_stones_per_pile", 5)
    ai_enabled = bool(config.get("ai_play", False))
    try:
        ai_depth = int(config.get("ai_depth", 9))
    except (TypeError, ValueError):
        print("Error: ai_depth must be an integer")
        sys.exit(1)
    if ai_depth <= 0:
        print("Error: ai_depth must be > 0")
        sys.exit(1)

    if piles_number <= 0:
        print("Error: piles_number must be > 0")
        sys.exit(1)

    if min_stones >= max_stones:
        print("Error: min_stones_per_pile must be smaller than max_stones_per_pile")
        sys.exit(1)

    piles = [random.randint(min_stones, max_stones) for _ in range(piles_number)]

    mode_desc = "human vs AI (misère Nim)" if ai_enabled else "two-player (misère Nim)"
    current = Player.HUMAN

    state = GameState(piles=piles, current_player=current)

    # log config and initial state here so main doesn't need the config object
    logging.info("Loaded config: %s", {k: config.get(k) for k in ("piles_number", "min_stones_per_pile", "max_stones_per_pile")})
    logging.info("AI enabled: %s", ai_enabled)
    logging.info("AI depth: %d", ai_depth)
    logging.info("Mode: %s (last stone loses)", mode_desc)
    logging.info("Initial state: piles=%s current_player=%s", state.piles, state.current_player.name)

    return state, mode_desc, ai_enabled, ai_depth

def main():
    """Run a two-player misère Nim game in the terminal.

    Rules: players alternate removing 1+ stones from a single pile.
    Misère Nim: the player who takes the last stone loses.
    """
    # configure logging: truncate the log file at each run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", filename=str(LOG_PATH), filemode="w")
    logging.info("Starting new game run (misère Nim)")
    state, mode_desc, ai_enabled, ai_depth = initialize_game()
    print(f"Mode: {mode_desc}")
    print("Initial game state:")
    render_game(state)

    ai_player = MisereNimAI(depth=ai_depth) if ai_enabled else None

    # Human vs Human loop
    human_turn = 1  # 1 or 2
    while True:
        state.current_player = Player.HUMAN if human_turn == 1 else Player.AI
        player_label = "AI" if ai_enabled and human_turn == 2 else f"Player {human_turn}"
        print(f"\nCurrent player: {player_label}")
        if is_game_over(state):
            # Misère Nim: if no stones at start of a turn, the current player loses,
            # so the opponent is the winner.
            winner_label = "AI" if ai_enabled and human_turn == 1 else f"Player {3 - human_turn}"
            print(f"Game over. Winner: {winner_label}")
            logging.info("Game over. Winner: %s", winner_label)
            return

        if ai_enabled and human_turn == 2:
            pile, remove = ai_player.choose_move(state.piles)
            print(f"AI removes {remove} from pile {pile + 1}")
        else:
            try:
                pile, remove = get_human_move(state, player_label=f"Player {human_turn}")
            except KeyboardInterrupt:
                print("\nExiting.")
                return

        apply_move(state, pile, remove)
        logging.info("Player %d move: pile=%d remove=%d", human_turn, pile + 1, remove)

        # Show updated state
        render_game(state)

        # Switch player
        if human_turn == 1:
            human_turn = 2
        else:
            human_turn = 1


def render_game(s: GameState):
    """Pretty-print the piles with one 'o' per stone.

    Shows pile indices, the exact count, and a literal bar of 'o's where
    the number of 'o's equals the number of stones in that pile.
    """
    max_len = max(len(str(x)) for x in s.piles) if s.piles else 1
    for i, count in enumerate(s.piles, start=1):
        bar = "o" * count if count > 0 else "(empty)"
        print(f" Pile {i:>2}: {count:>{max_len}} | {bar}")


def is_game_over(s: GameState) -> bool:
    """Return True if all piles are empty, else False."""
    return all(x == 0 for x in s.piles)


def apply_move(s: GameState, pile_index: int, remove: int):
    """Apply a move: remove `remove` stones from pile at `pile_index`.

    Raises ValueError on invalid indices or remove counts.
    """
    if pile_index < 0 or pile_index >= len(s.piles):
        print("Invalid pile index.")
        raise ValueError("pile_index out of range")
    if s.piles[pile_index] <= 0:
        print("Selected pile is empty.")
        raise ValueError("empty pile")
    if remove <= 0 or remove > s.piles[pile_index]:
        print(f"You must remove between 1 and {s.piles[pile_index]} stones.")
        raise ValueError("invalid remove count")
    s.piles[pile_index] -= remove


def get_human_move(s: GameState, player_label: str = "Player"):
    """Read and validate a human move from stdin.

    Input format: "pile_index remove" (1-based pile index). Returns (pile, remove).
    Type 'q' to quit.
    """
    while True:
        raw = input(f"{player_label}, enter move as 'pile_index remove' (or 'q' to quit): ").strip()
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print("Exiting game.")
            sys.exit(0)
        parts = raw.split()
        if len(parts) != 2:
            print("Invalid format. Use: pile_index remove")
            continue
        try:
            pile = int(parts[0]) - 1
            remove = int(parts[1])
        except ValueError:
            print("Both pile and remove must be integers.")
            continue
        # Validate here; if invalid, loop will continue
        if pile < 0 or pile >= len(s.piles):
            print("Pile index out of range.")
            continue
        if s.piles[pile] == 0:
            print("Selected pile is empty.")
            continue
        if remove <= 0 or remove > s.piles[pile]:
            print(f"You must remove between 1 and {s.piles[pile]} stones.")
            continue
        return pile, remove


if __name__ == "__main__":
	main()