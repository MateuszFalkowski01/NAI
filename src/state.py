"""State and player definitions for the Nim game (misère variant used).

This module exposes:
- Player: an Enum representing participants. The AI value is kept for
  compatibility with earlier versions but is currently unused.
- GameState: a dataclass holding the piles and the current player.
"""

from dataclasses import dataclass
from typing import List
from enum import Enum

class Player(Enum):
    """Players in the game.

    Attributes
    ----------
    HUMAN: Human player.
    AI: Dummy placeholder for AI player (not used in current version).  
    """
    HUMAN = 1
    AI = 2

@dataclass
class GameState:
    """Immutable-like container describing a Nim position.

    Fields
    ------
    piles: list of non-negative integers; piles[i] is stones in pile i.
    current_player: which player is to move next.
    """
    piles: List[int]
    current_player: Player
