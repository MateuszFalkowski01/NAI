"""Adversarial-search misère Nim agent built on easyAI."""
from __future__ import annotations

from typing import Sequence, Tuple

from easyAI import Negamax, TwoPlayerGame


def _is_current_position_winning(piles: Sequence[int]) -> bool:
    """Return True if the player to move has a winning misère Nim position."""
    non_empty = [stones for stones in piles if stones > 0]
    if not non_empty:
        return True
    if all(stones == 1 for stones in non_empty):
        return len(non_empty) % 2 == 0
    nim_sum = 0
    for stones in non_empty:
        nim_sum ^= stones
    return nim_sum != 0


class _MisereNimGame(TwoPlayerGame):
    """Minimal adapter exposing the current Nim position to easyAI."""

    def __init__(self, piles: Sequence[int]):
        self.players = [None, None]
        self.current_player = 1
        self.piles = [max(0, int(stones)) for stones in piles]

    def possible_moves(self):
        return [
            (index, remove)
            for index, stones in enumerate(self.piles)
            if stones > 0
            for remove in range(1, stones + 1)
        ]

    def make_move(self, move):
        pile_index, remove = move
        self.piles[pile_index] -= remove

    def unmake_move(self, move):
        pile_index, remove = move
        self.piles[pile_index] += remove

    def is_over(self):
        return all(stones == 0 for stones in self.piles)

    def scoring(self):
        return 100 if _is_current_position_winning(self.piles) else -100


class MisereNimAI:
    """Convenience wrapper that chooses moves via Negamax."""

    def __init__(self, depth: int = 9) -> None:
        if depth < 1:
            raise ValueError("depth must be positive")
        self._engine = Negamax(depth)

    def choose_move(self, piles: Sequence[int]) -> Tuple[int, int]:
        game = _MisereNimGame(piles)
        move = self._engine(game)
        if move is None:
            raise RuntimeError("AI could not find a legal move")
        pile_index, remove = move
        return int(pile_index), int(remove)
