import math
from utilities import *


class Cell:
    def __init__(self, pos, grid_size=(0, 0), exit_pos=(0, 0), is_wall=False, is_border=False):
        self._pos = pos
        self._grid_size = grid_size
        self._exit_pos = exit_pos
        self._sf = 0
        self._df = 0
        self._df_change = 0
        self._occupied = False
        self._is_wall = is_wall
        if self._is_wall:
            self._occupied = True
        self._is_border = is_border
        self.path = False
        self._calculate_sf()

    def _calculate_sf(self):
        distance_to_exit = math.sqrt((self._pos[0] - self._exit_pos[0]) ** 2 + (self._pos[1] - self._exit_pos[1]) ** 2)
        max_distance = math.sqrt((self._grid_size[0] - 1) ** 2 + (self._grid_size[1] - 1) ** 2)
        self._sf = 1 - (distance_to_exit / max_distance)

    def get_sf(self):
        return self._sf

    def get_df(self):
        return self._df

    def change_df(self, amount):
        self._df_change += amount

    def update_to_new_df(self):
        self._df = clamp(self._df + self._df_change, 0, 1)
        self._df_change = 0

    def set_occupied(self, occupied):
        self._occupied = occupied

    def get_occupied_multiplier(self):
        return int(self._occupied)

    def is_wall(self):
        return self._is_wall

    def is_border(self):
        return self._is_border

    def add_wall(self):
        self._is_wall = True
        self._occupied = True

    def clear_wall(self):
        self._is_wall = False
        self._occupied = False
