import math
from utilities import *


class Cell:
    def __init__(self, pos, grid_size=(0, 0), exit_pos=(0, 0), is_wall=False, is_border=False):
        """Create cell"""
        self._pos, self._grid_size, self._exit_pos, self._is_wall, self._is_border = pos, grid_size, exit_pos, is_wall, is_border
        self._sf, self._df, self._df_change = 0, 0, 0
        self._occupied, self.path = False, False
        if self._is_wall:
            self._occupied = True
        self._calculate_sf()

    def _calculate_sf(self):
        """Calculates sf value based on distance to the exit between 1 and 0"""
        distance_to_exit = math.sqrt((self._pos[0] - self._exit_pos[0]) ** 2 + (self._pos[1] - self._exit_pos[1]) ** 2)
        max_distance = math.sqrt((self._grid_size[0] - 1) ** 2 + (self._grid_size[1] - 1) ** 2)
        self._sf = 1 - (distance_to_exit / max_distance)

    def get_sf(self):
        """Accessor method"""
        return self._sf

    def get_df(self):
        """Accessor method"""
        return self._df

    def change_df(self, amount):
        """Add amount to df value of this cell"""
        self._df_change += amount

    def update_to_new_df(self):
        """Set df value to new clamped value"""
        self._df = clamp(self._df + self._df_change, 0, 1)
        self._df_change = 0

    def set_occupied(self, occupied):
        """Accessor method"""
        self._occupied = occupied

    def get_occupied_multiplier(self):
        """Returns 1 if occupied, 0 otherwise"""
        return int(self._occupied)

    def is_wall(self):
        """Accessor method"""
        return self._is_wall

    def is_border(self):
        """Accessor method"""
        return self._is_border

    def add_wall(self):
        """Set this cell to be a wall"""
        self._is_wall = True
        self._occupied = True

    def clear_wall(self):
        """Set this cell to no longer be a wall"""
        self._is_wall = False
        self._occupied = False
