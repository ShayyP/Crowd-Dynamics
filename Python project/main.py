import pygame as pg
from agents import Agent
from cells import Cell
from utilities import *
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time


class SpatialDynamics:
    def __init__(self, grid_size, cell_size, exit_pos, exit_capacity, cost_of_congestion, df_diffuse_rate, df_increase, df_strength, sf_strength, show_probs=True):
        # Cost of conflict
        self._c = cost_of_congestion
        self._exit_pos = (exit_pos[0] + 1, exit_pos[1] + 1)
        self._exit_capacity = exit_capacity
        self._agents = []
        self._grid_size = (grid_size[0] + 2, grid_size[1] + 2)
        self._cell_size = cell_size
        self._df_diffuse_rate = df_diffuse_rate
        self._df_increase = df_increase
        self._df_strength = df_strength
        self._sf_strength = sf_strength
        self._window = None
        self._time = 0
        self._patient_distribution = []
        self._impatient_distribution = []
        self._neutral_distribution = []
        self._grid = []
        self._mouse_button_down = None
        self._show_probs = show_probs
        self._auto_run = False
        self._running = False
        self._move = False
        self._auto_run_delay = 0.5
        self._create_grid()

    def fill_grid_random(self, patient_weight, impatient_weight, neutral_weight):
        """Fills the grid with agents, random but weighted"""
        for x in range(self._grid_size[0]):
            for y in range(self._grid_size[1]):
                if (x, y) != self._exit_pos:
                    choice = random.choices(['p', 'i', 'n', None], [patient_weight, impatient_weight, neutral_weight, 1 - patient_weight - impatient_weight - neutral_weight])[0]
                    if choice is not None:
                        self._add_agent((x, y), choice)

    def _create_grid(self):
        """Creates the grid, 2D array of cell instances"""
        for y in range(self._grid_size[1]):
            row = []
            for x in range(self._grid_size[0]):
                if x == 0 or y == 0 or x == self._grid_size[0] - 1 or y == self._grid_size[1] - 1:
                    row.append(Cell((x, y), self._grid_size, self._exit_pos, True, True))
                else:
                    row.append(Cell((x, y), self._grid_size, self._exit_pos))
            self._grid.append(row)

    def _add_agent(self, pos, strategy):
        """Adds an agent to the grid if it is within the bounds and there is nothing at that position already"""
        if strategy in ['p', 'i', 'n'] and 0 <= pos[0] <= self._grid_size[0] - 1 and 0 <= pos[1] <= self._grid_size[1] - 1:
            space_taken = self._grid[pos[1]][pos[0]].get_occupied_multiplier()
            if not space_taken:
                self._agents.append(Agent(pos, strategy))
                self._grid[pos[1]][pos[0]].set_occupied(True)
                return True
            if not self._grid[pos[1]][pos[0]].is_wall():
                for i in range(len(self._agents)):
                    if self._agents[i].get_pos() == pos:
                        self._agents[i].toggle_strategy()
                        return True
        return False

    def _add_wall(self, pos):
        """Adds a wall to the grid if it is within the bounds and there is nothing at that position already"""
        if 0 <= pos[0] <= self._grid_size[0]-1 and 0 <= pos[1] <= self._grid_size[1]-1:
            space_taken = self._grid[pos[1]][pos[0]].get_occupied_multiplier()
            if not space_taken:
                self._grid[pos[1]][pos[0]].add_wall()
                self._invalidate_paths()
                return True
        return False

    def _update_agent_strategies(self):
        """Updates all agent's strategies"""
        for agent in self._agents:
            agent.update_distance_to_exit(self._exit_pos)
        for agent in self._agents:
            agent.update_t_i(self._agents, self._exit_capacity)
        for agent in self._agents:
            agent.update_strategy(self._c, self._agents)
        patient_agents = 0
        impatient_agents = 0
        neutral_agents = 0
        for agent in self._agents:
            agent.move_to_new_strategy()
            if agent.get_strategy() == 'p':
                patient_agents += 1
            elif agent.get_strategy() == 'i':
                impatient_agents += 1
            else:
                neutral_agents += 1

        self._time += 1
        self._patient_distribution.append(patient_agents / (patient_agents + impatient_agents + neutral_agents))
        self._impatient_distribution.append(impatient_agents / (patient_agents + impatient_agents + neutral_agents))
        self._neutral_distribution.append(neutral_agents / (patient_agents + impatient_agents + neutral_agents))

    def _move_agents(self):
        """Moves agents, choices are made using sf and df values of each cell, impatient agents take priority"""
        moves = []
        # Calculate probability to move to each neighbouring cell
        for agent in self._agents:
            agent_moves = []
            agent_probabilities = []
            agent_x = agent.get_pos()[0]
            agent_y = agent.get_pos()[1]
            for x in range(-1, 2):
                for y in range(-1, 2):
                    move = (agent_x + x, agent_y + y)
                    cell = self._grid[move[1]][move[0]]
                    sf_multiplier = 1
                    df_multiplier = 1
                    if agent.get_strategy() == 'i':
                        sf_multiplier = 10
                    elif agent.get_strategy() == 'n':
                        df_multiplier = 10
                    probability = math.pow(math.e, cell.get_df() * self._df_strength * df_multiplier) * math.pow(math.e, (cell.get_sf() * self._sf_strength * sf_multiplier)) * (1 - cell.get_occupied_multiplier())
                    if move in agent.get_route_taken():
                        probability *= agent.get_deterrent(move)
                    agent_moves.append(move)
                    agent_probabilities.append(probability)

            # Normalise the probabilities:
            sum_prob = 0
            for prob in agent_probabilities:
                sum_prob += prob
            if sum_prob != 0:
                for i in range(len(agent_probabilities)):
                    agent_probabilities[i] /= sum_prob
                # Choose a move
                chosen_move = random.choices(agent_moves, agent_probabilities)[0]
                if chosen_move is not None:
                    move_contested = False
                    for move in range(0, len(moves)):
                        if chosen_move in moves[move]:
                            moves[move].append(agent)
                            move_contested = True
                    if not move_contested:
                        moves.append([chosen_move, agent])

        for move in moves:
            target = move[0]
            space_available = 1
            if target == self._exit_pos:
                space_available = self._exit_capacity
            agents_to_choose_from = move[1:]
            if len(agents_to_choose_from) <= space_available:
                for agent in agents_to_choose_from:
                    self._move_agent(agent, target)
            else:
                impatient_agents = []
                patient_agents = []
                neutral_agents = []
                for agent in agents_to_choose_from:
                    if agent.get_strategy() == 'i':
                        impatient_agents.append(agent)
                    elif agent.get_strategy() == 'p':
                        patient_agents.append(agent)
                    else:
                        neutral_agents.append(agent)

                while len(impatient_agents) > 0 and space_available > 0:
                    chosen_agent = random.choice(impatient_agents)
                    self._move_agent(chosen_agent, target)
                    impatient_agents.remove(chosen_agent)
                    space_available -= 1

                while len(patient_agents) > 0 and space_available > 0:
                    chosen_agent = random.choice(patient_agents)
                    self._move_agent(chosen_agent, target)
                    patient_agents.remove(chosen_agent)
                    space_available -= 1

                while len(neutral_agents) > 0 and space_available > 0:
                    chosen_agent = random.choice(neutral_agents)
                    self._move_agent(chosen_agent, target)
                    neutral_agents.remove(chosen_agent)
                    space_available -= 1

    def _move_agent(self, agent, pos):
        """Moves an agent to a specified position"""
        if pos == self._exit_pos:
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].set_occupied(False)
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].change_df(self._df_increase)
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].update_to_new_df()
            current_multiplier = 1
            length = len(agent.get_route_taken())
            for grid_pos in agent.get_route_taken():
                self._grid[grid_pos[1]][grid_pos[0]].change_df(self._df_increase * (current_multiplier / length))
                self._grid[grid_pos[1]][grid_pos[0]].update_to_new_df()
                current_multiplier += 1
            self._agents.remove(agent)
            print(f'Agent at: {agent.get_pos()} has left through the exit.')
        else:
            old_pos = agent.get_pos()
            agent.move(pos)
            print(f'Agent at: {old_pos} has moved to {agent.get_pos()}.')
            self._grid[old_pos[1]][old_pos[0]].set_occupied(False)
            # self._grid[old_pos[1]][old_pos[0]].change_df(self._df_increase)
            # self._grid[old_pos[1]][old_pos[0]].update_to_new_df()
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].set_occupied(True)

    def _diffuse_df(self):
        """Diffuses df values for each cell to neighbouring cells"""
        for x in range(self._grid_size[0]):
            for y in range(self._grid_size[1]):
                if not self._grid[y][x].is_wall():
                    neighbours = []
                    for x2 in range(-1, 2):
                        for y2 in range(-1, 2):
                            if not (x2 == 0 and y2 == 0):
                                neighbours.append((x + x2, y + y2))
                    diffuse_rate = self._df_diffuse_rate
                    if self._grid[y][x].get_df() <= self._df_diffuse_rate:
                        diffuse_rate = self._grid[y][x].get_df()
                    self._grid[y][x].change_df(-diffuse_rate)
                    diffuse_spread = diffuse_rate / len(neighbours)
                    for neighbour in neighbours:
                        self._grid[neighbour[1]][neighbour[0]].change_df(diffuse_spread)

        for x in range(self._grid_size[0]):
            for y in range(self._grid_size[1]):
                self._grid[y][x].update_to_new_df()

    def _draw_grid(self):
        """Draws everything to the window, to be called every frame"""
        self._window.fill((255, 255, 255))

        for x in range(0, self._grid_size[0]):
            for y in range(0, self._grid_size[1]):
                cell = self._grid[y][x]
                if self._show_probs:
                    if cell.is_wall():
                        colour = (255, 255, 255)
                    else:
                        multiplier = clamp(cell.get_df() + cell.get_sf(), 0, 1)
                        colour = (255 * multiplier, 0, 255 * multiplier)
                else:
                    if cell.is_wall():
                        colour = (0, 0, 0)
                    else:
                        colour = (255, 255, 255)
                pg.draw.rect(self._window, colour, (x * self._cell_size, y * self._cell_size, self._cell_size, self._cell_size))
        pg.draw.rect(self._window, (0, 255, 0), (self._exit_pos[0] * self._cell_size, self._exit_pos[1] * self._cell_size, self._cell_size, self._cell_size))

        for x in range(1, self._grid_size[0]):
            pg.draw.line(self._window, (0, 0, 0), (x * self._cell_size, 0), (x * self._cell_size, self._grid_size[0] * self._cell_size), 1)
        for y in range(1, self._grid_size[1]):
            pg.draw.line(self._window, (0, 0, 0), (0, y * self._cell_size), (self._grid_size[1] * self._cell_size, y * self._cell_size), 1)

        mouse_pos = pg.mouse.get_pos()
        highlighted_path = None
        for agent in self._agents:
            if (agent.get_pos()[0]*self._cell_size <= mouse_pos[0] <= agent.get_pos()[0]*self._cell_size+self._cell_size) and (agent.get_pos()[1]*self._cell_size <= mouse_pos[1] <= agent.get_pos()[1]*self._cell_size+self._cell_size):
                highlighted_path = agent.get_route_taken()

            if highlighted_path is not None:
                last_pos = None
                for cell in highlighted_path:
                    if last_pos is None:
                        last_pos = cell
                    # pg.draw.rect(self._window, (0, 255, 0), ((cell[0] * self._cell_size) + (0.4 * self._cell_size), (cell[1] * self._cell_size) + (0.4 * self._cell_size), self._cell_size * 0.2, self._cell_size * 0.2))
                    pg.draw.line(self._window, (255, 215, 0), (last_pos[0] * self._cell_size + (self._cell_size * 0.5), last_pos[1] * self._cell_size + (self._cell_size * 0.5)), (cell[0] * self._cell_size + (self._cell_size * 0.5), cell[1] * self._cell_size + (self._cell_size * 0.5)), 5)
                    last_pos = cell

            colour = (0, 0, 255)
            if agent.get_strategy() == 'i':
                colour = (255, 0, 0)
            elif agent.get_strategy() == 'n':
                colour = (0, 200, 0)
            centre_pos = ((agent.get_pos()[0]*self._cell_size)+(self._cell_size/2), (agent.get_pos()[1]*self._cell_size)+(self._cell_size/2))
            pg.draw.circle(self._window, colour, centre_pos, self._cell_size/3)

        pg.display.update()

    def _invalidate_paths(self):
        """Marks all agent paths invalid and creates a thread to calculate new paths"""
        for agent in self._agents:
            agent.path_valid = False

    def _clear_cell(self, pos):
        """Clears any agent or wall from cell, except for border walls"""
        if self._grid[pos[1]][pos[0]].is_wall():
            if not self._grid[pos[1]][pos[0]].is_border():
                self._grid[pos[1]][pos[0]].clear_wall()
                self._invalidate_paths()
                return True
        else:
            for agent in self._agents:
                if agent.get_pos() == pos:
                    self._agents.remove(agent)
                    self._grid[pos[1]][pos[0]].set_occupied(False)
                    return True
        return False

    def _on_mouse_down(self):
        pos = pg.mouse.get_pos()
        if self._mouse_button_down == 1:
            self._add_agent((math.floor(pos[0] / self._cell_size), math.floor(pos[1] / self._cell_size)), 'p')

    def _while_mouse_down(self):
        pos = pg.mouse.get_pos()
        if self._mouse_button_down == 2:
            self._add_wall((math.floor(pos[0] / self._cell_size), math.floor(pos[1] / self._cell_size)))
        elif self._mouse_button_down == 3:
            self._clear_cell((math.floor(pos[0] / self._cell_size), math.floor(pos[1] / self._cell_size)))

    def _plot_strategies(self):
        """Plots agent strategy distribution to graph"""
        plt.close()
        t = np.arange(0, self._time, 1)
        fig, ax = plt.subplots()

        ax.plot(t, self._patient_distribution, label='Patient', color=(0, 0, 1))
        ax.plot(t, self._impatient_distribution, label='Impatient', color=(1, 0, 0))
        ax.plot(t, self._neutral_distribution, label='Neutral', color=(0, 0.8, 0))
        ax.legend(shadow=True, fancybox=True)

        ax.set_xlabel('Time')
        ax.set_ylabel('Distribution')
        ax.set_title('Change in strategy')
        plt.show()

    def _run_one_step(self):
        if len(self._agents) > 0:
            if not self._move:
                self._update_agent_strategies()
                self._plot_strategies()
            else:
                self._move_agents()
                self._diffuse_df()
                self._invalidate_paths()
            self._move = not self._move

    def start(self):
        """Main loop for the simulation"""
        pg.init()
        pg.font.init()
        self._window = pg.display.set_mode(((self._grid_size[0]) * self._cell_size, (self._grid_size[1]) * self._cell_size))
        done = False
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                if event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self._auto_run_delay = clamp(self._auto_run_delay - 0.1, 0.1, 2)
                    elif event.button == 5:
                        self._auto_run_delay = clamp(self._auto_run_delay + 0.1, 0.1, 2)
                    else:
                        self._mouse_button_down = event.button
                        self._on_mouse_down()
                elif event.type == pg.MOUSEBUTTONUP:
                    self._while_mouse_down()
                    self._mouse_button_down = None
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        if not self._auto_run:
                            self._run_one_step()
                        else:
                            self._running = not self._running
                    elif event.key == pg.K_r:
                        self._auto_run = not self._auto_run
                        self._running = False
                elif self._mouse_button_down is not None:
                    self._while_mouse_down()
            if self._running:
                self._run_one_step()
                time.sleep(self._auto_run_delay)
            self._draw_grid()
        pg.quit()


if __name__ == '__main__':
    sim = SpatialDynamics((25, 25), 35, (24, 24), 2, 2, 0.4, 1, 10, 50, False)
    # sim.fill_grid_random(0.03, 0.03, 0.03)
    sim.start()
