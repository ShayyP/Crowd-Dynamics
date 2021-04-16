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
        """Creates the simulation"""
        # Model parameters
        self._c, self._df_diffuse_rate, self._df_increase, self._df_strength, self._sf_strength = cost_of_congestion, df_diffuse_rate, df_increase, df_strength, sf_strength
        # Pygame/simulation style variables
        self._window, self._mouse_button_down = None, None
        self._auto_run, self._running, self._move = False, False, False
        self._exit_pos = (exit_pos[0] + 1, exit_pos[1] + 1)
        self._exit_capacity, self._cell_size, self._show_probs = exit_capacity, cell_size, show_probs
        self._grid_size = (grid_size[0] + 2, grid_size[1] + 2)
        self._auto_run_delay = 0.5
        # Core variables used in simulation
        self._agents, self._grid, self._patient_distribution, self._impatient_distribution, self._neutral_distribution = [], [], [], [], []
        self._time = 0
        # Create grid after initialisation
        self._create_grid()

    def fill_grid_random(self, patient_weight, impatient_weight, neutral_weight):
        """Optional method that fills the grid with agents, random but weighted"""
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
        # Save agents distribution to be plotted later, progress one time step
        self._time += 1
        self._patient_distribution.append(patient_agents / (patient_agents + impatient_agents + neutral_agents))
        self._impatient_distribution.append(impatient_agents / (patient_agents + impatient_agents + neutral_agents))
        self._neutral_distribution.append(neutral_agents / (patient_agents + impatient_agents + neutral_agents))

    def _move_agents(self):
        """Moves agents using the probability based model"""
        moves = []
        # Calculate probability to move to each neighbouring cell
        for agent in self._agents:
            agent_moves, agent_probabilities = [], []
            # Loop through cells in Moore neighbourhood
            for x in range(-1, 2):
                for y in range(-1, 2):
                    move = (agent.get_pos()[0] + x, agent.get_pos()[1] + y)
                    cell = self._grid[move[1]][move[0]]
                    sf_multiplier, df_multiplier = 1, 1
                    # If agent is impatient, they are more inclined to rush to the exit, this is reflected by increasing sf
                    if agent.get_strategy() == 'i':
                        sf_multiplier = 10
                    # If agent is neutral, they are more inclined to follow other agents, this is reflected by increasing df
                    elif agent.get_strategy() == 'n':
                        df_multiplier = 10
                    # Calculate probability (not normalised)
                    probability = math.pow(math.e, cell.get_df() * self._df_strength * df_multiplier) * math.pow(math.e, (cell.get_sf() * self._sf_strength * sf_multiplier)) * (1 - cell.get_occupied_multiplier())
                    # Apply deterrent to move if agent has been here already
                    if move in agent.get_route_taken():
                        probability *= agent.get_deterrent(move)
                    # Save move and probability
                    agent_moves.append(move)
                    agent_probabilities.append(probability)

            # Normalise the probabilities
            sum_prob = 0
            for prob in agent_probabilities:
                sum_prob += prob
            if sum_prob != 0:
                for i in range(len(agent_probabilities)):
                    agent_probabilities[i] /= sum_prob
                # Choose a move
                chosen_move = random.choices(agent_moves, agent_probabilities)[0]
                # Check if any other agents are attempting to move here to, if so we must resolve this after
                if chosen_move is not None:
                    move_contested = False
                    for move in range(0, len(moves)):
                        if chosen_move in moves[move]:
                            # Save contested move to be resolved
                            moves[move].append(agent)
                            move_contested = True
                    if not move_contested:
                        # Other agents may still want to move here (after this agent in the list) so we save this move in case
                        moves.append([chosen_move, agent])

        # Loop through all moves, if not contested move the agent, otherwise resolve the contention
        for move in moves:
            target = move[0]
            # Normal cells can fit one agent, exit door can fit as many as specified by exit capacity
            space_available = 1
            if target == self._exit_pos:
                space_available = self._exit_capacity
            agents_to_choose_from = move[1:]
            # If enough space for all, move all
            if len(agents_to_choose_from) <= space_available:
                for agent in agents_to_choose_from:
                    self._move_agent(agent, target)
            else:
                # Split agents into list of each strategy
                impatient_agents, patient_agents, neutral_agents = [], [], []
                for agent in agents_to_choose_from:
                    if agent.get_strategy() == 'i':
                        impatient_agents.append(agent)
                    elif agent.get_strategy() == 'p':
                        patient_agents.append(agent)
                    else:
                        neutral_agents.append(agent)

                # Priority order is Impatient > Patient > Neutral
                # If multiple agents of same priority, chosen agent is random
                # Must resolve all Impatient before moving on
                while len(impatient_agents) > 0 and space_available > 0:
                    chosen_agent = random.choice(impatient_agents)
                    self._move_agent(chosen_agent, target)
                    impatient_agents.remove(chosen_agent)
                    space_available -= 1

                # Must resolve all Patient before moving on
                while len(patient_agents) > 0 and space_available > 0:
                    chosen_agent = random.choice(patient_agents)
                    self._move_agent(chosen_agent, target)
                    patient_agents.remove(chosen_agent)
                    space_available -= 1

                # If any space left, neutral agents can claim them
                while len(neutral_agents) > 0 and space_available > 0:
                    chosen_agent = random.choice(neutral_agents)
                    self._move_agent(chosen_agent, target)
                    neutral_agents.remove(chosen_agent)
                    space_available -= 1

    def _move_agent(self, agent, pos):
        """Moves an agent to a specified position"""
        # If move is the exit, remove the agent from the grid and add their df trail
        if pos == self._exit_pos:
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].set_occupied(False)
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].change_df(self._df_increase)
            self._grid[agent.get_pos()[1]][agent.get_pos()[0]].update_to_new_df()
            # Multiplier is used to scale df value dependant on how recently the agent was there
            # This encourages agents following the trail to move in the correct direction
            current_multiplier = 1
            length = len(agent.get_route_taken())
            for grid_pos in agent.get_route_taken():
                self._grid[grid_pos[1]][grid_pos[0]].change_df(self._df_increase * (current_multiplier / length))
                self._grid[grid_pos[1]][grid_pos[0]].update_to_new_df()
                current_multiplier += 1
            self._agents.remove(agent)
            print(f'Agent at: {agent.get_pos()} has left through the exit.')
        # Otherwise move agent as normal
        else:
            old_pos = agent.get_pos()
            agent.move(pos)
            print(f'Agent at: {old_pos} has moved to {agent.get_pos()}.')
            self._grid[old_pos[1]][old_pos[0]].set_occupied(False)
            # Deprecated method of increasing df on every move
            # New method of adding df when agent reaches exit encourages agents to only follow agents that were successful in their escape
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
        """Draws everything to the window using pygame, to be called every frame"""
        # Background fill
        self._window.fill((255, 255, 255))
        # Draw cells
        for x in range(0, self._grid_size[0]):
            for y in range(0, self._grid_size[1]):
                cell = self._grid[y][x]
                # If show probabilities is on, df values affect the brightness of the cell, otherwise it is one colour
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
        # Draw exit location
        pg.draw.rect(self._window, (0, 255, 0), (self._exit_pos[0] * self._cell_size, self._exit_pos[1] * self._cell_size, self._cell_size, self._cell_size))

        # Draw lines between cells
        for x in range(1, self._grid_size[0]):
            pg.draw.line(self._window, (0, 0, 0), (x * self._cell_size, 0), (x * self._cell_size, self._grid_size[0] * self._cell_size), 1)
        for y in range(1, self._grid_size[1]):
            pg.draw.line(self._window, (0, 0, 0), (0, y * self._cell_size), (self._grid_size[1] * self._cell_size, y * self._cell_size), 1)

        mouse_pos = pg.mouse.get_pos()
        highlighted_path = None
        for agent in self._agents:
            # Check if mouse is over an agent
            if (agent.get_pos()[0] * self._cell_size <= mouse_pos[0] <= agent.get_pos()[0] * self._cell_size+self._cell_size) and (agent.get_pos()[1] * self._cell_size <= mouse_pos[1] <= agent.get_pos()[1] * self._cell_size+self._cell_size):
                highlighted_path = agent.get_route_taken()

            # If mouse is over an agent, draw their path to the grid
            if highlighted_path is not None:
                last_pos = None
                for cell in highlighted_path:
                    if last_pos is None:
                        last_pos = cell
                    pg.draw.line(self._window, (255, 215, 0), (last_pos[0] * self._cell_size + (self._cell_size * 0.5), last_pos[1] * self._cell_size + (self._cell_size * 0.5)), (cell[0] * self._cell_size + (self._cell_size * 0.5), cell[1] * self._cell_size + (self._cell_size * 0.5)), 5)
                    last_pos = cell

            # Draw agents to the grid, colours used are: blue for patient, red for impatient, green for neutral
            colour = (0, 0, 255)
            if agent.get_strategy() == 'i':
                colour = (255, 0, 0)
            elif agent.get_strategy() == 'n':
                colour = (0, 200, 0)
            # Draw circle in middle of cell to represent agent
            centre_pos = ((agent.get_pos()[0]*self._cell_size)+(self._cell_size/2), (agent.get_pos()[1]*self._cell_size)+(self._cell_size/2))
            pg.draw.circle(self._window, colour, centre_pos, self._cell_size/3)

        # Update the display to add the new things we drew
        pg.display.update()

    def _clear_cell(self, pos):
        """Clears any agent or wall from cell, except for border walls"""
        if self._grid[pos[1]][pos[0]].is_wall():
            if not self._grid[pos[1]][pos[0]].is_border():
                self._grid[pos[1]][pos[0]].clear_wall()
                return True
        else:
            for agent in self._agents:
                if agent.get_pos() == pos:
                    self._agents.remove(agent)
                    self._grid[pos[1]][pos[0]].set_occupied(False)
                    return True
        return False

    def _on_mouse_down(self):
        """Called when a mouse button is clicked"""
        pos = pg.mouse.get_pos()
        # If left button, add agent at cursor position
        if self._mouse_button_down == 1:
            self._add_agent((math.floor(pos[0] / self._cell_size), math.floor(pos[1] / self._cell_size)), 'p')

    def _while_mouse_down(self):
        """Called every tick while a mouse button is held down"""
        pos = pg.mouse.get_pos()
        # If middle button, add wall at cursor position
        if self._mouse_button_down == 2:
            self._add_wall((math.floor(pos[0] / self._cell_size), math.floor(pos[1] / self._cell_size)))
        # If right button, clear cell at cursor position
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
        """Updates agent strategies or moves them, this changes each time it is called like a flip flop"""
        if len(self._agents) > 0:
            if not self._move:
                self._update_agent_strategies()
                self._plot_strategies()
            else:
                self._move_agents()
                self._diffuse_df()
            self._move = not self._move

    def start(self):
        """Main loop for the simulation"""
        # Start pygame and create a window
        pg.init()
        self._window = pg.display.set_mode(((self._grid_size[0]) * self._cell_size, (self._grid_size[1]) * self._cell_size))
        done = False
        # Loops until closed
        while not done:
            for event in pg.event.get():
                # Stop the simulation if red X is clicked
                if event.type == pg.QUIT:
                    done = True
                # If mouse button has been clicked
                if event.type == pg.MOUSEBUTTONDOWN:
                    # If scroll wheel, adjust auto run delay timer
                    if event.button == 4:
                        self._auto_run_delay = clamp(self._auto_run_delay - 0.1, 0.1, 2)
                    elif event.button == 5:
                        self._auto_run_delay = clamp(self._auto_run_delay + 0.1, 0.1, 2)
                    # Otherwise call related events
                    else:
                        self._mouse_button_down = event.button
                        self._on_mouse_down()
                # On mouse button release, stop while mouse down being called
                elif event.type == pg.MOUSEBUTTONUP:
                    self._while_mouse_down()
                    self._mouse_button_down = None
                # On key press
                elif event.type == pg.KEYDOWN:
                    # If spacebar pressed either run one step or start/stop auto run (if enabled)
                    if event.key == pg.K_SPACE:
                        if not self._auto_run:
                            self._run_one_step()
                        else:
                            self._running = not self._running
                    # If R key pressed toggle auto run
                    elif event.key == pg.K_r:
                        self._auto_run = not self._auto_run
                        self._running = False
                # Call while mouse down if the current mouse button being pressed is not null
                elif self._mouse_button_down is not None:
                    self._while_mouse_down()
            # Runs one step then waits specified delay (if auto run is enabled)
            if self._running:
                self._run_one_step()
                time.sleep(self._auto_run_delay)
            # Draw the grid with updated positions
            self._draw_grid()
        # Close pygame window before program ends
        pg.quit()


if __name__ == '__main__':
    # Create instance of the simulation
    sim = SpatialDynamics((25, 25), 35, (24, 24), 2, 2, 0.4, 1, 50, 50, False)
    # Uncomment line below to pre fill grid with agents
    # sim.fill_grid_random(0.03, 0.03, 0.03)
    # Run the simulation
    sim.start()
