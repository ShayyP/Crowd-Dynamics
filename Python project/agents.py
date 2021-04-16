import math


class Agent:
    def __init__(self, pos, starting_strategy):
        """Create new agent"""
        self._pos, self._strategy, self._last_strategy = pos, starting_strategy, starting_strategy
        self._distance_to_exit = 0
        self._order_payoff = 1
        self._t_aset = 20
        self._t0 = 10
        # Dictionary that returns the cost of each types, if both patient or impatient this must be calculated at runtime
        self._cost_table = {'i': {'i': 'ii', 'p': (-1, 1), 'n': (0, 0)},
                            'p': {'i': (1, -1), 'p': 'pp', 'n': (0, 0)},
                            'n': {'i': (0, 0), 'p': (0, 0), 'n': (0, 0)}}
        self._t_i = 0
        self._chosen_move = (0, 0)
        self._possible_strategies = ['p', 'i', 'n']
        self._route_taken = [pos]
        # Dictionary that stores positions and their current deterrent
        self._route_taken_deterrent = {pos : 0.001}

    def get_pos(self):
        """Accessor method"""
        return self._pos

    def get_strategy(self):
        """Accessor method"""
        return self._last_strategy

    def toggle_strategy(self):
        """Accessor method"""
        self._last_strategy = self._possible_strategies[(self._possible_strategies.index(self._last_strategy) + 1) % len(self._possible_strategies)]

    def update_distance_to_exit(self, exit_pos):
        """Calculates distance to the exit and stores it"""
        self._distance_to_exit = math.sqrt((self._pos[0] - exit_pos[0]) ** 2 + (self._pos[1] - exit_pos[1]) ** 2)

    def get_distance_to_exit(self):
        """Accessor method"""
        return self._distance_to_exit

    def update_t_i(self, agents, exit_capacity):
        """Updates value of ti"""
        agents_closer_to_exit = 0
        for agent in agents:
            if agent.get_distance_to_exit() < self._distance_to_exit:
                agents_closer_to_exit += 1
        self._t_i = agents_closer_to_exit / exit_capacity

    def get_t_i(self):
        """Accessor method"""
        return self._t_i

    def _calculate_delta_u(self, t_ij, c):
        """Returns value of delta u using provided values of t_ij and c"""
        if t_ij <= self._t_aset - self._t0:
            return 0
        return (c / self._t0) * (t_ij - self._t_aset + self._t0)

    def calculate_pp_cost(self, t_j, c):
        """Calculates the cost when both agents are patient"""
        t_ij = (self._t_i + t_j) / 2
        delta_u = self._calculate_delta_u(t_ij, c)
        if delta_u != 0:
            return -self._order_payoff / delta_u
        return 0

    def calculate_ii_cost(self, t_j, c):
        """Calculates the cost when both agents are impatient"""
        t_ij = (self._t_i + t_j) / 2
        delta_u = self._calculate_delta_u(t_ij, c)
        if delta_u != 0:
            return c / delta_u
        return 0

    def update_strategy(self, c, agents):
        """Updates agent to new strategy"""
        sum_patient, sum_impatient, sum_neutral = 0, 0, 0
        # Loop through agents for any that are within our Moore neighbourhood
        for agent in agents:
            if agent != self:
                if self._pos[0] - 1 <= agent.get_pos()[0] <= self._pos[0] + 1 and self._pos[1] - 1 <= agent.get_pos()[1] <= self._pos[1] + 1:
                    # Get cost to be patient
                    p_cost = self._cost_table['p'][agent.get_strategy()]
                    if p_cost != 'pp':
                        sum_patient += p_cost[0]
                    else:
                        sum_impatient += self.calculate_pp_cost(agent.get_t_i(), c)
                    # Get cost to be impatient
                    i_cost = self._cost_table['i'][agent.get_strategy()]
                    if i_cost != 'ii':
                        sum_impatient += i_cost[0]
                    else:
                        sum_impatient += self.calculate_ii_cost(agent.get_t_i(), c)
                    # Get cost to be Neutral
                    n_cost = self._cost_table['n'][agent.get_strategy()]
                    sum_neutral += n_cost[0]

        # Choose the strategy with the lowest cost
        lowest_cost = min(sum_patient, sum_impatient, sum_neutral)
        if lowest_cost == sum_patient:
            self._strategy = 'p'
            chosen_strategy = 'be patient'
        elif lowest_cost == sum_neutral:
            self._strategy = 'n'
            chosen_strategy = 'be neutral'
        else:
            self._strategy = 'i'
            chosen_strategy = 'be impatient'

        print(f'Agent at: {self._pos} chose to {chosen_strategy}. There cost to be patient was: {sum_patient}, their cost to be impatient was: {sum_impatient}, and their cost to be neutral was: {sum_neutral}.')

    def move_to_new_strategy(self):
        """Switch to next strategy"""
        self._last_strategy = self._strategy

    def get_chosen_move(self):
        """Accessor method"""
        return self._chosen_move

    def move(self, pos):
        """Move agent to new position"""
        self._pos = pos
        self._route_taken.append(pos)
        if pos in self._route_taken_deterrent:
            self._route_taken_deterrent[pos] /= 2
        else:
            self._route_taken_deterrent[pos] = 0.001

    def get_route_taken(self):
        """Accessor method"""
        return self._route_taken

    def get_deterrent(self, pos):
        """Accessor method"""
        if pos in self._route_taken_deterrent:
            return self._route_taken_deterrent[pos]
        return None
