from envs.BoardGameBase import BoardGameBase
from envs.Gravity import GravitySetting
from Config import InputConfig
import util

import numpy as np
import trueskill



# haven't use
class DimentionMatchingError(Exception):
    """Raise when dimention is not matching"""
    pass



class TicTacToe_env_core(BoardGameBase):
    """
    Member variable:
    map
    _flag_termination
    turn
    num_dim
    dims
    len_map
    num_in_a_row (win_condition, win_length)
    """

    REWARD_COMMON = 0
    REWARD_WIN = 10
    REWARD_LOSE = -5
    REWARD_PLACE_ON_AVAILABLE = 0
    REWARD_DUPLICATE_ACTION = -20

    def __init__(self, input_config=None):
        """
        description
        Args:
        Returns:
        Raises:
        """
        assert input_config != None, ('A InputConfig is needed before create an environment.')
        self._config = input_config
        
        self.num_dim = input_config['num_dim']
        self.dims = input_config['dims']
        self.num_in_a_row = input_config['num_in_a_row']
        self._flag_compute_used_left = input_config['flag_compute_used_left'] # 统计有子的位置和空余的位置
        self.len_map = input_config['len_map']
        self.num_players = input_config['num_players']

        self.num_episode = 0

        self._gravity = None
        if input_config['gravity_mode'] != 'no_gravity':
            self._gravity = GravitySetting(input_config)

        # winning checking direction
        self._winning_check_direction = self._get_winning_check_directions(self.num_dim)

        self.init_rating()
        
        self.reset()

    # gym style functions. reset, step

    def reset(self):
        # map related
        self.map = np.zeros(shape=self.dims, dtype="int8")
        self.count_pieces = 0
        self.used_positions = []
        self.leftover_positions = []
        if self._flag_compute_used_left:
            for i in range(self.len_map):
                self.leftover_positions.append(self.calculate_index_to_coordinate(i))
        
        # game state related
        self._flag_termination = False
        self.current_player = 1 # player number start from 1. I forgot why would I setted like this.
        self.previous_player = 0
        self.winner = -1
        self.duplicate_action_player = -1
        self.num_episode += 1

        return self._get_obs()
    
    def step(self, action):
        """
        Args:
        action, is a literable tuple or list, the coordinates of where the piece should be placed.
        """
        # assert self._flag_termination == False, (
        # 'Cant call step() once episode finished (call reset() instead)')
        reward = self.REWARD_COMMON
        # print(action)
        util.logger_env.info('player_' + str(self.current_player) + '_execute_action_' + str(action))
        if self._flag_termination == False:
            action = self._convert_to_coordinates(action)

            if not self._add_piece(action):
                util.logger_env.info('The_position_have_already_occupied')
                reward = self.REWARD_DUPLICATE_ACTION
                self._flag_termination = True
            else:
                reward = self.REWARD_PLACE_ON_AVAILABLE

                if self._check_full():
                    self._flag_termination = True
                if self._check_num_in_a_row_2_direction(action):
                    self._flag_termination = True
                    self.winner = self.current_player
                    reward = self.REWARD_WIN
                
            if self._flag_termination:
                self.update_rating(self.winner - 1)
        else:
            if self.current_player == self.winner:
                reward = self.REWARD_WIN
            elif self.winner == -1: # initial value, no winner
                reward = self.REWARD_COMMON
            else:
                reward = self.REWARD_LOSE

        self.previous_player = self.current_player
        # actually, ((current_player - 1) + 1) % num_players + 1
        self.current_player = self.current_player % self.num_players + 1

        observation = self._get_obs()
        done = self._flag_termination
        info = self._get_info()
        return observation, reward, done, info
    
    def _get_obs(self):
        return {
            'board': self.map
        }
    
    def _get_info(self):
        return {
            'leftover_actions': self.leftover_positions,
            'rating': self.players_rating
        }
    
    # util functions

    def _convert_to_index(self, args):
        if type(args[0]) == tuple and len(args[0]) == self.num_dim:
            return self.calculate_coordinate_to_index(args[0])
        elif len(args) == self.num_dim:
            return self.calculate_coordinate_to_index(args)
        elif len(args) == 1:
            return args[0]
    
    def _convert_to_coordinates(self, args):
        if type(args) == int or type(args) == np.int64:
            return self.calculate_index_to_coordinate(args)
        elif type(args[0]) == tuple and len(args[0]) == self.num_dim:
            return args[0]
        elif len(args) == self.num_dim:
            return args
        elif len(args) == 1:
            return self.calculate_index_to_coordinate(args[0])
    
    def calculate_coordinate_to_index(self, coordinates):
        index = 0
        product = 1
        for i in range(self.num_dim):
            index += coordinates[i] * product
            product *= self.dims[i]
        return index
    
    def calculate_index_to_coordinate(self, index):
        coordinates = []
        mask = 1
        for i in range(self.num_dim - 1):
            mask *= self.dims[i]
        for i in range(self.num_dim):
            coordinates.insert(0, int(index // mask))
            index %= mask
            mask /= self.dims[-(i+1)]
        return tuple(coordinates)
    
    @staticmethod
    def _get_winning_check_directions(num_dim):
        """
        """
        winning_check_direction = [[]]
        for i in range(num_dim):
            count = len(winning_check_direction)
            while count > 0:
                item = winning_check_direction.pop(0)
                for value in (1,0,-1):
                    item_copy = item.copy()
                    item_copy.append(value)
                    winning_check_direction.append(item_copy)
                count -= 1
            winning_check_direction.pop()
        winning_check_direction.pop()
        for i in range(len(winning_check_direction)):
            winning_check_direction[i] = tuple(winning_check_direction[i])
        return winning_check_direction

    def display_console(self, map = None):
        # haven't complete
        if map is None:
            print(self.map)
        else:
            print(map)
        print()

    # game logic functions
    
    # add piece according to the turn
    def _add_piece(self, *coordinates):
        # print(coordinates)
        # print(type(coordinates))
        coordinates = self._convert_to_coordinates(coordinates)
        util.logger_env.info('coordinates:' + str(coordinates))
        # print(type(coordinates))
        if self.map[coordinates] == 0:
            if self._gravity != None:
                coordinates = self._gravity.fall(self.map, coordinates)
            self.map[coordinates] = self.current_player
            self.count_pieces += 1
            if self._flag_compute_used_left:
                self.used_positions.append(coordinates)
                self.leftover_positions.remove(coordinates)
            return True
        else:
            return False
    
    ## win check related

    def _check_valid_coordinates(self, coordinates):
        # if len(coordinates) != self.num_dim:
        #     raise DimentionMatchingError
        for i in range(self.num_dim):
            if coordinates[i] < 0:
                return False
            if coordinates[i] > self.dims[i] - 1:
                return False
        return True
    
    def _check_num_in_a_row_simple(self, coordinates):
        position_value = self.map[coordinates]
        if not position_value:
            return 0
        for direction in self._winning_check_direction:
            for distance in range(self.num_in_a_row-1, -1, -1):
                position = util.add_coordinates(coordinates, util.multiply_coordinates(direction, distance))
                if not self._check_valid_coordinates(position):
                    break
                if position_value != self.map[position]:
                    break
            else:
                return position_value
        return 0
    
    def _check_num_in_a_row_2_direction(self, coordinates):
        position_value = self.map[coordinates]
        if not position_value:
            return 0
        for direction in self._winning_check_direction:
            count = 1
            for distance in range(1, self.num_in_a_row):
                position = util.add_coordinates(coordinates, util.multiply_coordinates(direction, distance))
                if not self._check_valid_coordinates(position):
                    break
                if position_value != self.map[position]:
                    break
                count += 1
            for distance in range(-1, -self.num_in_a_row, -1):
                position = util.add_coordinates(coordinates, util.multiply_coordinates(direction, distance))
                if not self._check_valid_coordinates(position):
                    break
                if position_value != self.map[position]:
                    break
                count += 1
            if count >= self.num_in_a_row:
                return position_value
        return 0
    
    def _check_full(self, board = None):
        # method 1
        if board == None:
            if self.count_pieces >= self.len_map:
                return True
            else:
                return False
        # method 2
        for i in range(self.len_map):
            if board.ravel()[i] == 0:
                return False
        return True
    
    def init_rating(self):
        self.true_skill_env = trueskill.TrueSkill()
        self.players_mu = []
        self.players_sigma = []
        
        self.players_rating = []
        for _ in range(self.num_players):
            # self.players_rating.append(trueskill.Rating())
            self.players_rating.append((self.true_skill_env.create_rating(),))
            self.players_mu.append([])
            self.players_sigma.append([])
        self.save_trueskill_info()
    
    def update_rating(self, winner):
        if winner >= 0:
            ranks = [1] * self.num_players
            ranks[winner] = 0
        else:
            ranks = [0] * self.num_players
        self.players_rating = self.true_skill_env.rate(self.players_rating, ranks=ranks)
        self.save_trueskill_info()
    
    def save_trueskill_info(self):
        util.logger_trueskill.info('-----new_result_' + str(self.num_episode) + '-----')
        for i, rating in enumerate(self.players_rating):
            # print(rating[0])
            util.logger_trueskill.info('player' + str(i) + ':' + str(rating))
            self.players_mu[i].append(rating[0].mu)
            self.players_sigma[i].append(rating[0].sigma)
