from BoardGameBase import BoardGameBase



# haven't use
class DimentionMatchingError(Exception):
    """Raise when dimention is not matching"""
    pass



class TicTacToe_env_core(BoardGameBase):
    """
    """



    def __init__(self, size=3, num_dim=2, dims=None, num_in_a_row=3, gravity_setting=None, num_players=2):
        """
        description
        Args:
        Returns:
        Raises:
        """
        if dims == None:
            dims = [0] * num_dim
            for i in range(num_dim):
                dims[i] = size
            self.num_dim = num_dim
        else:
            self.num_dim = len(dims)
        self.dims = dims
        self.num_in_a_row = num_in_a_row

        self.len_map = 1
        for i in self.dims:
            self.len_map *= i
        self.num_players = num_players

        # winning checking direction
        self._winning_check_direction = self._get_winning_check_direction(self.num_dim)
        
        self.reset()

    # gym style functions. reset, step

    def reset(self):
        # map related
        self.map = [0] * self.len_map
        self.count_pieces = 0
        self.used_positions = []
        self.leftover_positions = []
        for i in range(self.len_map):
            self.leftover_positions.append(i)
        
        # game state related
        self._flag_termination = False
        self.current_player = 1
    
    def step(self, action):
        if self._flag_termination:
            self.reset()
        action_coordinates = self.calculate_index_to_coordinate(action)
        self._add_piece(action)
        if self._check_full():
            self._flag_termination = True
        if self._check_num_in_a_row_2_direction(action_coordinates):
            self._flag_termination = True
            reward = 1
        else:
            reward = 0
        # actually, ((current_player - 1) + 1) % num_players + 1
        self.current_player = self.current_player % self.num_players + 1

        observation = self.map
        done = self._flag_termination
        info = {
            'leftover_actions': self.leftover_positions
        }
        return observation, reward, done, info
    
    # util functions

    def _convert_coordinate_to_index(self, args):
        if type(args[0]) == tuple and len(args[0]) == self.num_dim:
            return self.calculate_coordinate_to_index(args[0])
        elif len(args) == self.num_dim:
            return self.calculate_coordinate_to_index(args)
        elif len(args) == 1:
            return args[0]
    
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
        return coordinates
    
    @staticmethod
    def _get_winning_check_direction(num_dim):
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
        return winning_check_direction
    
    def _add_coordinates(self, coordinates_1, coordinates_2):
        # could check DimentionMatchingError
        result = []
        for i in range(len(coordinates_1)):
            result.append(coordinates_1[i] + coordinates_2[i])
        return result
    
    def _multiply_coordinates(self, coordinates, multiplier):
        result = []
        for i in range(len(coordinates)):
            result.append(coordinates[i] * multiplier)
        return result

    def display_console(self):
        # haven't complete
        for i in range(3):
            for j in range(3):
                if self.map[i * 3 + j] == 0:
                    print('0', end='')
                elif self.map[i * 3 + j] == 1:
                    print('A', end='')
                else:
                    print('B', end='')
            print('') # start a new line

    # game logic functions
    
    # add piece according to the turn
    def _add_piece(self, *coordinates):
        index = self._convert_coordinate_to_index(coordinates)
        if self.map[index] == 0:
            self.map[index] = self.current_player
            self.count_pieces += 1
            self.used_positions.append(index)
            self.leftover_positions.remove(index)
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
        position_value = self.map[self.calculate_coordinate_to_index(coordinates)]
        if not position_value:
            return 0
        for direction in self._winning_check_direction:
            for distance in range(self.num_in_a_row-1, -1, -1):
                position = self._add_coordinates(coordinates, self._multiply_coordinates(direction, distance))
                if not self._check_valid_coordinates(position):
                    break
                if position_value != self.map[self.calculate_coordinate_to_index(position)]:
                    break
            else:
                return position_value
        return 0
    
    def _check_num_in_a_row_2_direction(self, coordinates):
        position_value = self.map[self.calculate_coordinate_to_index(coordinates)]
        if not position_value:
            return 0
        for direction in self._winning_check_direction:
            count = 1
            for distance in range(1, self.num_in_a_row):
                position = self._add_coordinates(coordinates, self._multiply_coordinates(direction, distance))
                if not self._check_valid_coordinates(position):
                    break
                if position_value != self.map[self.calculate_coordinate_to_index(position)]:
                    break
                count += 1
            for distance in range(-1, -self.num_in_a_row, -1):
                position = self._add_coordinates(coordinates, self._multiply_coordinates(direction, distance))
                if not self._check_valid_coordinates(position):
                    break
                if position_value != self.map[self.calculate_coordinate_to_index(position)]:
                    break
                count += 1
            if count >= self.num_in_a_row:
                return position_value
        return 0

    # 0 for no one won the game yet
    # 1 for player1 won
    # 2 for player2 won
    def _check_winning_3x3(self, board = None):
        if board == None:
            board = self.map
        for i in range(3):
            if board[i * 3] != 0 and board[i * 3] == board[i * 3 + 1] and board[i * 3 + 1] == board[i * 3 + 2]:
                return board[i * 3]
            if board[i] != 0 and board[i] == board[i + 3] and board[i] == board[i + 6]:
                return self.map[i]
        if board[0] != 0 and board[0] == board[4] and board[4] == board[8]:
            return board[0]
        if board[2] != 0 and board[2] == board[4] and board[4] == board[6]:
            return board[2]
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
            if board[i] == 0:
                return False
        return True



if __name__ == "__main__":
    t3 = TicTacToe_env_core(dims=(3,3))