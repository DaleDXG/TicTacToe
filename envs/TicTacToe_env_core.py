from BoardGameBase import BoardGameBase

from gym import spaces
import numpy as np
import pygame



class TicTacToe_env_core(BoardGameBase):
    """
    Member variable:
    map
    flagTermination
    turn
    num_dim
    dim
    len_map
    num_in_a_row (win_condition, win_length)
    """



    def __init__(self, size=3, num_dim=2, dim=None, num_in_a_row=3, gravity_setting=None, players=(1, 1)):
        if dim == None:
            for i in range(num_dim):
                dim[i] = size
        self.num_dim = num_dim
        self.dim = dim
        self.num_in_a_row = num_in_a_row
        
        self.observation_space = spaces.Dict({
            "board": 
        })
        
        self.reset()

    def reset(self):
        self.len_map = 1
        for i in self.dim:
            self.len_map *= i
        self.map = [0] * self.len_map
        self.flagTermination = False
        self.turn = True # True for player1, flase for player2
    
    def step(self, action):
        observation = None
        reward = None
        done = None
        info = None
        return observation, reward, done, info
    
    def convert_coordinate_to_index(self, *args):
        if type(args[0]) == tuple:
            args[1] = args[0][1]
            args[0] = args[0][0]
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0] + args[1] * self.num_col
    
    def convert_index_to_coordinate(self, index):
        x = index % self.num_col
        y = index // self.num_col
        return (x, y)
        
    # def __

    def display_console(self):
        #
        for i in range(3):
            for j in range(3):
                if self.map[i * 3 + j] == 0:
                    print('0', end='')
                elif self.map[i * 3 + j] == 1:
                    print('A', end='')
                else:
                    print('B', end='')
            print('') # start a new line
    
    # add piece according to the turn
    def addPiece(self, row, column):
        num = row * 3 + column
        if self.map[num] == 0:
            if self.turn:
                self.map[num] = 1
            else:
                self.map[num] = -1
            self.turn = not self.turn
            return True
        else:
            return False

    # 0 for no one won the game yet
    # 1 for player1 won
    # 2 for player2 won
    def checkWinning(self, board = None):
        if board == None:
            board = self.map
            flagNoInput = True
        else:
            flagNoInput = False
        for i in range(3):
            if board[i * 3] != 0 and board[i * 3] == board[i * 3 + 1] and board[i * 3 + 1] == board[i * 3 + 2]:
                if flagNoInput:
                    self.flagTermination = True
                if board[i * 3] == 1:
                    return 1
                else:
                    return -1
            if board[i] != 0 and board[i] == board[i + 3] and board[i] == board[i + 6]:
                if flagNoInput:
                    self.flagTermination = True
                if self.map[i] == 1:
                    return 1
                else:
                    return -1
        if board[0] != 0 and board[0] == board[4] and board[4] == board[8]:
            if flagNoInput:
                self.flagTermination = True
            if board[0] == 1:
                return 1
            else:
                return -1
        if board[2] != 0 and board[2] == board[4] and board[4] == board[6]:
            if flagNoInput:
                self.flagTermination = True
            if board[2] == 1:
                return 1
            else:
                return -1
        return 0
    
    def checkFull(self, board = None):
        if board == None:
            board = self.map
            flagNoInput = True
        else:
            flagNoInput = False
        for i in range(9):
            if board[i] == 0:
                return False
        if flagNoInput:
            self.flagTermination = True
        return True

    def isTerminated(self):
        return self.flagTermination



class GameState_TicTacToe:    
    winners = [
        [1,1,1,]
    ]
    
    def __init__(self, board, playerTurn):
        self.board = board
        # 该谁落子是谁的回合
        self.playerTurn = playerTurn
    
    def get_state(self):
        pass
    
    def set_state(self):
        pass



if __name__ == "__main__":
    t3 = TicTacToe(num_dim = 3, dim=(3,4,4))
    print(t3.len_map)
    print(len(t3.map))