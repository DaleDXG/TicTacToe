



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