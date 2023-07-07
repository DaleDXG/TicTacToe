from abc import ABCMeta, abstractmethod

class BoardGameBase(metaclass=ABCMeta):
    map = []

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def addPiece(self, raw, colume):
        pass

    @abstractmethod
    def checkWinning(self):
        pass

    @abstractmethod
    def checkFull(self):
        pass
    
    # def checkTermination(self, board = None):
    #     if board == None:
    #         board = self.map
    #     if self.checkFull(board) or self.checkWinning(board) != 0:
    #         return True
    #     else:
    #         return False