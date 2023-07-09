from abc import ABCMeta, abstractmethod

class BoardGameBase(metaclass=ABCMeta):
    map = []

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass