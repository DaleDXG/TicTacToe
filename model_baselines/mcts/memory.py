import numpy as np
from collections import deque

import Config

class Memory:
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = Config.MEMORY_SIZE
		self.ltmemory = deque(maxlen=Config.MEMORY_SIZE)
		self.stmemory = deque(maxlen=Config.MEMORY_SIZE)

	def commit_stmemory(self, input, action_values, player_turn): #, identities
		# for r in identities(state, actionValues):
		# 	self.stmemory.append({
		# 		'board': r[0].board
		# 		, 'state': r[0]
		# 		, 'id': r[0].id
		# 		, 'AV': r[1]
		# 		, 'playerTurn': r[0].playerTurn
		# 		})
		# r[0] = state, r[1] = action_value
		self.stmemory.append({
			'inputs': input
			, 'action_values': action_values
			, 'player_turn': player_turn
			})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=Config.MEMORY_SIZE)
		