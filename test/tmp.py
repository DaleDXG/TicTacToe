
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import util
import numpy as np
from collections.abc import Iterable
from collections import deque

import Config
from model_baselines.dqn.agent01 import DQN


# import subprocess
# import time

# cmd_list = []
# cmd_list.append("ls -a")
# cmd_list.append("mkdir test")



# with open("output.txt", "a") as f:
#     for index, cmd in enumerate(cmd_list):
#         time_before = time.time()

#         subprocess.run(cmd, shell=True, stdout=f)
    
#         time_after = time.time()

#         f.write('\n the command ' + str(index) + ' \"'+ cmd +'\" consume ' + str(time_after-time_before) +' seconds \n')

# a = 3.1415926
# b = 1.4142145

# print('\nTest accuracy: %f \nTest loss: %f' % (a, b))


# a = 12
# b = [1, 2, 3]
# c = [[1, 2], [3, 4]]
# d = np.array([1])
# e = np.array([1, 2, 3])
# f = np.array([[1, 2], [3, 4]])

# print(util.shape_to_num(a))
# print(util.shape_to_num(b))
# print(util.shape_to_num(c))
# print(util.shape_to_num(d))
# print(util.shape_to_num(e))
# print(util.shape_to_num(f))

# print(d)
# print(e)
# print(f)

# if type(a) == int:
#     print('aha!')
# else:
#     print('ahaha!')
# if type(b) == int:
#     print('bha!')
# else:
#     print('bhaha!')
# if type(c) == int:
#     print('cha!')
# else:
#     print('chaha!')
# if type(d) == int:
#     print('dha!')
# else:
#     print('dhaha!')
# if type(e) == int:
#     print('eha!')
# else:
#     print('ehaha!')
# if type(f) == int:
#     print('fha!')
# else:
#     print('fhaha!')


# def func1(int):
#     return int + 1

# def func2(callback):
#     a = 1
#     return callback(a)

# def func3(callback):
#     return callback(func1)

# print(func3(func2))



# a = [[0,0,0],[0,0,0],[0,0,0]]
# b = np.array([[0,0,0],[0,0,0],[0,0,0]])
# # print(type(a))
# # print(isinstance(a, Iterable))
# print(util.flatten_list(a))
# print(util.flatten_list(b))



test_dequeue = deque(maxlen=5)
for i in range(30):
    test_dequeue.append(i)
    print(len(test_dequeue))
    print(test_dequeue[-1])



# model_config = Config.InputConfig_Method(shape_layers=(9,9,9,9),
#                                          epsilon_greedy = 0)
# model = DQN(model_config)

# test_a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# test_b = [test_a, test_a, test_a, test_a]
# print(test_b)
# test_c = np.array([test[0] for test in test_b])
# print(test_c)
# np.array([test_a])

# model.eval_model.fit()