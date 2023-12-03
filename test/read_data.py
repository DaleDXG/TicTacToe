
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

import util

# util.read_log_trueskill('logs/logger_trueskill_dqn_cnn_type00_00.log', 2)
util.read_log_env('logs/logger_env_dqn_cnn_type00_00.log', 2)