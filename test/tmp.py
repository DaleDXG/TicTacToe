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

a = 3.1415926
b = 1.4142145

print('\nTest accuracy: %f \nTest loss: %f' % (a, b))
