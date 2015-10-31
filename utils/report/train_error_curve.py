# Read the log file and generate training error and validation error

import sys
import matplotlib.pyplot as plt

extension = "_error.png"

try:
	assert(len(sys.argv) >= 2)
except Exception:
	print("Oops! OPlease enter the file name.")
	sys.exit(0)

src_path = sys.argv[1]
last_point = src_path.rfind('.')
dest_path = src_path[:last_point] + extension

start = False
start_train = False
start_test = False
iteration = 0

train_err = []
val_err = []
test_err = 0

with open (src_path, "r") as f:
    for line in [_line.rstrip('\n') for _line in f]: # Delete the last \n
    	if line == "XNet v1.0":
    		start = True
    	if start:
    		if start_test:
    			pos = line.find(':')
    			if pos >= 0:
    				test_err = float(line[pos + 2:])
    		if not start_train and not start_test and line[0] == 'T':
    			int_start = line.find(' ') + 1;
    			int_end = line.find("times") - 1;
    			iteration = int(line[int_start:int_end])
    			start_train = True
    		elif start_train:
    			if line[0] == 'T':
    				error = float(line[line.find(':') + 2:])
    				train_err.append(error)
    			if line[0] == 'V':
    				error = float(line[line.find(':') + 2:])
    				val_err.append(error)
    			if line[0] == 'D':
    				start_train = False
    				start_test = True

x = range(iteration)

skip = 3 # skip some first data which are too big

plt.ylabel('Error rate (%)')
plt.xlabel('Epoch')
plt.annotate('test error', xy=(iteration * 9 // 10, test_err * 0.9), 
	xytext=(iteration * 7 // 10, test_err * 0.7), 
	arrowprops=dict(facecolor='black', shrink=0.1))

plt.plot(x[skip:], train_err[skip:], 'b', label='training error')
plt.plot(x[skip:], val_err[skip:], 'r', label='validation error')
plt.plot(x[iteration * 4 // 5:], 
	[test_err] * (iteration - iteration * 4 // 5), 'g--')
plt.legend(loc='upper right', shadow=True)

plt.savefig(dest_path)
plt.show()
