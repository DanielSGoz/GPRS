import matplotlib.pyplot as plt
import math
import numpy

ACC = 10000
alpha = 0.0001


def infinite_sum(z):
	global ACC, alpha
	SUM = 0.0

	for k in range(ACC, -1, -1):
		SUM = SUM * z / (k + 1)
		# SUM = SUM + (k + 1)**alpha
		SUM = SUM + math.log(k + 1)
		
	return SUM

def val(x):
	global alpha
	# return infinite_sum(x) / (math.exp(x) * (x + 1)**alpha)
	# return math.log2(infinite_sum(x) / (math.exp(x) * (x + 1)**alpha)) / alpha
	return math.exp(-x) * infinite_sum(x) - math.log(x + 1)



x = numpy.linspace(0, 10, 300)
y = []
for i in range(len(x)):
	y.append(val(x[i]))


plt.plot(x, y, label='1')
plt.legend()
plt.show()
