import math
import numpy as np
import matplotlib.pyplot as plt

x = np.sort(np.random.rand(50) ) * 100.0

def plot_results(x,xdf):
	plt.subplot(2,1,1)
	plt.plot(x)
	plt.title('x')
	plt.plot(x,'+')
	plt.ylabel('distance')
	plt.xlabel('sample#')
	plt.legend(['original distances'], loc='lower right')
	plt.subplot(2,1,2)
	plt.plot(xdf)
	plt.plot(xdf,'+')
	plt.title('xdf')
	plt.ylabel('softmax')
	plt.xlabel('sample#')
	plt.legend(['probability'], loc='upper right')

	plt.tight_layout()
	plt.show()


# x≥μ;β>0
def pdf(x, β=1, μ=0):
	xp =  (1/β) * np.exp(-(x-μ)/β)
	return xp / np.sum(xp)
	

def softmax(x):

	_mean = np.mean(x)
	_scale = np.sqrt(50) #_mean / 10.0
	_offset = np.min(x)
	expd = pdf(x, β=_scale, μ=_offset)
	print (["{0:.2f}".format(t) for t in x][:5],'=>' , ["{0:.2f}".format(t) for t in expd][:5])
	print ('mean {0:.2f} offset {1:.2f} scale {2:.2f} sum {3:.2f}'.
		   format(_mean, _offset , _scale , np.sum(expd)))
	plot_results(x,expd)



def main():
	softmax(x)

if __name__ == '__main__':
 	main() 

