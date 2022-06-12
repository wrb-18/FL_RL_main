import numpy
a = numpy.array([[2, 3], [3, 4], [4, 3]])
a[a[:, 0] > 2,0] = 2
print(a)