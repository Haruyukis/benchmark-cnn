import numpy as np
import matplotlib.pyplot as plt

x = [32, 640, 500, 1000, 236, 128]
y_gpu = [.099468, .098807, .088489, .177026, .079913, .075884]
y_cpu = [.004099, .193556, .124304, .497047, .030151, .011374]
_, y_gpu = zip(*sorted(zip(x, y_gpu)))
x, y_cpu = zip(*sorted(zip(x, y_cpu)))


plt.figure()
plt.plot(x, y_gpu, label='GPU')
plt.plot(x, y_cpu, label='CPU')
plt.xlabel("Dimensions de l'image (pixels)")
plt.ylabel("Durée (s)")
plt.legend()
plt.show()