import numpy as np
import matplotlib.pyplot as plt

x = [32, 640, 500, 1000, 236, 128]
y = [.290957, .335536, .335766, .491832, .289599, .281801]
x, y = zip(*sorted(zip(x, y)))
x = np.array(x)
y = np.array(y)


plt.figure()
plt.plot(x, y)
plt.xlabel("Dimensions de l'image (pixels)")
plt.ylabel("Durée (s)")
plt.show()