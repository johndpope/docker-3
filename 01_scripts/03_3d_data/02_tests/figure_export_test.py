import matplotlib.pyplot as plt
import numpy as np
import pickle

# Plot
fig_object = plt.figure()
x = np.linspace(0,3*np.pi)
y = np.sin(x)
plt.title("Hallo1")
plt.plot(x,y)
# Save to disk
pickle.dump(fig_object,open('sinus.pickle','wb'))

fig_object = pickle.load(open('sinus.pickle','rb'))
plt.title("Hallo")
plt.legend("Hallo")
plt.show()
# fig_object.show()