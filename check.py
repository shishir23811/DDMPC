import matplotlib.pyplot as plt
import numpy as np
from sim.references import figure_eight, double_lane_change

t = np.linspace (0 , 30 , 600)
fig8 = np.array ([figure_eight(ti) for ti in t ])
lane = np.array ([double_lane_change(ti) for ti in t ])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(fig8[:,0], fig8[:,1])
ax[0].set_title('Figure-8')
ax[0].axis ('equal')

ax[1].plot(lane[:,0], lane[:,1])
ax[1].set_title('Double lane')
# ax[1].axis ('equal')

plt.savefig( 'results/day1_references.png' , dpi =150)
plt.show ()