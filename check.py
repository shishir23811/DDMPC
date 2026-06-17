import matplotlib.pyplot as plt
import numpy as np
from sim.references import circle, figure_eight, double_lane_change, lane_change

t = np.linspace (0 , 32, 600)
fig8 = np.array ([figure_eight(ti) for ti in t ])
lane = np.array ([double_lane_change(ti) for ti in t ])
single_lane = np.array ([lane_change(ti) for ti in t ])
circle_path = np.array ([circle(ti) for ti in t ])

fig, ax = plt.subplots(2, 2, figsize=(20, 4))

ax[0][0].plot(fig8[:,0], fig8[:,1])
ax[0][0].set_title('Figure-8')
ax[0][0].axis ('equal')

ax[0][1].plot(lane[:,0], lane[:,1])
ax[0][1].set_title('Double lane')
# ax[1].axis ('equal')

ax[1][0].plot(single_lane[:,0], single_lane[:,1])
ax[1][0].set_title('Single lane')
# ax[2].axis ('equal')

ax[1][1].plot(circle_path[:,0], circle_path[:,1])
ax[1][1].set_title('Circle')
# ax[3].axis ('equal')

plt.savefig( 'results/day1_references.png' , dpi =150)
plt.show ()