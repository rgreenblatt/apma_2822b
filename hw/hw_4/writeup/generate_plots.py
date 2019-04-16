# libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# font = {'family' : 'normal',
#         'weight': 'normal',
#         'size': 16}

# matplotlib.rc('font', **font)

# set width of bar
barWidth = 0.25

# set height of bar
gpu_times = [5.627600e-05, 7.177600e-05]
cpu_times = [1.163711e-03, 1.532791e-03]

# Set position of bar on X axis
r1 = np.arange(len(gpu_times))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, gpu_times, color='#7f6d5f', width=barWidth, edgecolor='white',
        label='gpu')
plt.bar(r2, cpu_times, color='#557f2d', width=barWidth, edgecolor='white',
        label='cpu')

# Add xticks on the middle of the group bars
plt.xlabel('Method', fontweight='bold')
plt.ylabel('Average running time (s)', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(gpu_times))], ['CRS', 'ELLPACK'])

# Create legend & Show graphic
plt.legend()
plt.savefig("running_times.png")
