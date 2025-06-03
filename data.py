import numpy as np
import csv

num_steps = 100
num_targets = 2
data = []

for t in range(num_steps):
    row = []
    for i in range(num_targets):
        x = 50 + 10 * np.sin(0.1 * t + i)
        y = 50 + 10 * np.cos(0.1 * t + i)
        row.extend([x, y])
    data.append(row)

with open('sensor_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)