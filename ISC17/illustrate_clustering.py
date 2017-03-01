#!/usr/bin/python

import time
import re
import subprocess
import shlex
import math
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

def normal_distribution_3d(x, y, mu, sigma): # mean, standard deviation
    p = [x, y]
    result = 1.0
    for i in range(2):
        first_factor = 1.0/math.sqrt(2*math.pi*sigma[i])
        exponent = (- (p[i] - mu[i])*(p[i] - mu[i])) / (2 * sigma[i] * sigma[i])
        result *= first_factor * np.exp(exponent)
    return result

level = 3
density_scale = 1.0
mus = [[0.25, 0.25], [0.75, 0.75]] # distribution means
sigmas = [[0.08, 0.08], [0.12, 0.12]] # distribution standard deviation

x = np.linspace(0, 1, 2 ** level)
y = np.linspace(0, 1, 2 ** level)
X, Y = np.meshgrid(x, y)

eval_list_first = normal_distribution_3d(X, Y, mus[0], sigmas[0])
first_distribution_grid = (X, Y, eval_list_first)

eval_list_second = normal_distribution_3d(X, Y, mus[1], sigmas[1])
first_distribution_grid = (X, Y, eval_list_second)

sum_of_results = eval_list_first + eval_list_second

# create dataset
samples = [200, 400]
datasets = []
for i in range(2):
    x_dataset = np.random.normal(mus[i][0], sigmas[i][0], samples[i]).tolist()
    y_dataset = np.random.normal(mus[i][1], sigmas[i][1], samples[i]).tolist()
    z_dataset = samples[i] * [0.]

    # remove out of bound entries
    x_dataset_fixed = []
    y_dataset_fixed = []
    z_dataset_fixed = []
    for i in range(len(x_dataset)):
        if x_dataset[i] >= 0.0 and x_dataset[i] <= 1.0 and y_dataset[i] >= 0.0 and y_dataset[i] <= 1.0 and z_dataset[i] >= 0.0 and z_dataset[i] <= 1.0:
            x_dataset_fixed += [x_dataset[i]]
            y_dataset_fixed += [y_dataset[i]]
            z_dataset_fixed += [z_dataset[i]]

    x_dataset = x_dataset_fixed
    y_dataset = y_dataset_fixed
    z_dataset = z_dataset_fixed

    datasets += [(x_dataset, y_dataset, z_dataset)]

# add noise
add_noise=True
noise_dataset = []
noise_datapoints=100
if add_noise:
    X = np.random.uniform(0, 1, noise_datapoints)
    Y = np.random.uniform(0, 1, noise_datapoints)
    noise_dataset = (X, Y)

# now do the plotting

# write dataset to arff file
f = open('dataset.arff', 'w')
f.write("""@RELATION "generated_dataset.arff"

@ATTRIBUTE x0 NUMERIC
@ATTRIBUTE x1 NUMERIC

@DATA
""")
# @ATTRIBUTE class NUMERIC
for i in range(len(datasets)):
    for p in range(len(datasets[i][0])):
        # f.write(str(datasets[i][0][p]) + "," + str(datasets[i][1][p]) + "," + str(i) + "\n")
        f.write(str(datasets[i][0][p]) + "," + str(datasets[i][1][p]) + "\n")

for p in range(len(noise_dataset[0])):
    # f.write(str(noise_dataset[0][p]) + "," + str(noise_dataset[1][p]) + "," + str(2) + "\n")
    f.write(str(noise_dataset[0][p]) + "," + str(noise_dataset[1][p]) + "\n")

f.close()

cmd = "clustering_poster --datasetFileName dataset.arff --eval_grid_level=" + str(level) + " --level=6 --lambda=0.000001"
print cmd
args = shlex.split(cmd)
print args
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False) #, shell=True , stdout=subprocess.PIPE, shell=0
# p.wait()
time.sleep(15)
output_streams = p.communicate()
stdout = output_streams[0]
stderr = output_streams[1]

# print output
m = re.search(r"\[(.*?)\]", stdout, re.MULTILINE)
print "stdout"
print stdout
print "stderr"
print stderr
print "------stdout-------"
print m.group(1)
evals = m.group(1).split(",")
evals = [float(e) for e in evals]
print evals

# colors = ["r", "g"]
# markers = ["o", "^"]
colors = ["r", "r", "r"]
markers = ["o", "o", "o"]

##################################################
# Plot dataset
##################################################

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

# plt.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='viridis') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

# if add_noise:
#     ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2])

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

# # ax.view_init(elev=90, azim=5)
# plt.show()
fig.savefig("clustering_1.png", dpi=300)

plt.close()

##################################################
# Estimate density
##################################################

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

plt.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='viridis', aspect='auto') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

# if add_noise:
#     ax.scatter(noise_dataset[0], noise_dataset[1], c='b')

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

# # ax.view_init(elev=90, azim=5)
# plt.show()
fig.savefig("clustering_2.png", dpi=300)

plt.close()

##################################################
# Now plot with knn
##################################################

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

plt.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='viridis', aspect='auto') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

display_clustered = True
density_threshold = 0.0
if display_clustered:
    X = []
    for d in datasets:
        for i in range(len(d[0])):
            X += [[d[0][i], d[1][i]]]

    if add_noise:
        for i in range(len(noise_dataset[0])):
            X += [[noise_dataset[0][i], noise_dataset[1][i]]]

    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)


    # middle_X = []
    # middle_Y = []
    for neighbor_indices in indices:
        for i in range(len(neighbor_indices)):
            if i == 0:
                continue
            # print X[neighbor_indices[0]], X[neighbor_indices[i]]
            P1 = [X[neighbor_indices[0]][0], X[neighbor_indices[0]][1]]
            P2 = [X[neighbor_indices[i]][0], X[neighbor_indices[i]][1]]
            dist_vector = [P2[0] - P1[0], P2[1] - P1[1]]
            dist_vector = [0.5 * dist_vector[0], 0.5 * dist_vector[1]]
            middle = [P1[0] + dist_vector[0], P1[1] + dist_vector[1]]

            # middle_X += [middle[0]]
            # middle_Y += [middle[1]]

            density = normal_distribution_3d(middle[0], middle[1], mus[0], sigmas[0])
            density += normal_distribution_3d(middle[0], middle[1], mus[1], sigmas[1])

            if density >= density_threshold:
                ax.plot([X[neighbor_indices[0]][0], X[neighbor_indices[i]][0]], [X[neighbor_indices[0]][1], X[neighbor_indices[i]][1]], color='y', zorder=1)

    # ax.scatter(middle_X, middle_Y, c='y')

# # ax.view_init(elev=90, azim=5)
# plt.show()
fig.savefig("clustering_3.png", dpi=300)

plt.close()


##################################################
# Now plot clusters
##################################################

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

plt.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='viridis', aspect='auto') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

# if add_noise:
#     ax.scatter(noise_dataset[0], noise_dataset[1], c='b')

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

display_clustered = True
density_threshold = 0.15
if display_clustered:
    X = []
    for d in datasets:
        for i in range(len(d[0])):
            X += [[d[0][i], d[1][i]]]

    if add_noise:
        for i in range(len(noise_dataset[0])):
            X += [[noise_dataset[0][i], noise_dataset[1][i]]]

    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)


    # middle_X = []
    # middle_Y = []
    for neighbor_indices in indices:
        for i in range(len(neighbor_indices)):
            if i == 0:
                continue
            # print X[neighbor_indices[0]], X[neighbor_indices[i]]
            P1 = [X[neighbor_indices[0]][0], X[neighbor_indices[0]][1]]
            P2 = [X[neighbor_indices[i]][0], X[neighbor_indices[i]][1]]
            dist_vector = [P2[0] - P1[0], P2[1] - P1[1]]
            dist_vector = [0.5 * dist_vector[0], 0.5 * dist_vector[1]]
            middle = [P1[0] + dist_vector[0], P1[1] + dist_vector[1]]

            # middle_X += [middle[0]]
            # middle_Y += [middle[1]]

            density = normal_distribution_3d(middle[0], middle[1], mus[0], sigmas[0])
            density += normal_distribution_3d(middle[0], middle[1], mus[1], sigmas[1])

            if density >= density_threshold:
                ax.plot([X[neighbor_indices[0]][0], X[neighbor_indices[i]][0]], [X[neighbor_indices[0]][1], X[neighbor_indices[i]][1]], color='y', zorder=1)

    # ax.scatter(middle_X, middle_Y, c='y')

# # ax.view_init(elev=90, azim=5)
# plt.show()
fig.savefig("clustering_4.png", dpi=300)

plt.close()
