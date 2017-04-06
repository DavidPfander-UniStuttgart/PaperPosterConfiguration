#!/usr/bin/python

import time
import re
import subprocess
import shlex
import math
import numpy as np
import matplotlib as mpl
# mpl.use('svg')
mpl.use('agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 32}

# plt.rc('font', **font)
plt.rcParams['font.family'] = "Liberation Sans"
plt.rcParams['font.size'] = 32

fig, axarr = plt.subplots(2, 2) # , sharey=True
print axarr
fig.set_figheight(15)
fig.set_figwidth(17)

def normal_distribution_3d(x, y, mu, sigma): # mean, standard deviation
    p = [x, y]
    result = 1.0
    for i in range(2):
        first_factor = 1.0/math.sqrt(2*math.pi*sigma[i])
        exponent = (- (p[i] - mu[i])*(p[i] - mu[i])) / (2 * sigma[i] * sigma[i])
        result *= first_factor * np.exp(exponent)
    return result

level = 8
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
print "sum_of_results"
print sum_of_results

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

overall_dataset = []
for d in datasets:
    for i in range(len(d[0])):
        overall_dataset += [[d[0][i], d[1][i]]]

if add_noise:
    for i in range(len(noise_dataset[0])):
        overall_dataset += [[noise_dataset[0][i], noise_dataset[1][i]]]

knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(overall_dataset)
neighbor_distances, all_neighbor_indices = knn.kneighbors(overall_dataset)

middle_evaluation_points = []
for neighbor_indices in all_neighbor_indices:
    for i in range(len(neighbor_indices)):
        if i == 0:
            continue
        # print overall_dataset[neighbor_indices[0]], overall_dataset[neighbor_indices[i]]
        P1 = [overall_dataset[neighbor_indices[0]][0], overall_dataset[neighbor_indices[0]][1]]
        P2 = [overall_dataset[neighbor_indices[i]][0], overall_dataset[neighbor_indices[i]][1]]
        dist_vector = [P2[0] - P1[0], P2[1] - P1[1]]
        dist_vector = [0.5 * dist_vector[0], 0.5 * dist_vector[1]]
        middle = [P1[0] + dist_vector[0], P1[1] + dist_vector[1]]

        middle_evaluation_points += [[P1, P2, middle, None]] # includes placeholder for evaluated density

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

# write density evaluation points (middle points)
f = open('middle_points_dataset.arff', 'w')
f.write("""@RELATION "middle_points_dataset.arff"

@ATTRIBUTE x0 NUMERIC
@ATTRIBUTE x1 NUMERIC

@DATA
""")
# @ATTRIBUTE class NUMERIC
for middle_evaluation_tuple in middle_evaluation_points:
    middle = middle_evaluation_tuple[2]
    f.write(str(middle[0]) + "," + str(middle[1]) + "\n")
f.close()

# cmd = "clustering_poster --datasetFileName dataset.arff --middlePointsFileName middle_points_dataset.arff --eval_grid_level=" + str(level) + " --level=9 --lambda=0.05"
cmd = "clustering_poster --datasetFileName dataset.arff --middlePointsFileName middle_points_dataset.arff --eval_grid_level=" + str(level) + " --level=8 --lambda=0.15"
print cmd
args = shlex.split(cmd)
print args
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False) #, shell=True , stdout=subprocess.PIPE, shell=0
# p.wait()
# time.sleep(15)
output_streams = p.communicate()
stdout = output_streams[0]
stderr = output_streams[1]

# print output
m = re.search(r"\[(.*?)\].*?\[(.*?)\]", stdout, re.MULTILINE | re.DOTALL)
print "stdout"
print stdout
print "stderr"
print stderr
print "grid evals"
print m.group(1)
evals = m.group(1).split(",")
evals = [float(e) for e in evals]
print evals
print "middle evals"
middle_evals = m.group(2).split(",")
middle_evals = [float(e) for e in middle_evals]
print middle_evals
print len(middle_evals)
print len(middle_evaluation_points)
for i in range(len(middle_evaluation_points)):
    middle_evaluation_tuple = middle_evaluation_points[i]
    middle_evaluation_tuple[3] = middle_evals[i]

sum_of_results = []
outer_index = 0
for i in range(2**level):
    inner = []
    for j in range(2**level):
        inner += [evals[outer_index]]
        outer_index += 1
    sum_of_results += [inner]

# for middle_evaluation_tuple in middle_evaluation_points:
#     middle = middle_evaluation_tuple[2]
#     density = normal_distribution_3d(middle[0], middle[1], mus[0], sigmas[0])
#     density += normal_distribution_3d(middle[0], middle[1], mus[1], sigmas[1])
#     middle_evaluation_tuple[3] = density

# colors = ["r", "g"]
# markers = ["o", "^"]
colors = ["r", "r", "r"]
markers = ["o", "o", "o"]

##################################################
# Plot dataset
##################################################

# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)

ax = axarr[0][0]

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])
# ax.set_zlim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

ax.set_xlabel('The dataset')

# # ax.view_init(elev=90, azim=5)
# plt.show()
# fig.savefig("clustering_1.svg", dpi=80)

# plt.close()

##################################################
# Estimate density
##################################################

# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)

ax = axarr[0][1]

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

im = ax.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='hot', aspect='auto') # cmap='hot', , interpolation='nearest' 'viridis'
# cbar = fig.colorbar(im)

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

# if add_noise:
#     ax.scatter(noise_dataset[0], noise_dataset[1], c='b')

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

ax.set_xlabel('1) Approximate density')

# # ax.view_init(elev=90, azim=5)
# plt.show()

# fig.savefig("clustering_2.svg", dpi=80)

# plt.close()

##################################################
# Now plot with knn
##################################################

# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
ax = axarr[1][0]

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

ax.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='hot', aspect='auto') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)

for middle_evaluation_tuple in middle_evaluation_points:
    P1 = middle_evaluation_tuple[0]
    P2 = middle_evaluation_tuple[1]
    density = middle_evaluation_tuple[3]

    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color='y', zorder=1)

# # ax.view_init(elev=90, azim=5)
# plt.show()
# fig.savefig("clustering_3.svg", dpi=80)

ax.set_xlabel('2) Create k-nearest-neighbor graph')

# plt.close()


##################################################
# Now plot clusters
##################################################

# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)

ax = axarr[1][1]

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

plt.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='hot', aspect='auto') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

for i in range(2):
    c=colors[i]
    m=markers[i]
    ax.scatter(datasets[i][0], datasets[i][1], c=c, marker=m, zorder=2)

ax.scatter(noise_dataset[0], noise_dataset[1], c=colors[2], zorder=2)


density_threshold = 0.75

X = []
for d in datasets:
    for i in range(len(d[0])):
        X += [[d[0][i], d[1][i]]]

if add_noise:
    for i in range(len(noise_dataset[0])):
        X += [[noise_dataset[0][i], noise_dataset[1][i]]]

for middle_evaluation_tuple in middle_evaluation_points:
    P1 = middle_evaluation_tuple[0]
    P2 = middle_evaluation_tuple[1]
    density = middle_evaluation_tuple[3]

    if density >= density_threshold:
        ax.plot([P1[0], P2[0]], [P1[1], P2[1]], color='y', zorder=1)

ax.set_xlabel('3) Remove low-density edges')

# # ax.view_init(elev=90, azim=5)
# plt.show()
# fig.savefig("clustering_4.svg", dpi=80)

# plt.close()

plt.tight_layout()

fig.savefig("all_in_one.png", bbox_inches='tight', dpi=300)
