import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_from_csv(filename):
	ret_val = []
	with open(filename, newline='') as csvfile:
		for row in csv.reader(csvfile):
			feature_x = []
			for correspond in row:
				feature_x.append(float(correspond))
			ret_val.append(feature_x)
	return ret_val


x_correspondance = np.array(np.matrix(read_from_csv("midtermdata x.csv")).T)
y_correspondance = np.array(np.matrix(read_from_csv("midtermdata y.csv")).T)
x_0 = []
y_0 = []
for each_camera in range(len(x_correspondance)):
	x_0.append(np.mean(x_correspondance[each_camera]))
	y_0.append(np.mean(y_correspondance[each_camera]))


for each_camera in range(len(x_correspondance)):
	for each_point in range(len(x_correspondance[each_camera])):
		x_correspondance[each_camera][each_point] -= x_0[each_camera]
		y_correspondance[each_camera][each_point] -= y_0[each_camera]

W = np.vstack((x_correspondance, y_correspondance))

U, D, V = np.linalg.svd(W)
U_dot = U[:,0:3]
D_dot = np.sqrt(D[0:3] * np.identity(3))
V_dot = V.T[:,:3]
M_i = np.dot(U_dot, D_dot)
'''
for i in range(int(len(M_i) / 2)):
	print("Camera motion ", i, ": ")
	print(np.vstack((M_i[i], M_i[i + 10], [0, 0, 0])))
'''
S = np.dot(D_dot, V_dot.T)


x_s =S[0,]
y_s =S[1,]
z_s =S[2,]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=x_s, ys=y_s, zs=z_s)
plt.show()