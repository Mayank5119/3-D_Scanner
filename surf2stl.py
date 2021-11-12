# df
import numpy as np
from stl import mesh


def surf2stl(filename,x,y,z):

    triangles = []

    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):

            p1 = np.array([x[i, j], y[i, j], z[i, j]])
            p2 = np.array([x[i, j + 1],   y[i, j + 1],   z[i, j + 1]])
            p3 = np.array([x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]])

            triangles.append([p1, p2, p3])

            p1 = np.array([x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1]])
            p2 = np.array([x[i + 1, j],   y[i + 1, j],   z[i + 1, j]])
            p3 = np.array([x[i, j],     y[i, j],     z[i, j]])
            triangles.append([p1, p2, p3])

    num_triangles = len(triangles)
    data = np.zeros(num_triangles, dtype=mesh.Mesh.dtype)
    for i in range(num_triangles):
        data["vectors"][i] = np.array(triangles[i])
    m = mesh.Mesh(data)
    m.save(filename+'.stl')
    return

def local_find_normal(p1,p2,p3):
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = np.cross(v1, v2)
    # print("np.sqrt((np.multiply(v3, v3).sum(axis=0)",np.sqrt((np.multiply(v3, v3).sum(axis=0))))
    n = np.divide(v3, np.sqrt((np.multiply(v3, v3).sum(axis=0))))
    return n


