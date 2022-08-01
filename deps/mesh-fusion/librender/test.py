import pyrender
import numpy as np
from matplotlib import pyplot
import math

# render settings
img_h = 480
img_w = 480
fx = 480.
fy = 480.
cx = 240
cy = 240

def model():

    # note that xx is height here!
    xx = -0.2
    yy = -0.2
    zz = -0.2

    v000 = (xx, yy, zz)  # 0
    v001 = (xx, yy, zz + 0.4)  # 1
    v010 = (xx, yy + 0.4, zz)  # 2
    v011 = (xx, yy + 0.4,  zz + 0.4)  # 3
    v100 = (xx + 0.4, yy, zz)  # 4
    v101 = (xx + 0.4, yy, zz + 0.4)  # 5
    v110 = (xx + 0.4, yy + 0.4, zz)  # 6
    v111 = (xx + 0.4, yy + 0.4, zz + 0.4)  # 7

    f1 = [0, 2, 4]
    f2 = [4, 2, 6]
    f3 = [1, 3, 5]
    f4 = [5, 3, 7]
    f5 = [0, 1, 2]
    f6 = [1, 3, 2]
    f7 = [4, 5, 7]
    f8 = [4, 7, 6]
    f9 = [4, 0, 1]
    f10 = [4, 5, 1]
    f11 = [2, 3, 6]
    f12 = [3, 7, 6]

    vertices = [v000, v001, v010, v011, v100, v101, v110, v111]
    faces = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
    return vertices, faces

def render(vertices, faces):

    x = 0
    y = math.pi/4
    z = 0
    R_x = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    R_y = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    R_z = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    R = R_z.dot(R_y.dot(R_x))

    np_vertices = np.array(vertices).astype(np.float64)
    np_vertices = R.dot(np_vertices.T).T
    np_vertices[:, 2] += 1.5
    np_faces = np.array(faces).astype(np.float64)
    np_faces += 1

    depthmap, mask, img = pyrender.render(np_vertices.T.copy(), np_faces.T.copy(), np.array([fx, fy, cx, cy]), np.array([1., 2.]), np.array([img_h, img_w], dtype=np.int32))
    pyplot.imshow(depthmap)
    pyplot.show()
    pyplot.imshow(img)
    pyplot.show()

if __name__ == '__main__':
    vertices, faces = model()
    render(vertices, faces)
