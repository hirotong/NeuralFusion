'''
Author: Jinguang Tong
Affliction: Australia National University, DATA61, Black Mountain
'''

# from mayavi import mlab
import numpy as np
import h5py
import mcubes
# from graphics.visualization import plot_tsdf, plot_mesh
from graphics.utils import extract_mesh_marching_cubes
from glob import glob
import os
import math
import pyrender
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_spherical_pose(theta, phi, radius): 
    trans_t = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, radius],
        [0, 0, 0, 1],
    ])

    rot_phi = np.array([
        [1, 0, 0, 0],
        [0, math.cos(phi), -math.sin(phi), 0],
        [0, math.sin(phi), math.cos(phi), 0],
        [0, 0, 0, 1]
    ])

    rot_theta = np.array([
        [math.cos(theta), 0, -math.sin(theta), 0],
        [0, 1, 0, 0],
        [math.sin(theta), 0, math.cos(theta), 0],
        [0, 0, 0, 1]
    ])

    c2w = rot_theta @ rot_phi @ trans_t

    return c2w


def load_tsdf_file(path):
    
    h5f = h5py.File(path, 'r')
    for key in h5f.keys():
        print(key)
    tsdf = h5f['TSDF'][()]
    print(tsdf.data)
    h5f.close()
    
    return tsdf 

if __name__ == '__main__':
    files = glob("experiments/210803-170452/output/*.tsdf.hf5")
    render_intrinsics = np.array([640, 640, 320, 320], dtype=float)
    image_size = np.array([480, 640] , dtype=np.int32)
    for file in files:
        tsdf = load_tsdf_file(file)
        # print(tsdf)
        # mesh = extract_mesh_marching_cubes(tsdf)
        vertex, faces = mcubes.marching_cubes(tsdf, -1e-7)
        ims = []
        fig = plt.figure()
        for theta in np.arange(0, 2 * math.pi, 2 * math.pi / 100):
            c2w = get_spherical_pose(theta, 0, 10)
            c2w = c2w[:3, :]
            np_vertices = c2w[:3, :3].dot(vertex.T).T
            np_vertices += c2w[:, 3]
            
            depthmap, mask, img = pyrender.render(np_vertices.T.copy(), faces.T.copy(), render_intrinsics, np.array([1., 2.]), image_size)
            im = plt.imshow(img)
            ims.append(im)
        ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
        ani.save('test.gif', writer='pillow')



        print(vertex.shape)
        # mesh.write('tsdf/' + file.split('/')[-1].replace('hf5', 'ply'))
        # tsdf = tsdf[::4, ::4, ::4]
    # plot_tsdf(tsdf)
    # plot_mesh(mesh)
        