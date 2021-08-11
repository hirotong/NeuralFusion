'''
Author: Jinguang Tong
Affliction: Australia National University, DATA61, Black Mountain
'''

import numpy as np
import h5py
from mayavi import mlab
from graphics.visualization import plot_tsdf, plot_mesh
from graphics.utils import extract_mesh_marching_cubes

def load_tsdf_file(path):
    
    h5f = h5py.File(path, 'r')
    for key in h5f.keys():
        print(key)
    tsdf = h5f['TSDF'][()]
    print(tsdf.data)
    h5f.close()
    
    return tsdf 

if __name__ == '__main__':
    tsdf = load_tsdf_file("experiments/210803-170452/output/02691156.b7fd11d4af74b4ffddaa0161e9d3dfac.tsdf.hf5")
    print(tsdf)
    mesh = extract_mesh_marching_cubes(tsdf)
    # tsdf = tsdf[::4, ::4, ::4]
    tsdf[tsdf == 0.1] = 0
    print(tsdf)
    plot_tsdf(tsdf)
    plot_mesh(mesh)
        