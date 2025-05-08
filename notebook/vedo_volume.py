"""Create a Volume from a numpy array"""
import numpy as np
from vedo import Volume, show

data_matrix = np.zeros([70, 80, 90], dtype=np.uint8)
data_matrix[ 0:2,  0:2,  0:1] = 1
# data_matrix[ 1:2,  1:2,  1:2] = 3
# data_matrix[30:50, 30:60, 30:70] = 2
# data_matrix[50:70, 60:80, 70:90] = 3
# 创建一个 2,2,2 的体素数据矩阵
data_matrix = np.array([
    [[0, 0],
     [0, 0]],
    [[1, 1],
     [1, 1]],
   ]
, dtype=np.uint8)


vol = Volume(data_matrix)
# vol.cmap(['white','b','g','r']).mode(1)
vol.add_scalarbar()

print(vol.pos())
print(vol.center())

show(vol, __doc__, axes=1).close()