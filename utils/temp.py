import numpy as np
temp = [[0,0,0,3,0,6,0,0],
        [0,0,0,0,0,0,6,0],
        [0,2,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [6,0,0,0,0,0,0,0],
        [0,2,6,0,0,0,0,0],
        [0,0,0,0,0,0,0,6],
        [0,2,0,3,6,0,0,0]
        ]

temp = np.array(temp)

init = np.array([1,1,1,1,1,1,1,1])

init = np.reshape(init,[-1,1])

div1 = 48
div2 = 48 * 6

v1 = np.matmul(temp, init)
print("version1",v1)
print(np.sum(v1))
v2 = np.matmul(temp, v1)
print("version2", v2)
print(np.sum(v2))
