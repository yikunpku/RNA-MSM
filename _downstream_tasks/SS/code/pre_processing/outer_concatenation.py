import numpy as np
'''
def Outer_concatenation(matrix_A):
    "l1xn--> l1xl1x2n"
    matrix_B=np.zeros((matrix_A.shape[0],matrix_A.shape[0],None))
    print(matrix_B.shape)
    for v1_dx, v1 in enumerate(matrix_A):
        # print(v1)
        for v2_dx, v2 in enumerate(matrix_A):
            v12 = np.hstack([v1, v2])
            # print(v12)
            matrix_B[v1_dx, v2_dx] = v12
            print(matrix_B.shape)
    return matrix_B
'''

def matrix_concatenation(matrix_A,matrix_B):
    matrix_A=matrix_A
    matrix_B=matrix_B
    if matrix_A.shape[0]==matrix_B.shape[0] and matrix_A.shape[1]==matrix_B.shape[1]:
        matrix_C=np.concatenate([matrix_A,matrix_B],axis=2)
    return matrix_C


def outer_concatenation(matrix_A,matrix_B):
    "l1xn--> l1xl1x2n"
    matrix_A=matrix_A
    matrix_B=matrix_B
    matrix_C=np.zeros((matrix_A.shape[0],matrix_B.shape[0],matrix_A.shape[1]+matrix_B.shape[1]))
    #print(matrix_C.shape)
    for v1_dx, v1 in enumerate(matrix_A):
        for v2_dx, v2 in enumerate(matrix_B):
            v12 = np.hstack([v1, v2])
            # print(v12)
            matrix_C[v1_dx, v2_dx] = v12
            #print(matrix_C.shape)
    return matrix_C