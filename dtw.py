from math import *
import numpy as np
import sys

def DTW(A, B, window = sys.maxsize, d = lambda x,y: abs(x-y)):
    # create the cost matrix
    A, B = np.array(A), np.array(B)
    M, N = len(A), len(B)
    cost = sys.maxsize * np.ones((M, N))
    ##
    '''
    for i in range(0,M):
        print(A[i]," ",end='')
    print("\n")

    for i in range(0,N):
        print(B[i]," ",end='')
    print("\n")
    '''
    ##
    # initialize the first row and column
    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(A[0], B[j])
    # fill in the rest of the matrix
    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    ##
    '''
    for i in range(0,M):
        for j in range(0,N):
            print(cost[i][j]," ",end='')
        print('\n')
    ##
    '''
    # find the optimal path
    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
        #print("m:",m," n:",n)

    path.append((0,0))
    return cost[-1, -1], path

def main():
    A = np.array([1,2,3,4,2,3,1,5,3,4,2,4,3])
    B = np.array([70,80,50,90,110,90,20,30,25,45,79,57,30])
    for i in range(0,len(B)):
        B[i] = B[i] / 10
    #A = [1,2,3,4,2,3]
    #B = [7,8,5,9,11,9,2,3]
    cost, path = DTW(A, B, window = 100)
    print('Total Distance is ', cost)
    import matplotlib.pyplot as plt
    offset = 5
    #plt.xlim([-1, max(len(A), len(B)) + 1])
    plt.plot(A)
    plt.plot(B + offset)
    for (x1, x2) in path:
        plt.plot([x1, x2], [A[x1], B[x2] + offset])
    plt.show()

if __name__ == '__main__':
    main()
