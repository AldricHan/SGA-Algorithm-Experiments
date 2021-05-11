import numpy as np
import random as rand
import networkx as nx

def getF(samples):
    num_samples, num_nodes = samples.shape
    F = np.zeros((num_nodes,num_nodes,4))
    for i in range(num_nodes):
        for j in range(i,num_nodes):
            if i == j:
                col = samples[:,i]
                i0 = np.where(col==0)[0]
                i1 = np.where(col==1)[0]
                F[i,i] = [len(i0), len(i1), 0, 0]
                continue
            f = [0,0,0,0]
            colI = samples[:,i]
            colJ = samples[:,j]
            arrI = [np.where(colI==0)[0], np.where(colI==1)[0]]
            arrJ = [np.where(colJ==0)[0], np.where(colJ==1)[0]]
            for b in range(4):
                binary = format(b,'002b')
                tI = arrI[int(binary[0])]
                tJ = arrJ[int(binary[1])]
                f[b] = len(np.intersect1d(tI,tJ))
            F[i,j] = f
    return F


def getTree(samples):
    num_samples, num_nodes = samples.shape
    F = getF(samples)/num_samples
    I = np.zeros((num_nodes,num_nodes))
    fufv = np.zeros(F.shape)
    col1 = np.diagonal(F).T
    for i in range(num_nodes):
        temp = np.zeros((4, num_nodes))
        for k in range(0,4,2):
            temp[k] = col1[:,0]*col1[i,k//2]
            temp[k+1] = col1[:,1]*col1[i,k//2]
        fufv[i] = temp.T
    t1 = F/fufv
    t1 = np.log(t1)
    t1 = F*t1
    I = np.nan_to_num(np.sum(t1, axis = 2))

    res = kruskals(I)


    return res

def root(i,sets):
    parent = sets[i]
    while parent != sets[parent]:
        parent = sets[parent]
    return parent

def isSameSet(i,j,sets):
    return root(i,sets) == root(j,sets)

def union(i,j, sets):
    sets[root(j,sets)] = i
    return sets


def kruskals(I):
    num_nodes = I.shape[0]
    edges = np.zeros(((num_nodes-1)*(num_nodes)//2,3))#, dtype = dtype)
    G = nx.Graph()
    sets = list(range(num_nodes))

    idx = 0
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            edges[idx] = [i,j,I[i,j]]
            idx += 1
            
    while edges.size:
        curr = np.max(edges[:,2])
        idx = rand.choice(np.where(edges[:,2] == curr)[0])
        j,k,weight = edges[idx]
        j = int(j)
        k = int(k)
        if isSameSet(j,k,sets):
            edges = np.delete(edges, idx, 0)
            pass
        else:
            G.add_edge(j,k, weight = weight)
            edges = np.delete(edges, idx, 0)
            sets = union(j,k, sets)
        
    return G