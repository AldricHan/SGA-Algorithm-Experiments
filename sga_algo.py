import numpy.random as rand
import numpy as np
import numpy.linalg as la
import timeit
import time
from numpy.linalg import inv
from tempfile import TemporaryFile
import scipy.stats as stats
import math
import random
from math import sqrt

def find_EC(emp_cov, root, prox1, prox2, subtree, eps):
    EC = []
    if (len(subtree) <= 2):
        return subtree
    common_nodes_big = list(set(prox1[root])& set(subtree))
    
    for i in common_nodes_big:
        is_leaf = True
        common_nodes_root_i = list(set(prox2[root])& set(prox2[i]) & set(subtree))
        for k1 in common_nodes_root_i:
            sig_i_k1 = emp_cov[i,k1]
            sig_root_k1 = emp_cov[root, k1]
            common_nodes = list(set(prox2[root])& set(prox2[i]) & set(prox2[k1]) & set(subtree))
            for k2 in common_nodes:
                #Here is the full check that could be done with "is_non_star"
                non_star, pair1, pair2 = is_non_star(emp_cov, [root,i,k1,k2], eps)
                pair1 = set(pair1)
                pair2 = set(pair2)
                if {i,k2} in [pair1,pair2] or {i,k1} in [pair1,pair2]:
                    is_leaf = False
        if is_leaf:
            EC.append(i)       
    return EC



def star_helper(emp_corr, nodes1, nodes2, nodes3 ,nodes4):
    return sqrt(emp_corr[nodes1][nodes3]*emp_corr[nodes2][nodes4]*emp_corr[nodes1][nodes4]*emp_corr[nodes2][nodes3])/(emp_corr[nodes1][nodes2]*emp_corr[nodes3][nodes4])

def is_non_star(emp_corr, nodes, eps):
    v1 = star_helper(emp_corr, nodes[0], nodes[1], nodes[2], nodes[3])
    v2 = star_helper(emp_corr, nodes[0], nodes[2], nodes[1], nodes[3])
    v3 = star_helper(emp_corr, nodes[0], nodes[3], nodes[1], nodes[2])
    # lst = [v1,v2,v3]
    if v1 < eps or v2 < eps or v3 < eps:
        outputNodes = [0]*3#np.zeros((3,2,2))
        outputNodes[0] = [[nodes[2],nodes[3]], [nodes[0],nodes[1]]]
        outputNodes[1] = [[nodes[1],nodes[3]], [nodes[0],nodes[2]]]
        outputNodes[2] = [[nodes[1],nodes[2]], [nodes[0],nodes[3]]]
        if v1 < v2 and v1 < v3:
            outputNode = outputNodes[0] #[nodes[2],nodes[3]], [nodes[0],nodes[1]]
        elif v2 < v3 and v2 < v1:
            outputNode = outputNodes[1]
        elif v3 < v2 and v3 < v1:
            outputNode = outputNodes[2]
        elif v1 == v2 and v2 == v3:
            outputNode = outputNodes[rand.randint(0,3)]
        elif v1 == v2:
            outputNode = outputNodes[rand.randint(0,2)]
        elif v1 == v3:
            outputNode = outputNodes[random.choice([0,2])]
        else:
            outputNode = outputNodes[random.choice([1,2])]
        return True, outputNode[0], outputNode[1]
    else:
        return False, [], []



def merge_splits(split1, split2):
    len_common_r_e = len(split1)
    merged = [-1]*len_common_r_e
    for i in range(len_common_r_e):
        if split1[i] == 1 or split2[i] == 1:
            merged[i] = 1
    return merged      


def split_subtree(emp_cov, root, ext, prox1, prox2, prohibited, eps):
    common_root_ext = list(set(prox1[root]) & set(prox1[ext]) )
    len_common_r_e = len(common_root_ext)
    split = [[-1]*len_common_r_e for i in range(len_common_r_e)]
    
    #1 in split[i][j] indicates that i and j are in the same subtree, -1 means they are not
    for i in range(len_common_r_e):
        split[i][i] = 1
    
    for i in range(len_common_r_e):
        node_i = common_root_ext[i]
        if node_i in prohibited: 
            split[i][i] = -1
            continue      
        for j in range(len_common_r_e):
            node_j = common_root_ext[j]
            if node_j == node_i or node_j not in prox2[node_i]:
                continue
            nodes = [root, ext, node_i, node_j]
            #note here, if (X1, X2) form a pair in a non-star, then they are in the same connected component
            status, pair1, pair2 = is_non_star(emp_cov, nodes, eps)
            if status and root in pair2 and ext in pair2:
                if node_j in prohibited:
                    #this means that j and i are in the same subtree, but j is prohibited, so i is in the prohibited component
                    split[i][:] = [-1]*len_common_r_e
                    break
                split[i][j] = 1

    deleted = []
    #Code for merging 
    for i in range(len_common_r_e):
        first = 1
        for j in range(len_common_r_e):
            if j in deleted:
                continue
            if split[j][i] == 1: #to find if i is in same subtree as j
                if first == 1: #collect first j, ind is the main list that the subtree is collected in
                    first = 0
                    ind = j
                else: 
                    split[ind] = merge_splits(split[ind], split[j])
                    deleted.append(j)   
    subtrees = []
    for j in range(len_common_r_e):
        if j in deleted:
            continue
        subtree_j = []
        for k in range(len_common_r_e):
            if split[j][k] == 1:               
                subtree_j.append(common_root_ext[k])
        subtrees.append(subtree_j)
    # print(subtrees,deleted)
    return subtrees


def alg_init(emp_cov, prox1, prox2, eps, edges, prohibited):
    n = emp_cov.shape[0]
    subtree = range(n)
    for i in range(n):
        EC_init = find_EC(emp_cov, i, prox1, prox2, subtree, eps)
        len_EC_init = len(EC_init)
        if len_EC_init > 0:
            prohibited.append(i)
            for j in EC_init:
                edges.add(frozenset([i,j]))
                prohibited.append(j)
            break
    return i, EC_init, edges


def full_alg_recurse(emp_cov, prox1, prox2, eps, edges, xi, xext, prohibited):
    
    sub_subtrees = split_subtree(emp_cov, xext, xi, prox1, prox2, prohibited, eps)
    for i in sub_subtrees:
        for j in i:
            prohibited.append(j)
    num_sub_subtrees = len(sub_subtrees)  
    for i in range(num_sub_subtrees):
        local_prohibit = [-1]*len(prohibited)
        local_prohibit[:] = prohibited[:]
        for j in sub_subtrees[i]:
            local_prohibit.remove(j)
        EC = find_EC(emp_cov, xi, prox1, prox2, sub_subtrees[i], eps)
        len_EC = len(EC)
        
        for j in range(1, len_EC): 
            edges.add(frozenset([EC[j], EC[0]]))
            # local_prohibit.append(EC[j])
        if len_EC>0:
            edges.add(frozenset([xi, EC[0]])) 
            local_prohibit.append(EC[0])
            full_alg_recurse(emp_cov, prox1, prox2, eps, edges, EC[0], xi, local_prohibit)


def get_prox(emp_corr, thres1, thres2):
    prox1 = []
    prox2 = []
    n = emp_corr.shape[0]
    for i in range(n):
        prox1_i = []
        prox2_i = []
        for j in range(n):
            if j == i:
                continue
            if abs(emp_corr[i,j])>thres1:
                prox1_i.append(j)
            if abs(emp_corr[i,j])>thres2:
                prox2_i.append(j)
        prox1.append(prox1_i)
        prox2.append(prox2_i)
    return prox1, prox2


def find_tree(emp_corr, prox1, prox2, rho_max):
    n = emp_corr.shape[0]
    edges = set()
    eps = (1+rho_max**2)/2
    prohibited = []
    root, EC_init, edges = alg_init(emp_corr, prox1, prox2, eps, edges, prohibited)
    error = 0
    if len(EC_init) == 0:
        return [], -1
    subtree = range(n)
    subtree = list(set(subtree) - set(EC_init))
    full_alg_recurse(emp_corr, prox1, prox2, eps, edges, root, EC_init[0], prohibited)
    return edges, error
 

def deduplicate_edges(edges):
    for i in edges:
        for j in edges:
            if i==j:
                continue
            if set(i) == set(j):
                edges.remove(j)
    return edges