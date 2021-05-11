import numpy as np
import math
import numpy.random as rand
from numpy.random import Generator, PCG64
from collections import deque
import networkx as nx
import sga_algo 
import k_algo_extended as k_algo
import ChowLiu as CL

rg = Generator(PCG64())

'''
get_children, get_ECs, makeTree, makeEdges, generate_equivalence_class, genSigma and getStatistics are utilities for running the experiment.
parallelTestFunc and parallelGaussianTest are used for running the simulations.

'''
def get_children(tree):
    '''
    tree: tree as a disjoint set. array A, A[i] is parent of node i, root node has itself as the parent
    returns: list of lists A, where A[i] is the child nodes of node i
    '''
    tree = np.asarray(tree)
    child = [np.where(tree == i)[0].tolist() for i in range(len(tree))]
    children = [0]*len(tree)
    for i in range(len(child)):
        temp = []
        for j in child[i]:
            if j == i:
                continue
            temp.append(j)
        children[i] = temp
    return children


def get_ECs(edges):
    '''
    edges: List of lists or tuples. Edge set of a tree.
    returns: Equivalence classes as frozensets
    '''
    tree = makeTree(edges)
    tree = np.asarray(tree)
    num_nodes = len(tree)
    children = np.asarray(get_children(tree))

    #list of list of ecs
    ECs = []

    #identify leaf nodes, nodes attached to leaves and internal nodes
    root_node = 0
    for i in range(num_nodes):
        if tree[i] == i:
            root_node = i
    root_is_leaf = False

    leaf_nodes = np.where(np.logical_not(children))[0]
    leaf_roots = np.unique(tree[leaf_nodes])

    internal_nodes = np.delete(list(range(num_nodes)), np.append(leaf_nodes, leaf_roots))
    if len(children[root_node]) == 1:
        leaf_roots = np.append(leaf_roots, children[root_node][0])
        leaf_nodes = np.append(leaf_nodes, root_node)
        internal_nodes = np.delete(internal_nodes, [root_node,children[root_node][0]])
        root_is_leaf = True

    leaf_roots = list(set(leaf_roots))
    #ECs are tuples (root, [nodes in the EC])
    for i in range(len(leaf_roots)):
        root = leaf_roots[i]
        EC = np.intersect1d(children[root], leaf_nodes)
        if root_is_leaf and tree[root] == root_node:
            EC = np.append(EC,[root_node])
        EC = np.append(EC, root)
        ECs += [list(set(EC.tolist()))]

    return ECs

def makeTree(edges):
    '''
    edges: List of lists or tuples. Edge set of tree. Tree must have nodes numbered 0 to n-1 for n node tree.
    returns: array of tree as a UFDS
    '''
    root = 0
    tree = [0]*(len(edges)+1)
    queue = deque()
    queue.append(root)
    while queue:
        curr = queue.popleft()
        temp = [i for i in edges]
        for edge in edges:
            temp1 = [i for i in edge]
            if curr in temp1:
                temp1.remove(curr)
                tree[temp1[0]] = curr
                queue.append(temp1[0])
                temp.remove(edge)
        edges = temp
    return tree

def makeEdges(tree):
    '''
    Tree: List of integers. UFDS representation of a tree
    returns: List of edges(as tuples)
    '''
    edges = []
    for i in range(len(tree)):
        if i == tree[i]:
            continue
        edges += [sorted((i,tree[i]))]
    return edges

def generate_equivalence_class(edges):
    '''
    edges: List of doubles. Edge set of the tree
    returns: List of Sets of frozensets. A set of trees represented by edge sets within the equivalence class of the tree with the given edge set.
    '''
    ogTree = nx.Graph()
    ogTree.add_edges_from(edges)

    ECs = get_ECs(edges)
    mappings = [] #list of dictionaries
    for EC in ECs:
        for i in range(1,len(EC)):
            mapping = {}
            for n in range(len(EC)):
                mapping[EC[n]] = EC[(i+n)%len(EC)]
            mappings += [mapping]

    eqClass = [ogTree]
    counter = 0
    for i in range(len(ECs)):
        EC = ECs[i]
        n = len(EC)-1
        tempClass = [i.copy() for i in eqClass]
        for tree in tempClass:
            for mapping in mappings[counter:counter+n]:
                newTree = nx.relabel_nodes(tree, mapping, copy = True)
                # print(newTree.edges)
                eqClass.append(newTree)
        counter += n

    equivalence_class = [edges_as_set_of_sets(list(g.edges)) for g in eqClass]
    
    return equivalence_class


def edges_as_set_of_sets(edges):
    '''
    edges: List of lists. Edge set of the tree
    returns: Set of frozensets. edge set of the tree.
    '''
    edges_set = set([frozenset(i) for i in edges])
    return edges_set



def generate_n_tree(num_samples, t, q, noisy_nodes, tree):
    '''
    num_samples: number of samples
    t: theta, influence between nodes where t[i,j] = (1-rho[i,j])/2
    q: maximum crossover probability of a noisy node
    noisy_nodes: list of noisy nodes
    tree: tree as a disjoint set. array A, A[i] is parent of node i, root node has itself as the parent
    returns: np.array of shape (num_samples, len(tree)), where rows are a sample of the distribution, columns are the variables
    '''
    # tree = makeTree(edges)
    tree = np.asarray(tree)
    children = get_children(tree)

    #find the root of the tree
    root = 0
    for i in range(len(tree)):
        if tree[i] == i:
            root = i
            break
    
    #initialize samples
    samples = np.zeros((num_samples, len(tree)))
    samples[:,root] = np.random.choice([-1,1], size = (num_samples,))

    #do a bfs to generate samples
    discovered = [root]
    queue = deque() 
    queue.append(root)
    while queue:
        v = queue.popleft()
        positiveidx = np.where(samples[:,v] == 1)
        negativeidx = np.where(samples[:,v] == -1)
        child = children[v]
        if len(child) != 0:
            temp1 = rg.choice([1,-1],size = (positiveidx[0].shape[0], len(child)), p=[1-t,t])
            temp2 = rg.choice([1,-1],size = (negativeidx[0].shape[0], len(child)), p=[t,1-t])

            #build column
            temp = np.zeros((num_samples, len(child)))
            temp[positiveidx] = temp1
            temp[negativeidx] = temp2

            samples[:, child] = temp
            
            queue += child
    

    if q != 0:
        temp = rg.choice([1,-1],size = (num_samples, len(noisy_nodes)), p=[1-q, q])
        samples[:,noisy_nodes] = np.multiply(samples[:,noisy_nodes],temp)
    
    return samples

'''
generate_n_prob provides an alternative, possibly more efficient method to generate samples from an Ising model.
'''

def generate_n_prob(num_samples, prob):
    d = int(np.log2(len(prob)))
    samples = rand.choice(2**d, size = (num_samples,), replace = True, p = prob)
    samples = unpackbits(samples, d)*2-1
    return samples

def gen_chain_W(t, tree):
    '''
    t: theta, influence between nodes
    tree: Array where tree is represented as a UFDS
    returns: weight matrix W to generate a probability distribution
    '''
    num_nodes = len(tree)
    w = math.atanh(1-2*t)
    W = np.zeros((num_nodes, num_nodes))
    for i in range(len(tree)):
        W[i, tree[i]] = w
        W[tree[i], i] = W[i, tree[i]]
    return W

def gen_prob(W):
    d = W.shape[0] 
    prob = np.zeros(2**d)
    for i in range(2**d): #goes through all possible x \in all possible inputs and gives it the probability P(x) based on weight matrix
        x = np.ones((d,1))*-1
        x[d - len(list(bin(i)[2:])):, 0] = np.array(list(bin(i)[2:])).astype(int)*2-1  
        prob[i] = np.exp(np.matmul(np.transpose(x), np.matmul(W, x))*0.5)
    prob = prob/sum(prob)
    return prob

def add_noise(q, noisy_nodes, samples):
    if q:
        num_samples, num_nodes = samples.shape
        temp = rg.choice([1,-1],size = (num_samples, len(noisy_nodes)), p=[1-q, q])
        samples[:,noisy_nodes] = np.multiply(samples[:,noisy_nodes],temp)
    return samples

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def parallelTestFunc(num_samples, THETA, Q, NOISY_NODES, tree, equivalence_class, prob):
    '''
    Function used to do n runs for one sample number in parallel for testing with an Ising model.
    num_samples: int. Number of samples to generate for this experiment
    THETA: float. Theta value between nodes in the Ising model.
    Q: float. Crossover probability to be applied to nodes in NOISY_NODES
    NOISY_NODES: List of ints. Nodes to apply noise to.
    tree: List. Tree in UFDS form.
    equivalence_class: List of set of frozenset. Edge sets for all trees in an Equivalence class of the given tree.
    prob: List. Probability distribution for the alternative method of sample generation.
    returns: ndarray which represents if SGA, KA or ChowLiu makes an error. Returns 1 in their respective position if that algorithm makes an error, returning a tree not in the Equivalence class.
    '''
    sga_error = 0 
    k_error = 0
    cl_error = 0
    # vk_error = 0
    RHO_MAX = 1-2*THETA
    samples = generate_n_tree(num_samples, THETA, Q, NOISY_NODES, tree)

    '''
    alternative method for generating samples based on weight matrix W:
    samples = generate_n_prob(num_samples, prob)
    samples = add_noise(Q, NOISY_NODES, samples)
    '''
    #get statistics
    emp_cov = np.abs(np.corrcoef(samples, rowvar = False))
    means = np.mean(samples, axis = 0)
    mu_max = max(means)

    #compute proximal sets
    thres1 = (RHO_MAX)**4 * (1 - 2 * Q) ** 2*(1-mu_max**2)/2
    thres2 = min([thres1 * (1 - 2 * Q) * np.sqrt((1-mu_max**2)) / RHO_MAX, thres1])
    prox1, prox2 = sga_algo.get_prox(emp_cov, thres1, thres2)

    #find trees for each rule
    try:
        k_tree, k_init_error = k_algo.find_tree(emp_cov, prox1, prox2, RHO_MAX)
        sga_tree, m_init_error = sga_algo.find_tree(emp_cov, prox1, prox2, RHO_MAX)
        graph = CL.getTree((samples+1)/2)
        clEdges = list(graph.edges)
        clEdges = edges_as_set_of_sets(clEdges)

        if sga_tree not in equivalence_class or m_init_error == -1:
            sga_error = 1
        if k_tree not in equivalence_class or k_init_error == -1:
            k_error = 1
        if clEdges not in equivalence_class:
            cl_error = 1
    except IndexError:
        sga_error = 1
        k_error = 1
        cl_error = 1
    return np.asarray([sga_error, k_error, cl_error])


def parallelGaussianTest(num_samples, rho_min, RHO_MAX, SigmaQ, equivalence_class, Q):
    '''
    Function used to do n runs for one sample number in parallel for testing with a Gaussian model.
    num_samples: int. Number of samples to generate for this experiment
    rho_min: minimum true correlation coefficient between any two nodes before noise is added.
    RHO_MAX: maximum true correlation cofficient between any two nodes before noise is added.
    SigmaQ: ndarray. covariance matrix to generate the distribution from.
    equivalence_class: List of set of frozenset. Edge sets for all trees in an Equivalence class of the given tree.
    Q: maximum of (Noise Variance of node i)/(Variance of node i)
    returns: ndarray which represents if SGA or KA makes an error. Returns 1 in their respective position if that algorithm makes an error, returning a tree not in the Equivalence class.
    '''
    sga_errors = 0
    k_errors = 0
    num_nodes = SigmaQ.shape[0]
    samples = np.random.multivariate_normal([0]*num_nodes, SigmaQ, size = (num_samples))
    emp_cov = np.abs(np.corrcoef(samples, rowvar = False))
    means = np.mean(samples, axis = 0)
    mu_max = max(means)
    
    #compute proximal sets
    thres1 = (rho_min)**4 * (1/(1+Q)) /2
    thres2 = min([thres1 / (RHO_MAX*math.sqrt(1+Q)), thres1])
    prox1, prox2 = sga_algo.get_prox(emp_cov, thres1, thres2)
    #find trees for each rule

    k_tree, k_init_error = k_algo.find_tree(emp_cov, prox1, prox2, RHO_MAX)
    sga_tree, m_init_error = sga_algo.find_tree(emp_cov, prox1, prox2, RHO_MAX)
    if sga_tree not in equivalence_class or m_init_error == -1:
        sga_errors += 1
    if k_tree not in equivalence_class or k_init_error == -1:
        k_errors += 1
    return np.asarray([sga_errors,k_errors])

def genSigma(tree, w):
    '''
    Generates the inverse covariance matrix from which to generate Gaussian samples.
    tree: List. Tree in UFDS form
    w: Weight of the off-diagonal non-zero entries for the precision matrix
    '''
    num_nodes = len(tree)
    W = np.zeros((num_nodes, num_nodes))
    for i in range(len(tree)):
        W[i, tree[i]] = w
        W[tree[i], i] = W[i, tree[i]]
    np.fill_diagonal(W,1)
    return W

def getStatistics(tree, t, noisy_nodes, q):
    '''
    tree: List. Tree in UFDS form
    t: weight for the off-diagonal non-zero entries of the precision matrix
    q: Variance of noise
    returns: noisy covariance matrix, minimum correlation between non-noisy nodes, maximum correlation between non-noisy nodes, Q
    '''
    a = genSigma(tree, t)
    num_nodes = a.shape[0]
    Sigma = np.linalg.inv(a)
    SigmaQ = np.copy(Sigma)

    noise_var = q

    max_standardized_noise = 0
    for i in noisy_nodes:
        if noise_var/SigmaQ[i,i] > max_standardized_noise:
            max_standardized_noise = noise_var/SigmaQ[i,i]
        SigmaQ[i,i] += noise_var

    sd = np.sqrt(Sigma.diagonal())
    m = np.outer(sd, sd)

    corr = np.abs(Sigma/m)
    off_diag = []
    for i in range(num_nodes-1):
        off_diag.append(corr[i,i+1])
    rho_min = np.min(off_diag)
    np.fill_diagonal(corr,0)
    RHO_MAX = np.max(corr)

    return SigmaQ, rho_min, RHO_MAX, max_standardized_noise