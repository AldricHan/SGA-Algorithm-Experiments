
def original_find_EC(emp_cov, root, prox1, prox2, subtree, eps):
    '''
    Original Find_EC implemented by Katiyar et al. (2020)
    '''
    EC = []
    if (len(subtree) <= 2):
        return subtree
    common_nodes_big = list(set(prox1[root])& set(subtree))
    for i in common_nodes_big:
        is_leaf = True
        k1 = i
        count = 0
        while k1==i or k1 == root:
            k1 = common_nodes_big[count]
            count +=1
        sig_i_k1 = emp_cov[i,k1]
        sig_root_k1 = emp_cov[root, k1]
        common_nodes = list(set(prox2[root])& set(prox2[i]) & set(prox2[k1]) & set(subtree))
        for k2 in common_nodes:
            sig_root_k2 = emp_cov[root, k2] 
            sig_i_k2 = emp_cov[i,k2]

            #Here they make an abridged check for the relationship between i,k1 and i,k2
            if min((sig_i_k1*sig_root_k2) / (sig_i_k2*sig_root_k1),(sig_i_k2*sig_root_k1) / (sig_i_k1*sig_root_k2))<eps:
                is_leaf = False

        if is_leaf:
            EC.append(i)       
    return EC
