def compute_trie(Xtr, k, EOW = '$'):
    n = len(Xtr)
    roots = []
    for i in range(n):
        roots.append({})
        for l in range(len(Xtr[i])-k+1):
            tmp = roots[i]
            for level in range(k):
                tmp = tmp.setdefault(Xtr[i][l+level],{})
            tmp[EOW] = EOW
    return roots

def compute_occurences(Xtr,k):
    n = len(Xtr)
    occs = []
    for i in range(n):
        occs.append({})
        for l in range(len(Xtr[i])-k+1):
            if Xtr[i][l:l+k] in occs[i].keys():
                occs[i][Xtr[i][l:l+k]] += 1
            else:
                occs[i][Xtr[i][l:l+k]] = 1
    return occs
