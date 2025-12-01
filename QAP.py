#Quadratic Assignment Problem.

from tkinter import N
import numpy as np 


#i need for functions let's implemet them one by one 
#the origial QAP formula  >> i need F as flow matrix , D as distance matrix and p as permutation 

def original_qap(F,D,P):
    N = len(P)
    cost = 0.0
    for i in range (N):
        for j in range (N):
            cost = cost + F[i, j] * D[P[i], P[j]]
    return cost 


#the second functions would be pealty cost 


def penalty_cost(x,pc):
    row = x.sum(axis=1) - 1
    col = x.sum(axis=0) - 1
    return pc * (np.sum(row**2) + np.sum(col**2))


# the third function is delta swap 
def delta_swap(F, D, P, a, b):
    N = len(P)
    ra, rb = P[a], P[b] # original locations of facility a and b
    delta = 0.0

    # 1. & 2. Contributions for all t != a,b (your original loop)
    for t in range(N):
        if t == a or t == b:
            continue
        rt = P[t]
        delta += (F[a, t] * (D[rb, rt] - D[ra, rt]) # (a, t)
                  + F[t, a] * (D[rt, rb] - D[rt, ra]) # (t, a)
                  + F[b, t] * (D[ra, rt] - D[rb, rt]) # (b, t)
                  + F[t, b] * (D[rt, ra] - D[rt, rb])) # (t, b)

    # 3. Internal pair contribution (a,b) and (b,a) (your original code)
    delta += F[a, b] * (D[rb, ra] - D[ra, rb])
    delta += F[b, a] * (D[ra, rb] - D[rb, ra])
    
    # 4. Diagonal contributions (a,a) and (b,b) 
    delta += F[a, a] * (D[rb, rb] - D[ra, ra]) # (a, a)
    delta += F[b, b] * (D[ra, ra] - D[rb, rb]) # (b, b)

    return delta



#the forth function is the total cost that would be objective enery + the penealty energy 

def total_cost(F,D,P,pc= 0.0):
    N = len(P)
    x = np.zeros((N,N), dtype=int)
    x[np.arange(N), P] = 1
    return original_qap(F, D, P) + penalty_cost(x, pc)


# cht gbt says it is better to check the delta swap formula 

def apply_swap_and_check(F, D, P, a, b, pc=0.0):
    old = total_cost(F, D, P, pc)
    delta = delta_swap(F, D, P, a, b)

    N = len(P)
    x = np.zeros((N,N), dtype=int)
    x[np.arange(N), P] = 1
    old_pen = penalty_cost(x, pc)

    # change x for penalty check
    x[a, P[a]] = 0; x[a, P[b]] = 1
    x[b, P[b]] = 0; x[b, P[a]] = 1
    new_pen = penalty_cost(x, pc)

    pred = old + delta + (new_pen - old_pen) # pred = prediction 

    p2 = P.copy()
    p2[a], p2[b] = p2[b], p2[a]
    true = total_cost(F, D, p2, pc)

    return pred, true, pred - true


if __name__ == "__main__":
    np.random.seed(0)
    N = 5
    F = np.random.randint(0,10,(N,N))
    D = np.random.randint(0,10,(N,N))
    p = np.arange(N)
    np.random.shuffle(p)
    a, b = 1, 3

    pred, true, diff = apply_swap_and_check(F, D, p, a, b)
    print("pred:", pred)
    print("true:", true)
    print("diff:", diff)



    
