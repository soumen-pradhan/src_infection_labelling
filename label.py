
# Page 4
def labelRankingScore(G, y, a):
    pass
    # W = adj(G) Adjacency Matrix
    # D(i, i) = W(i, j) Diagonal Matrix

    # S
    f = y

    while true: # f is NOT convergent
        pass
        # f = a * (S*f) + (1 - a) * y

    return f

def BLRSI(G, y, a):
    f = labelRankingScore(G, y, a)
    s = max(f)

    return s

# Page 9
def GFGH(G, y):
    pass
    # W = adj(G) Adjacency Matrix
    # D(i, i) = W(i, j) Diagonal Matrix

    # P
    # F

    while true: # F is NOT convergent
        pass
        # F = P * F

        # F_l = Y_l

    # O = max(F)
    #return O

# Page 10
def LGC(G, Y, a):
    pass
    # W = adj(G) Adjacency Matrix
    # D(i, i) = W(i, j) Diagonal Matrix

    # S
    # F

    while true: # F is NOT convergent
        pass
        # F = a * (S * F) + (1-a) * Y

    # O = max(F)
    #return O


# Ultimate Function
def TSSI_GFGH(G, Y, a):
    O = GFGH(G, Y)
    # label vecotr y from O
    s = BLRSI(G, y, a)
    return s

def TSSI_LGC(F, Y, a1, a2):
    O = LGC(G, Y, a1)
    # label vecotr y from O
    s = BLRSI(G, y, a2)
    return s
