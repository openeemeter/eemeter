EPSILON = 1e-5

def arrays_similar(iterable1,iterable2):
    if not len(iterable1) == len(iterable2):
        return False
    for i,j in zip(iterable1,iterable2):
        if not abs(i - j) < EPSILON:
            return False
    return True
