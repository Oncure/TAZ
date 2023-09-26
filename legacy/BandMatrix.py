import numpy as np

class BandMatrix:
    """
    Saves variables to a band matrix to save on memory.

    ...
    """

    # bl:  int
    # bu:  int
    # N:   int
    # K:   int
    # mtx: np.ndarray

    """ Methods: """
    def __init__(self, N:int, bl:int, bu:int, K:int = None, dtype='float', order='C'):
        self.bl  = bl  # Lower band width
        self.bu  = bu  # Upper band width
        self.N   = N   # Size of the NxN matrix
        self.K   = K   # Third dimension
        if K == None:
            self.mtx = np.zeros((N,bu+bl+1), dtype=dtype, order=order)
        else:
            self.mtx = np.zeros((N,bu+bl+1,K), dtype=dtype, order=order)

    def __getitem__(self, indices):
        if (not isinstance(indices, tuple)) or (len(indices) != 2):
            raise IndexError("Indices must be 2-dimensional.")

        if self.K == None:      I,J   = indices
        else:                   I,J,k = indices
        Idx = self.index(I,J)[0]
        if Idx == None:
            return []
        elif Idx == 0:
            return 0
        elif isinstance(Idx,list):
            Idx1, Idx2 = Idx
            return np.concatenate((self.mtx[Idx1[0],Idx1[1]],self.mtx[Idx2[0],Idx2[1]]),axis=0)
        else:
            if self.K == None:      return self.mtx[Idx[0],Idx[1]]
            else:                   return self.mtx[Idx[0],Idx[1],k]

    def __setitem__(self, indices, value):
        if (not isinstance(indices, tuple)) or (len(indices) in (2,3)):
            raise IndexError("Indices must have 2 or 3 dimensions.")

        if self.K == None:      I,J   = indices
        else:                   I,J,k = indices
        Idx = self.index(I,J)
        if Idx in (None,0):
            raise RuntimeWarning("Index not within band matrix.")
        elif hasattr(Idx,'__iter__'):
            Idx1, Idx2 = Idx
            Split = len(Idx1)
            if self.K == None:
                self.mtx[Idx1[0],Idx1[1]]   = value[:Split]
                self.mtx[Idx2[0],Idx2[1]]   = value[Split:]
            else:
                self.mtx[Idx1[0],Idx1[1],k] = value[:Split]
                self.mtx[Idx2[0],Idx2[1],k] = value[Split:]
        else:
            if self.K == None:      self.mtx[Idx[0],Idx[1]]   = value
            else:                   self.mtx[Idx[0],Idx[1],k] = value

    def index(self,I,J):
        TL = self.bl+self.bu+1
        if isinstance(I,int):
            i  = I % TL
            if isinstance(J,int):
                if (J-I <= self.bu and I-J <= self.bl):
                    return (i,J)
                else:
                    return 0
            elif isinstance(J,slice):
                if J.step == None:
                    J.step = 1
                low  = I-self.bl
                high = I+self.bu+1
                if J.step > 0:
                    if J.start == None or J.start <= low:
                        J.start = low
                    elif J.start >= high:
                        return None
                    if J.stop == None or J.stop >= high:
                        J.stop = high
                    elif J.stop <= low:
                        return None
                else:
                    if J.start == None or J.start >= high:
                        J.start = high
                    elif J.start <= low:
                        return None
                    if J.stop == None or J.stop <= low:
                        J.stop = low
                    elif J.stop >= high:
                        return None
                return (i,J)
            else:
                raise IndexError("Unsupported index data type.")

        elif isinstance(I,slice):
            if I.step == None:
                I.step = 1
            if isinstance(J,int):
                low  = J-self.bu
                high = J+self.bl+1
                if I.step > 0:
                    if I.start == None or I.start <= low:
                        I.start = low
                    elif I.start >= high:
                        return None
                    if I.start == None or I.start >= high:
                        I.start = high
                    elif I.start <= low:
                        return None
                else:
                    if I.start == None or I.start >= high:
                        I.start = high
                    elif I.start <= low:
                        return None
                    if I.stop == None or I.stop <= low:
                        I.stop = low
                    elif I.stop >= high:
                        return None
                
                if I.start//TL == I.stop//TL:
                    i = slice(I.start % TL, I.stop % TL, I.step)
                    return (i,J)
                else:
                    if I.step > 0:
                        i1 = slice(I.start % TL,TL,I.step)
                        i2 = slice(0,I.stop % TL,I.step)
                    else:
                        i1 = slice(I.start % TL, 0, I.step)
                        i2 = slice(TL, I.stop % TL, I.step)
                    return [(i1,J),(i2,J)]
            elif isinstance(J,slice):
                raise NotImplementedError("There is currently no functionality for double slice indices.")
            else:
                raise IndexError("Unsupported index data type.")
        else:
            raise IndexError("Unsupported index data type.")
        
    def size_comp(self):
        # Returns the size of the compressed matrix
        return self.mtx.size()
    def shape_comp(self):
        # Returns the shape of the compressed matrix
        return self.mtx.shape()
    def shape_mtx(self):
        # Returns the shape of the representative matrix
        if self.K == None:      return (self.N, self.N)
        else:                   return (self.N, self.N, self.K)



