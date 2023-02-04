
import math
import numpy as np
from scipy.special import erfc

pid4 = math.pi/4
sqrt_pid4 = math.sqrt(pid4)

def MeanLevelSpacings(X, verbose=False):
    N = X.size
    S2 = np.sum(X**2)
    MLS = pid4 * math.sqrt(S2) * np.prod([1 - 0.5 / k for k in range(1,N+1)])
    MLS_SD = math.sqrt(MLS**2 + (N + 0.5) / (pid4 * S2) - 2.0)

    if verbose:     print(f'Mean level-spacing = {MLS} +/- {MLS_SD}')
    return MLS, MLS_SD

def LSGoodnessFit(res, MP, samples):
    NSamples = samples.shape[1]
    L = samples.shape[0]
    n = MP.Freq.shape[1] - 1
    
    A2 = 0
    T  = 0
    KS = 0
    for g in range(n):
        Indices = [[idx for idx in range(L) if samples[idx,K] == g] for K in range(NSamples)]

        LS  = MP.Freq[0,g] * np.array([ls for indices in Indices for ls in np.diff(res.E[indices])])
        LSB = MP.Freq[0,g] * np.array([ls for indices in Indices for ls in [res.E[indices[0]]-res.EB[0], res.EB[1]-res.E[indices[-1]]]])

        # FIXME: DOES NOT INCLUDE BRODY DISTRIBUTION!!!!
        Y  = np.exp(-pid4 * LS**2)
        YB = erfc(sqrt_pid4 * LSB)
        Y_sorted = np.concatenate((Y, YB))
        Y_sorted.sort()
        
        N = Y_sorted.size
        I = np.array(range(1,N+1))

        A2 += -N - np.sum((2*I-1) / N * np.log(Y_sorted*(1.0 - np.flip(Y_sorted))))
        T  += 1 / (12 * N) + np.sum(((2*I-1)/(2*N) - Y_sorted)**2)
        KS += np.max(abs(I/N - Y_sorted))

    return A2, T, KS

def Anderson_Darling_pdf(A2):
    'Returns the Emperically derived PDF for the Anderson-Darling Statistic'
    Poly = np.polynomial.polynomial.Polynomial([-4.72725274, -2.22442511, -6.01114371,  0.83633392, -6.35931921])
    logA2 = np.log(A2)
    return np.exp(Poly(logA2))

def Anderson_Darling_log_pdf(A2):
    'Returns the Empirically-derived PDF for the Anderson-Darling Statistic'
    Poly = np.polynomial.polynomial.Polynomial([-4.72725274, -2.22442511, -6.01114371,  0.83633392, -6.35931921])
    logA2 = np.log(A2)
    return Poly(logA2)

poly = np.polynomial.polynomial.Polynomial([-4.72725274, -2.22442511, -6.01114371,  0.83633392, -6.35931921])
def rep_score(E, SGs, Freqs):
    '...'

    # Getting Anderson-Darling statistic:
    A2 = np.zeros(SGs.shape[1:])
    for g in range(SGs.shape[2]):     # Spin group index
        for j in range(SGs.shape[1]): # Epoch index
            ls = Freqs[g]*np.diff(E[SGs[:,j,g] == g])
            n = len(ls)

            # Anderson-Darling statistic:
            R = np.sort(np.exp(-math.pi/4*(ls*ls)))
            A2[j,g] = -n - np.mean(np.arange(1,2*n,2).reshape(-1,1) * np.log(R*(1.0-np.flip(R, axis=0))), axis=0)

    # Empirically-derived Anderson-Darling PDF:
    logA2 = np.log(A2)
    log_probs = poly(logA2)

    # Combining Probabilities over spin groups and epochs:
    combined_stat = np.mean(np.exp(np.sum(log_probs, axis=1)))
    return combined_stat


