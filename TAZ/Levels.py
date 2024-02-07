from typing import Tuple
import numpy as np

from TAZ.Encore import Encore
from TAZ.Theory.LevelSpacingDists import SpacingDistribution, merge

__doc__ = """
This module serves as a preprocessor and postprocessor for the 2-spingroup assignment algorithm,
"Encore.py". This module extends the 2-spingroup algorithm to multiple-spingroups by "merging"
spingroups. This module finds the probabilities for various merge cases and combines the expected
probabilities. Unlike the 2-spingroup case that gives the best answer given its information
(i.e. mean level spacings, reduced widths, false resonance probabilities, etc.), the
multiple-spingroup case is an approximation.
"""

class RunMaster:
    f"""
    A wrapper over Encore responsible for partitioning spingroups, merging distributions, and
    combining spingroup probabilities. For 1 or 2 spingroups, partitioning and merging
    distributions is not necessary and RunMaster will pass to Encore. Once a RunMaster object has
    been initialized, the specific algorithm can be chosen (i.e. WigBayes, WigSample, etc.).

    ...
    """

    def __init__(self, E, EB:tuple,
                 level_spacing_dists:Tuple[SpacingDistribution], false_dens:float=0.0,
                 Prior=None, log_likelihood_prior:float=None,
                 err:float=1e-9):
        """
        Initializes a RunMaster object.

        Parameters:
        ----------
        E                    :: float, array-like
            Resonance energies for the ladder.
        EB                   :: float [2]
            The ladder energy boundaries.
        level_spacing_dists  :: Tuple[SpacingDistribution]
            The level-spacing distributions object.
        false_dens           :: float
            The false level-density. Default = 0.0.
        Prior                :: float, array-like
            The prior probabilitiy distribution for each spingroup. Default = None.
        log_likelihood_prior :: float
            The log-likelihood provided from the prior. Default = None.
        err                  :: float
            A probability threshold where resonances are considered to be too far apart to be
            nearest neighbors.
        """
        
        # Error Checking:
        # if type(level_spacing_dists) != Distributions:
        #     raise TypeError('The level-spacing distributions must be a "Distributions" object.')
        if not (0.0 < err < 1.0):
            raise ValueError('The probability threshold, "err", must be strictly between 0 and 1.')
        if len(EB) != 2:
            raise ValueError('"EB" must be a tuple with two elements: an lower and upper bound on the resonance ladder energies.')
        if EB[0] >= EB[1]:
            raise ValueError('EB[0] must be strictly less than EB[1].')
        
        self.E  = np.sort(E)
        self.EB = tuple(EB)
        self.level_spacing_dists = np.array(level_spacing_dists)

        self.lvl_dens = np.array([lvl_spacing_dist.lvl_dens for lvl_spacing_dist in level_spacing_dists] + [false_dens])

        self.L = len(E) # Number of resonances
        self.G = len(self.lvl_dens) - 1 # number of spingroups (not including false group)

        if Prior is None:
            self.Prior = np.tile(self.lvl_dens/self.lvl_dens_tot, (self.L,1))
        else:
            self.Prior = Prior
        self.log_likelihood_prior = log_likelihood_prior
        self.err = err
    
    @property
    def lvl_dens_tot(self):
        return np.sum(self.lvl_dens)
    @property
    def false_dens(self):
        return self.lvl_dens[-1]

    def mergePartitioner(s, partitions:list):
        """
        ...
        """

        n = len(partitions)

        # Merged level-spacing calculation:
        level_spacing_probs = np.zeros((s.L+2, s.L+2, n), 'f8')
        iMax = np.zeros((s.L+2, 2, n), 'i4')
        for g, group in enumerate(partitions):
            distribution = merge(*s.level_spacing_dists[group])
            iMax[:,:,g] = s.find_iMax(distribution)
            level_spacing_probs[:,:,g] = s.find_probs(distribution, s.Prior[:,group], iMax[:,:,g])

        # Merged prior calculation:
        prior_merged = np.zeros((s.L, n+1), 'f8')
        for g, group in enumerate(partitions):
            if hasattr(group, '__iter__'):
                prior_merged[:,g] = np.sum(s.Prior[:,group], axis=1)
            else:
                prior_merged[:,g] = s.Prior[:,group]
        prior_merged[:,-1] = s.Prior[:,-1]
        return level_spacing_probs, iMax, prior_merged

    def find_iMax(s, distribution:SpacingDistribution):
        """
        ...
        """

        L = s.E.size
        iMax = np.full((L+2,2), -1, dtype='i4')
        xMax_f0 = distribution.xMax_f0(s.err)
        xMax_f1 = distribution.xMax_f1(s.err)

        # Lower boundary cases:
        for j in range(L):
            if s.E[j] - s.EB[0] >= xMax_f1:
                iMax[0,0]    = j
                iMax[:j+1,1] = 0
                break

        # Intermediate cases:
        for i in range(L-1):
            for j in range(iMax[i,0]+1,L):
                if s.E[j] - s.E[i] >= xMax_f0:
                    iMax[i+1,0] = j
                    iMax[iMax[i-1,0]:j+1,1] = i+1
                    break
            else:
                iMax[i:,0] = L+1
                iMax[iMax[i-1,0]:,1] = i+1
                break

        # Upper boundary cases:
        for j in range(L-1,-1,-1):
            if s.EB[1] - s.E[j] >= xMax_f1:
                iMax[-1,1] = j
                iMax[j:,0] = L+1
                break

        return iMax
    
    def find_probs(s, distribution:SpacingDistribution, prior, iMax):
        """
        ...
        """

        L = s.E.size
        level_spacing_probs = np.zeros((L+2,L+2), dtype='f8')
        for i in range(L-1):
            X = s.E[i+1:iMax[i+1,0]-1] - s.E[i]
            prior_L = np.tile(prior[i,:], (iMax[i+1,0]-i-2, 1))
            prior_R = prior[i+1:iMax[i+1,0]-1,:]
            level_spacing_probs[i+1,i+2:iMax[i+1,0]] = distribution.f0(X, prior_L, prior_R)
        # Boundary distribution:
        level_spacing_probs[0,1:-1]  = distribution.f1(s.E - s.EB[0], prior)
        level_spacing_probs[1:-1,-1] = distribution.f1(s.EB[1] - s.E, prior)

        # Error checking:
        if (level_spacing_probs == np.nan).any():   raise RuntimeError('Level-spacing probabilities have "NaN" values.')
        if (level_spacing_probs == np.inf).any():   raise RuntimeError('Level-spacing probabilities have "Inf" values.')
        if (level_spacing_probs <  0.0).any():      raise RuntimeError('Level-spacing probabilities have negative values.')

        # The normalization factor is duplicated in the prior. One must be removed: FIXME!!!!!
        level_spacing_probs /= distribution.lvl_dens
        return level_spacing_probs

    def WigBayes(s, return_log_likelihood:bool=False, verbose:bool=False):
        """
        Returns spingroup probabilities for each resonance based on level-spacing distributions,
        and any provided prior.

        Parameters:
        ----------
        return_log_likelihood :: bool
            Determines whether to return the resonance ladder log-likelihood. Default = False.
        verbose :: bool
            The verbosity controller. Default = False.

        Returns:
        -------
        sg_probs :: int [L,G]
            The sampled IDs for each resonance and trial.
        """

        # 1 spingroup (merge not needed):
        if   s.G == 1:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            sg_probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')

            if return_log_likelihood:
                log_likelihood = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)
                return sg_probs, log_likelihood
            else:
                return sg_probs
        
        # 2 spingroups (merge not needed):
        elif s.G == 2:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            sg_probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')

            if return_log_likelihood:
                log_likelihood = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)
                return sg_probs, log_likelihood
            else:
                return sg_probs

        # More than 2 spingroups (merge needed):
        else:
            sg_probs = np.zeros((s.L,3,s.G),dtype='f8')
            if return_log_likelihood:
                log_likelihood = np.zeros(s.G, dtype='f8')

            # Partitioning:
            for g in range(s.G):
                partition = [[g_ for g_ in range(s.G) if g_ != g], [g]]
                if verbose: print(f'Preparing for Merge group, {g}')
                level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner(partition)
                if verbose: print(f'Finished spingroup {g} level-spacing calculation')
                ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
                if verbose: print(f'Finished spingroup {g} CP calculation')
                sg_probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spingroup {g} WigBayes calculation')

                if return_log_likelihood:
                    # FIXME: I DON'T KNOW LOG Likelihood CORRECTION FACTOR FOR MERGED CASES! 
                    # lvl_dens_comb = np.array([s.lvl_dens[0,g], s.lvl_dens_tot-s.lvl_dens[0,g]]).reshape(1,-1)
                    log_likelihood[g] = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)

            # Combine probabilities for each merge case:
            combined_sg_probs = s.probCombinator(sg_probs)
            if return_log_likelihood:
                if verbose: print('Preparing for Merge group, 999!!!')
                level_spacing_probs_1, iMax_1, prior_1 = s.mergePartitioner([list(range(s.G))])
                if verbose: print('Finished spingroup 999 level-spacing calculation')
                ENCORE = Encore(prior_1, level_spacing_probs_1, iMax_1)
                if verbose: print('Finished spingroup 999 CP calculation')
                base_log_likelihood = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)
                combined_log_likelihood = s.logLikelihoodCombinator(log_likelihood, base_log_likelihood)
                if verbose: print('Finished!')
                return combined_sg_probs, combined_log_likelihood
            else:
                if verbose: print('Finished!')
                return combined_sg_probs
            
    def WigSample(s, trials:int=1, verbose:bool=False):
        """
        Returns random spingroup assignment samples based on its Bayesian probability.

        Parameters:
        ----------
        trials  :: int
            The number of trials of sampling the resonance ladder. Default = 1.
        verbose :: bool
            The verbosity controller. Default = False.

        Returns:
        -------
        samples :: int [L,trials]
            The sampled IDs for each resonance and trial.
        """

        # 1 spingroup (merge not needed):
        if s.G == 1:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            samples = ENCORE.WigSample(trials)
            if verbose: print(f'Finished WigBayes calculation')
            return samples
        
        # 2 spingroups (merge not needed):
        elif s.G == 2:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            samples = ENCORE.WigSample(trials)
            if verbose: print(f'Finished WigBayes calculation')
            return samples

        # More than 2 spingroups (merge needed):
        else:
            raise NotImplementedError('WigSample for more than two spingroups has not been implemented yet.')

    def probCombinator(self, sg_probs):
        """
        Combines probabilities from various spingroup partitions.

        ...
        """

        combined_sg_probs = np.zeros((self.L,self.G+1), dtype='f8')
        for g in range(self.G):
            combined_sg_probs[:,g] = sg_probs[:,1,g]
        combined_sg_probs[:,-1] = np.prod(sg_probs[:,1,:], axis=1) * self.Prior[:,-1] ** (1-self.G)
        combined_sg_probs[self.Prior[:,-1]==0.0,  -1] = 0.0
        combined_sg_probs /= np.sum(combined_sg_probs, axis=1).reshape((-1,1))
        return combined_sg_probs

    def logLikelihoodCombinator(self, partition_log_likelihoods, base_log_likelihoods:float):
        """
        Combines log-likelihoods from from various partitions.
        """

        return np.sum(partition_log_likelihoods) - (self.G-1)*base_log_likelihoods