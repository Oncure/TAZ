# List of Unit Tests for TAZ

### Test Theory
- Test that the explicit and implicit equations for penetrability match. (NOT IMPLEMENTED YET)

### Test Distributions
- Test that each Distribution integrate correctly, have correct mean, etc.

### Test Spacing Distributions
- Test that each SpacingDistribution integrate correctly, have correct mean, etc.
- Test that Poisson distributions merge to another Poisson distribution with known level-density.

### Test Samplers
- Test that the sampling algorithms follow expected distributions.
- Test merged distribution as well. (NOT IMPLEMENTED YET)

### Test WigBayes
- Test that a 2 spingroup case with a small second level-density converges to the 1 spingroup case. (NOT IMPLEMENTED YET)
- Test that a 3 spingroup case with a small third level-density converges to the 2 spingroup case. (NOT IMPLEMENTED YET)
- Test that providing Poisson distributions to WigBayes will return the same as the prior.
- Test that the correct assignment rate matches the assignment probabilities within statistical error.

### Test WigSample
- Test that WigSample returns spingroups with the correct frequency based on the underlying level-densities.
- Test that WigSample returns spingroups that produce the underlying distribution. Verify with Chi-square test.

### Test WigMaxLikelihood
- Test that WigMaxLikelihood returns the maximum of the prior probabilities when provided Poisson distributions.