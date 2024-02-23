# List of Unit Test Ideas for TAZ

### WigBayes
- Test that a 2 spingroup case with a small second level-density converges to the 1 spingroup case.
- Test that a 3 spingroup case with a small third level-density converges to the 2 spingroup case.
- Test that providing Poisson distributions to WigBayes will return the same as the prior.
- Test that the correct assignment rate matches the assignment probabilities within statistical error.

### WigSample
- Test that WigSample returns spingroups with the correct frequency based on the underlying level-densities.
- Test that WigSample returns spingroups that produce the underlying distribution. Verify with Chi-square test.

### WigMaxLikelihood
- Test that WigMaxLikelihood returns the maximum of the prior probabilities when provided Poisson distributions.