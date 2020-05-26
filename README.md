# EMForGMM
## Algorithm theory
I won't explain the principle of theory of EM algorithm here. My main reference book is `"Pattern recognition and machine learning"`.

Divide the implement into four steps:
* 1.Initialize  parameter
* 2.E-step: Evaluate the responsibilities using the current parameter values(from the last M-step)
* 3.M-step: Re-estimate the parameters using the current responsibilities
* 4.Return to step 2, if the convergence criterion is not satisfied.In this implement I use log likelihood as convergence criterion.
## Implementation details
Although there is a lot of code for EM, this code has the following `advantages`:
* `Extendibility`：There are no constraints on the dimensions of the data.The parameters of the GMM can be initialized adaptively to the data, as long as the size of the data is N*C.
* `clean and fast`: Full use of matrix calculations by numpy, which reduces the number of for loops and speeds up the code.
In detial, there are there equations that need to be optimized, which are compute multivariate normal distribution, re-estimate mean, and re-estimate covariance.
And in this code, I used matrix operations to optimize these calculations all.
