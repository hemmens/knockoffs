# knockoffs
The code in this repository accompanies the Robert and Hemmens paper about using linear algerbra to derive Model-X Knockoffs, as originally described in Candès (2018).

## knockoff_lib.py
This is a library we use to perform a number of actions:

- Calculate the coskewness and cokurtosis matrices for a collection of random variables. These actions are not available in the scipy package; instead just the skewness and kurtosis for a single variable are available.
- It includes functions that provide the optimization algorithm with constraints that ensure that the covariance, coskewness, and cokurtosis values between a knockoff and the other features and knockoffs are the same as those for the knockoff's feature.
- Generate an initial knockoff guess using the Candès methodology of generating knockoffs from multivariate Normal distributions.

## linalg2.py
In this code, you can sample a set of p independent random Uniform(0,1) variables with n observations. The values of these random variables are then shuffled to create non-zero or non-independence covariance, coskewness, and cokurtosis values between the variables.

The uniform variables are then transformed into Normal(0,1) variables with a positive semidefinite covariance matrix. We use the Candès (2018) methodology to generate an initial value for the Model-X Knockoffs of a Gaussian multivariate dsitribution. These knockoffs are very fast to generate and normally satisfy the covariance conditions for the knockoffs. We give this as our first guess into the optimization algorithm that works to minimize the covariance between each knockoff and its feature, while ensuring that the coskewness and cokurtosis conditions for the knockoffs are met.

## linalg_SMI2.py
In this code, we use real world data from the Swiss Stock Market Index, the SMI. We use daily returns data for the 20 constituent stocks from 29th May 2017 to 8th April 2019, giving us 481 observations. We use a copula to estimate the univariate distributions of these 20 return profiles and find that most of them are Student-t distributions, with one fitting a LogLaplace distribution. We use these distributions to convert the returns into Uniform(0,1) variables. These Uniform(0,1) variables are available in smi_uniform.csv.

We can then use the code from linalg2.py to generate knockoffs for this real-world dataset.
