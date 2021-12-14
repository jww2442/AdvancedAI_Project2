# CS-7713_Proj2
Probabilistic Reasoning over Time

Team project over Github. Implemented three inference algorithms for HMMs (Hidden Markov Models): Country-dance algorithm, Online fixed lag smoothing, and Viterbi algorithm. Also Implemented particle filtering for DBNs (Dynamic Bayesian Networks). 

Team members: Noah Schrick, Noah Hall, Jordan White

All Figures and Exercises are from Stuart Russell & Peter Norvig, Artificial Intelligence: A Modern Approach (Fourth Edition), Pearson, 2021.

## PART 1 (HMMs)

Develops Hidden Markov Models (HMMs) and uses the algorithms below to compute the state estimation, smoothing, and fixed lag smoothing

### Implements:
  - exact HMM smoothing algorithm using constant space, the forward-backward Country-Dance algorithm
  - the online fixed-lag smoothing algorithm in Figure 14.6
  - the most likely sequence of states, using the Viterbi algorithm

### Reports:
  -  results from lag values over the range [2,5]
  -  probabilities for the scenarios in Exercise 17, for all t ∈ {1 . . . 25}
  

## PART 2 (DBNs)

For the domain presented in Figure 14.7, consider the DBN formulation where each state consists of a {location, heading} pair which results in a state space size of 42 × 4 = 168. Uses the environment simulator provided to obtain the sequence of observations and computes the filtering probabilities of the posterior distribution of the robot state from observations received for each time step t = [1, . . . , 100]

### Implements:
  - the particle filtering algorithm (Figure 14.17) for approximate inference in DBNs

### Reports:
  - the most likely state(s)
  - corresponding probability
