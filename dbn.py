#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time: Part 2 - DBN
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White


from CS5313_Localization_Env import localization_env as le
import numpy as np
import random

#Environment variables
action_bias = 0
observation_noise = 0
action_noise = 0
dimensions = (6,7)
#Optional
#seed = 1

#Optional
#x = 6
#y = 7

def main():

    env = le.Environment(
        action_bias, 
        observation_noise, 
        action_noise, 
        dimensions) 
        #seed=seed, 
        #window_size=[x,y]

    
    pf(1, 1, env)  


class DBN:
    def __init__(self, action_bias, action_noise, dimensions, seed, x, y):
        self.action_bias = action_bias
        self.action_noise = action_noise
        self.dimensions = dimensions
        self.seed = seed
        self.x = x
        self.y = y
        

def pf(e, N, env):
    #Init S to empty list
    S = []

    #Init S[0] with priors
    S.append(env.location_priors)
    print(S)

    #Init weights to 0
    w = [0 for _ in range(N)]

    #Increment to the t=1 state. Probability table for t=1 is auto-updated with the env.update() fcn.
    #To update to t=1, our loc probs and head probs are pulled from prior prob table.
    env.update(env.location_priors, env.heading_priors)
    observation = env.move()

    S.append(env.location_priors)
    sorted_S = sorted(S, key=S.get, reverse=True)

    flag = 0
    w_tot = 0
    for i in range(1, N):
        rand_num = random.random()
        for j in sorted_S.keys():
            #Get most likely robot position and see if it passes rng
            if (rand_num <= sorted_S[j].value()):
                #If it does, update our weights: P(X|e) * P(e)
                w[i] = sorted_S[j].value() * env.observation_tables[j[0]][j[1]][observation]
                flag = 1
        #If we've gone through entire dictionary and still haven't decided
        if (flag = 0):
            w[i] = sorted_S[-1].value() * env.observation_tables[S[-1].keys()[0]][S[-1].keys()[1]][observation]

        w_tot += w[i]


    # Normalize all the weights
    for i in range(N):
        w[i] = w[i] / w_tot

    S = weighted_sample_replacement(N, S, w, env)
    return S


def weighted_sample_replacement(N, S, weight, env):
    #Make empty list
    totals = []
    for w in weights:
        if totals:
            totals.append(w + totals[-1])
        else:
            total.append(w)

def selection(weights, N):
    #Highest weight is at front
    sorted_weights = sorted(weights, key = lambda x: x[1])

    #Make a probability list using exponential distribution
    probs = np.random.exponential(scale=1, size = len(sorted_weights))

    #Normalize the data so that it adds "close enough" to 1.
    probs /= probs.sum()

    #Sort the probabilities. The greatest probability should be at front to match weighted_samples
    sorted_probs = sorted(probs, reverse=True)

    #Pick N samples
    final_samples = np.random.choice(sorted_weights, N, p=sorted_probs)

    return final_samples

#Weighted random selection of an action based on a list of possible actions
def choose_action(poss_actions, env, state):
    #If only one possible move, just take that one
    if (len(poss_actions) == 1):
        return poss_actions

    #Init list of weights
    weights = []
    for moves in poss_actions:
        weights.append(env.location_priors[moves])
    #Get size of weights
    w_size = len(weights)
    for w in range(w_size):
        #Get the location of the state we're most sure of
        move_loc = poss_actions[np.argmax(weights)]
        #Generate a random number
        rand_num = random.random()

        print(weights)
        #If the number is within the probability, take the action
        if (rand_num <= max(weights)):
            return move_loc
        #If we only have one possible move left, take it
        elif (len(weights) == 1):
            return move_loc
        #Otherwise, remove the action and go check the others
        else:
            tmp = weights.remove(max(weights))


#Returns a list of possible actions from the current location
#INPUTS: state - current robot location
#        env - robot localization environment
def find_actions(state, env):
    #List to hold possible moves
    poss_moves = []
    #Loop through all free cells to see what's possible
    for coord in env.free_cells:
        #Make sure we only go +-1 from either X OR Y, NOT BOTH, and we cannot stay in same location
        if (abs(coord[0] - state[0]) == 1 and not (coord[0] - state[0] == 0)):
            if (coord[1] == state[1]):
                poss_moves.append(coord)

        elif (abs(coord[1] - state[1]) == 1 and not (coord[1] - state[1] == 0)):
            if (coord[0] == state[0]):
                poss_moves.append(coord)   
       
    return poss_moves

if __name__ == '__main__':
    main()