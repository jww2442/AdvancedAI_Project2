#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time: Part 2 - DBN
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White


from CS5313_Localization_Env import localization_env as le
import numpy as np
import random
import argparse
from csv import writer


#Environment variables
action_bias = 0
observation_noise = 0
action_noise = 0
dimension_x = (6,7)
#Optional
#seed = 1

#Optional
#x = 6
#y = 7

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--action-bias", dest = "ACTION_BIAS", default = 0, type = range_limited_float_type, help = "Action bias. Float between 0 and 1.")
    parser.add_argument("-o", "--observation-noise", dest = "OBSERVATION_NOISE", default = 0, type = range_limited_float_type, help = "Observation noise. Float between 0 and 1.")
    parser.add_argument("-a", "--action-noise", dest = "ACTION_NOISE", default = 0, type = range_limited_float_type, help = "Action noise. Float between 0 and 1.")
    parser.add_argument("-x", "--dimensions-x", dest = "DIMENSIONS_X", default = 6, type = int, help = "Size of dimension x for maze.")
    parser.add_argument("-y", "--dimensions-y", dest = "DIMENSIONS_Y", default = 7, type = int, help = "Size of dimension y for maze.")
    parser.add_argument("-t", "--time-steps", dest = "TIME_STEPS", default = 50, type = int, help = "Number of time steps.")
    args = parser.parse_args()

    args_tuple = (args.DIMENSIONS_X, args.DIMENSIONS_Y)
    env = le.Environment(
        args.ACTION_BIAS, 
        args.OBSERVATION_NOISE, 
        args.ACTION_NOISE, 
        args_tuple) 
        #seed=seed, 
        #window_size=[x,y]


    #Loop through a total of t time steps
    for t in range(args.TIME_STEPS):    
        samples, env = pf(100, env)
        most_likely = max(samples, key=samples.get)
        prob = samples[most_likely]
        #print("Most likely state at time t is", most_likely, "with a probability of", prob)
        #print("We are actually at:", env.robot_location)
        #print()
        to_write = [args.ACTION_BIAS, args.OBSERVATION_NOISE, args.ACTION_NOISE, t, most_likely, prob]
        append_csv(to_write)

#Argparser definition to limit range of float values
def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("Argument must be < 1 and > 0")
    return f

#Write results to a CSV
def append_csv(list_of_ele):
    with open('results.csv', 'a+', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(list_of_ele)
    file.close()

class DBN:
    def __init__(self, action_bias, action_noise, dimensions, seed, x, y):
        self.action_bias = action_bias
        self.action_noise = action_noise
        self.dimensions = dimensions
        self.seed = seed
        self.x = x
        self.y = y
        

def pf(N, env):
    #Init S to empty list
    S = []

    #Init S[0] with priors
    S.append(env.location_priors)

    #Init weights to 0
    w = [0 for _ in range(N)]

    #Increment to the t+1 state. Probability table for t+1 is auto-updated with the env.update() fcn.
    #To update to t+1, our loc probs and head probs are pulled from prior prob table.
    env.update(env.location_priors, env.heading_priors)
    observation = env.move()

    #Save the probability table for later use
    S.append(env.location_priors)
    #Sort in descending order based on key
    sorted_S = sorted(S, key=S.get, reverse=True)

    #Flag as a lazy way to keep track of when we need to use the last item in dict
    flag = 0
    #Running total of weights for normalization
    w_tot = 0
    #ID what state we are assigning
    state_id = []
    for i in range(1, N):
        rand_num = random.random()
        for j in sorted_S.keys():
            #Get most likely robot position and see if it passes rng
            if (rand_num <= sorted_S[j].value()):
                #If it does, update our weights: P(X|e) * P(e)
                w[i] = sorted_S[j].value() * env.observation_tables[j[0]][j[1]][observation]
                state_id[i] = sorted_S[j].key()
                flag = 1
                #If we have identified the state we're in, break so that we don't keep overwriting w[i]
                break
        #If we've gone through entire dictionary and still haven't decided
        if (flag == 0):
            w[i] = sorted_S[-1].value() * env.observation_tables[S[-1].keys()[0]][S[-1].keys()[1]][observation]
            state_id[i] = sorted_S[-1].key()

        #Add the weight to a running total of all the weights
        w_tot += w[i]


    # Normalize all the weights
    for i in range(N):
        w[i] = w[i] / w_tot

    state_id = weighted_sample_replacement(N, state_id, w, env)
    samples = {}

    #Get corresponding probability
    for state in state_id:
        samples[state] = S[state]
    return samples, env


def weighted_sample_replacement(N, S, weight, env):
    #Make empty list to hold our selections
    collection = []
    for w in weight:
        #If we have things in our collection list already, then we just want to add w to the last element of the list
        if collection:
            collection.append(w + collection[-1])
        else:
            #Otherwise, just add w to collection since we don't have anything yet
            collection.append(w)

    #We now need to choose a set of samples and return it
    samples = selection(colection, N, S)
    return samples


def selection(weights, N, S):
    #Highest weight is at front
    sorted_weights = sorted(weights, key = lambda x: x[1])

    #Make a probability list using exponential distribution
    probs = np.random.exponential(scale=1, size = len(sorted_weights))

    #Normalize the data so that it adds "close enough" to 1.
    probs /= probs.sum()

    #Sort the probabilities. The greatest probability should be at front to match weighted_samples
    sorted_probs = sorted(probs, reverse=True)

    #Pick N samples in form of (index,element)
    #final_samples = np.random.choice(list(enumerate(sorted_weights)), N, p=sorted_probs)
    final_samples = []
    for i in range(N):
        final_samples.append(S[(np.random.choice(list(enumerate(sorted_weights)), p=sorted_probs))[0]])
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