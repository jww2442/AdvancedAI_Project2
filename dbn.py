#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time: Part 2 - DBN
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White


from numpy.lib.function_base import append
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


    '''PREPARE ARG PARSING FOR SCRIPTING PURPOSES'''
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--action-bias", dest = "ACTION_BIAS", default = 0, type = range_limited_float_type, help = "Action bias. Float between 0 and 1.")
    parser.add_argument("-o", "--observation-noise", dest = "OBSERVATION_NOISE", default = 0, type = range_limited_float_type, help = "Observation noise. Float between 0 and 1.")
    parser.add_argument("-a", "--action-noise", dest = "ACTION_NOISE", default = 0, type = range_limited_float_type, help = "Action noise. Float between 0 and 1.")
    parser.add_argument("-x", "--dimensions-x", dest = "DIMENSIONS_X", default = 6, type = int, help = "Size of dimension x for maze.")
    parser.add_argument("-y", "--dimensions-y", dest = "DIMENSIONS_Y", default = 7, type = int, help = "Size of dimension y for maze.")
    parser.add_argument("-t", "--time-steps", dest = "TIME_STEPS", default = 100, type = int, help = "Number of time steps.")
    parser.add_argument("-s", "--samples", dest = "SAMPLES", default = 42, type = int, help = "Number of samples to take.")
    parser.add_argument("-g", "--ghosts", dest = "NUM_GHOSTS", default = 0, type = int, help= "Number of ghosts to spawn.")
    args = parser.parse_args()

    '''INITIALIZE ENVIRONMENT'''
    args_tuple = (args.DIMENSIONS_X, args.DIMENSIONS_Y)
    env = le.Environment(
        args.ACTION_BIAS, 
        args.OBSERVATION_NOISE, 
        args.ACTION_NOISE, 
        args_tuple) 
        #seed=seed, 
        #window_size=[x,y]
    env.running = True

    ghosts = gen_ghosts(args.NUM_GHOSTS, env)
    samples = {}

    '''PREPARE CSV'''
    #to_write = [args.ACTION_BIAS, args.OBSERVATION_NOISE, args.ACTION_NOISE, args.NUM_GHOSTS, i, curr_cell, curr_max, curr_heading, loc_corr, head_corr]
    to_write = ["Action Bias", "Observation Noise", "Action Noise", "Num Ghosts", "Time Step", "Guessed Cell", "Probability of Guess", "Guessed heading", "Location Correct", "Heading Correct"]
    append_csv(to_write)


    '''Init S with P(X0)'''
    S = {}
    for i in range(42):
        for cell in env.location_priors:
            for head in env.heading_priors:
                S[(cell[0],cell[1],head)] = env.location_priors[cell] * env.heading_priors[head]

    '''RUN PARTICLE FILTERING'''
    times_correct = 0
    heading_correct = 0
    both_correct = 0
    times_opposite = 0
    for i in range(args.TIME_STEPS):
        loc_flag = 0
        heading_flag = 0
        samples = pf(args.SAMPLES, env, S, samples, ghosts)
        print()
        #Convert to list for parsing/weight update purposes
        all_keys = list(samples.keys())

        #Find most likely cell, heading, and probability. Init to first sample
        curr_max = all_keys[0][1] * (samples[all_keys[0]]/args.SAMPLES)
        curr_cell = (all_keys[0][0][0], all_keys[0][0][1])
        curr_heading = all_keys[0][0][2]
        #Loop through all samples
        #k is in form ((x, y, heading, prob) : count)
        for k in all_keys:
            if k[1]* (samples[k]/args.SAMPLES) > curr_max:
            #if k[1]/(samples[k]/args.SAMPLES) > curr_max:
                #curr_max = k[1]/(samples[k]/args.SAMPLES
                curr_max = k[1] * (samples[k]/args.SAMPLES)
                curr_cell = (k[0][0], k[0][1])
                curr_heading = (k[0][2])

        print("Guessing we are in cell", curr_cell, "and oriented as", curr_heading)
        if(not isinstance(curr_cell, tuple)):
            print(samples)
        print("Probability of this state",curr_max)
        print("We are actually in", env.robot_location, "and oriented as", env.robot_heading)
        if curr_cell == env.robot_location:
            times_correct += 1
            loc_flag = 1
        if curr_heading == env.robot_heading:
            heading_correct += 1
            heading_flag = 1
        if ((str(curr_heading) == "Headings.E" and str(env.robot_heading) == "Headings.W") or (str(curr_heading) == "Headings.W" and str(env.robot_heading) == "Headings.E") or (str(curr_heading) == "Headings.N" and str(env.robot_heading) == "Headings.S") or (str(curr_heading) == "Headings.S" and str(env.robot_heading) == "Headings.N")):
            times_opposite += 1
        if curr_cell == env.robot_location and curr_heading == env.robot_heading:
            both_correct += 1
        
        loc_corr = False
        head_corr = False
        if (loc_flag):
            loc_corr = True
        if(heading_flag):
            head_corr = True
        to_write = [args.ACTION_BIAS, args.OBSERVATION_NOISE, args.ACTION_NOISE, args.NUM_GHOSTS, i, curr_cell, curr_max, curr_heading, loc_corr, head_corr]
        append_csv(to_write)

    print("Times location correct:", times_correct)
    print("Times heading correct:", heading_correct)
    print("Times both correct:",both_correct) 
    print("Times heading is opposite:",times_opposite)
    #chosen = selection()

def pf(N, env, S, samples = None, ghosts = None):
    #Convert prior probabilities to list
    loc_list = []
    for i in range(6):
        loc_list.append([])

        for j in range(7):
            loc_list[i].append([])

            ij_tuple = (i+1,j+1)
            if(ij_tuple in env.location_priors):
        #if ij_tuple in env.location_priors:
                loc_list[i][j] = env.location_priors[ij_tuple]
            else:
                loc_list[i][j] = 0
    
    sample_list = []
    #Convert samples to list
    for i in range(6):
        sample_list.append([])

        for j in range(7):
            sample_list[i].append([])

            ij_tuple = (i+1,j+1)
            if(ij_tuple in samples):
                sample_list[i][j] = samples[ij_tuple]
            else:
                sample_list[i][j] = 0

    #Move robot
    if(samples):
        env.update(sample_list, env.heading_priors)
    else:
        env.update(loc_list, env.heading_priors)      
    observation = env.move()

    new_samples = {}
    #Take N samples  
    print(samples)  
    for i in range(N):
        #choice = random.choice(list(env.location_transitions[cell[0]][cell[1]][head].items()))
        if (samples):
            #Init empty weights
            weights = {}
            #Get all the keys
            test_s = list(samples.keys())
            #Go through keys and get assoc weight from env prob tables
            for s in test_s:
                weights[s] = s[1]
            #Pass in our weights list, say we want 1 sample, then pass in our list of samples
            chosen = selection(weights, 1, test_s)
            choice = chosen[0]
            
            '''COMMENT OUT FOLLOWING LINE IF YOU WANT A WEIGHTED SAMPLE INSTEAD OF RANDOM'''
            #Random choice instead of weighted choice
            choice = random.choice(list(samples.keys()))
        
        #Time Step 1
        else:
            choice = random.choice(list(S.items()))

        if choice not in new_samples:
            new_samples[choice] = 1
        else:
            new_samples[choice] += 1

    #Update weights based on observation table
    all_keys = list(new_samples.keys())
    for k in all_keys:
        as_list = list(k)
        val = new_samples[k]
        del new_samples[k]
        as_list[1] = env.observation_tables[k[0][0]][k[0][1]][tuple(observation)]
        k = tuple(as_list)
        new_samples[k] = val

    return (new_samples)

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

#Delete all csv data
def clear_csv():
    f = open('results.csv', 'r+')
    f.truncate(0)


def gen_ghosts(num, env):
    ghosts = []
    for i in range(num):
        ghosts.append(Ghost())
        ghosts[i].init_loc(env)
    return ghosts

class DBN:
    def __init__(self, action_bias, action_noise, dimensions, seed, x, y):
        self.action_bias = action_bias
        self.action_noise = action_noise
        self.dimensions = dimensions
        self.seed = seed
        self.x = x
        self.y = y
        
class Ghost:         
    #Spawns a ghost on a free cell
    def init_loc(self, env):
        spawn = random.choice(env.free_cells)
        self.update_free_cells(env, spawn)
        self.pos = spawn

    #Updates the free cells after 1) A ghost spawn, or 2) a ghost move
    def update_free_cells(self, env, next_pos, curr_pos = None):
        if (curr_pos):
            env.free_cells.append(curr_pos)
        if (isinstance(next_pos, list)):
            env.free_cells.remove(next_pos[0])
            self.pos = next_pos
        else:
            env.free_cells.remove(next_pos)
            self.pos = next_pos
    
    def move(self, env):
        poss_actions = self.find_actions(self.pos, env)
        if poss_actions[0] == env.robot_location:
            return 1
        action = self.choose_action(poss_actions, env, self.pos)
        self.update_free_cells(env, action, self.pos)



    #Weighted random selection of an action based on a list of possible actions
    def choose_action(self, poss_actions, env, state):
        #If only one possible move, just take that one
        if (len(poss_actions) == 1):
            return poss_actions[0]

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
    def find_actions(self,state, env):
        #List to hold possible moves
        poss_moves = []
        pacman_loc = env.robot_location
        #See if ghost can force a game over
        if (abs(pacman_loc[0] - state[0]) == 1 and not (pacman_loc[0] - state[0] == 0)):
                if (pacman_loc[1] == state[1]):
                    poss_moves.append(pacman_loc)

        elif (abs(pacman_loc[1] - state[1]) == 1 and not (pacman_loc[1] - state[1] == 0)):
            if (pacman_loc[0] == state[0]):
                poss_moves.append(pacman_loc) 
        
        #If we can force a game over
        if poss_moves:
            return poss_moves

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

def selection(weights, N, S):
    #Highest weight is at front
    sorted_weights = sorted(weights, key = lambda x: x[1], reverse = True)
    #sorted_weights = sorted(weights, key = lambda x: x)

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
        #final_samples.append(S[(np.random.choice(list(enumerate(sorted_weights)), p=sorted_probs))[0]])
        final_samples.append(sorted_weights[np.random.choice(len(sorted_weights), p = sorted_probs)])
    return final_samples

if __name__ == '__main__':
    main()