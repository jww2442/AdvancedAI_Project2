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
    parser.add_argument("-t", "--time-steps", dest = "TIME_STEPS", default = 100, type = int, help = "Number of time steps.")
    parser.add_argument("-s", "--samples", dest = "SAMPLES", default = 42, type = int, help = "Number of samples to take.")
    parser.add_argument("-g", "--ghosts", dest = "NUM_GHOSTS", default = 0, type = int, help= "Number of ghosts to spawn.")
    args = parser.parse_args()

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
    sample_list = []
    '''
    #Loop through a total of t time steps
    for t in range(args.TIME_STEPS):    
        #sample_list = []
        samples, env, state_id = pf(args.SAMPLES, env, samples, ghosts)
        
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
        
        if samples == "Game Over":
            print("GAME OVER, YOU LOSE")
            return
        print(samples)
        #most_likely = max(samples, key=samples.get)
        most_likely = max(state_id, key=state_id.count)
        prob = samples[most_likely]
        #print("Most likely state at time t is", most_likely, "with a probability of", prob)
        #print("We are actually at:", env.robot_location)
        #print()
        actual_state = env.robot_location
        if most_likely == actual_state:
            print("Correct state prediction.")
        else:
            print("Incorrect state prediction.")
        to_write = [args.ACTION_BIAS, args.OBSERVATION_NOISE, args.ACTION_NOISE, args.NUM_GHOSTS, t, most_likely, prob]
        append_csv(to_write)

    print(env.location_transitions)
    '''
    times_correct = 0
    for i in range(args.TIME_STEPS):
        samples = testing(args.SAMPLES, env, samples, ghosts)
        print()
        all_keys = list(samples.keys())
        curr_max = 0
        curr_cell = 0
        curr_heading = 0
        #print(all_keys)
        print()
        for k in all_keys:
            #print(k[0])
            if k[1]/(samples[k]/args.SAMPLES) > curr_max:
                curr_max = k[1]/(samples[k]/args.SAMPLES)
                curr_cell = (k[0][0], k[0][1])
                curr_heading = (k[0][2])

        print(samples)
        print("Guessing we are in cell", curr_cell, "and oriented as", curr_heading)
        print("Probability of this state",curr_max)
        print("We are actually in", env.robot_location, "and oriented as", env.robot_heading)
        if curr_cell == env.robot_location:
            times_correct += 1

    print("Times correct:", times_correct) 

    #print(list(samples.keys())[0])

def testing(N, env, samples = None, ghosts = None):
    #samples = {}
    #Init S with P(X0)
    S = {}
    for i in range(42):
        for cell in env.location_priors:
            for head in env.heading_priors:
                S[(cell[0],cell[1],head)] = env.location_priors[cell] * env.heading_priors[head]

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


    if(samples):
        env.update(sample_list, env.heading_priors)
    else:
        env.update(loc_list, env.heading_priors)
    observation = env.move()
    flag = 0
    if(samples):
        flag = 1
    new_samples = {}
    #Take N samples    
    for i in range(42):
        #choice = random.choice(list(env.location_transitions[cell[0]][cell[1]][head].items()))
        if (flag):
            choice = random.choice(list(samples.keys()))
        else:
            choice = random.choice(list(S.items()))
        #print(choice) 
        if choice not in new_samples:
        #if choice not in samples:
            #samples[choice] = 1
            new_samples[choice] = 1
        else:
            new_samples[choice] += 1
            #samples[choice] += 1
    print()
    #print(samples)
    #all_keys = list(samples.keys())
    all_keys = list(new_samples.keys())
    for k in all_keys:
        as_list = list(k)
        #val = samples[k]
        #del samples[k]
        val = new_samples[k]
        del new_samples[k]
        #samples.remove(k)
        as_list[1] = env.observation_tables[k[0][0]][k[0][1]][tuple(observation)]

        k = tuple(as_list)
        new_samples[k] = val
        #samples[k] = val

    #print(samples)
    #return samples
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

def pf(N, env, samples = None, ghosts=None):
    #Init S to empty list
    S = []

    if(samples):
        S.append(samples)
    else:
        #Init S[0] with priors
        S.append(env.location_priors)

    #Init weights to 0
    w = [0 for _ in range(N)]

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

    #Increment to the t+1 state. Probability table for t+1 is auto-updated with the env.update() fcn.
    #env.update(loc_list, env.heading_priors)
    if(samples):
        env.update(sample_list, env.heading_priors)
    else:
        env.update(loc_list, env.heading_priors)
    observation = env.move()
    #Ghosts move
    if ghosts:
        for ghost in ghosts:
            check = ghost.move(env)
            if check:
                return "Game Over", 0, 0

    #Save the probability table for later use
    #S.append(env.location_priors)
    #Sort in descending order based on key
    #sorted_S = sorted(S[-1], key=S[-1].get, reverse=True)
    sorted_S = {k: v for k, v in sorted(S[-1].items(), key=lambda item: item[1])}
 
    #Flag as a lazy way to keep track of when we need to use the last item in dict
    flag = 0
    #Running total of weights for normalization
    w_tot = 0
    #ID what state we are assigning
    state_id = []
    state_dict = {}
    for i in range(1, N):
        rand_num = random.random()
        for j in sorted_S.keys():
            #Get most likely robot position and see if it passes rng
            #if (rand_num <= sorted_S[j].value()):
            if(rand_num <= sorted_S[j]):
                #If it does, update our weights: P(X|e) * P(e)
                w[i] = sorted_S[j] * env.observation_tables[j[0]][j[1]][tuple(observation)]
                state_id.append(j)
                if j in state_dict:
                    state_dict[j] += 1
                else:
                    state_dict[j] = 1
                flag = 1
                #If we have identified the state we're in, break so that we don't keep overwriting w[i]
                break
        #If we've gone through entire dictionary and still haven't decided
        if (flag == 0):
            #Get x coord of last key in current time slice
            x_coord = list(S[-1].keys())[-1][0]
            #Get y coord
            y_coord = list(S[-1].keys())[-1][1]
            #Get the probability of being in that x,y cell
            prob = sorted_S[(x_coord, y_coord)]
            #Github code returns observation as list, but observation_tables requires it to be a tuple
            obs = tuple(observation)
            w[i] = prob * env.observation_tables[x_coord][y_coord][obs]
            
            state_id.append(list(S[-1].keys())[-1])
            if list(S[-1].keys())[-1] in state_dict:
                state_dict[list(S[-1].keys())[-1]] += 1
            else:
                state_dict[list(S[-1].keys())[-1]] = 1

        #Add the weight to a running total of all the weights
        w_tot += w[i]

    # Normalize all the weights
    for i in range(N):
        if(w[i] == 0):
            w[i] = w[i]
        else:
            w[i] = w[i] / w_tot

    #state_id = weighted_sample_replacement(N, state_id, w, env)
    
    #samples = {}
    samples = env.location_priors

    #Get corresponding probability
    #for state in state_id:
    #    samples[state] = S[-1][state]
    for kv in state_dict:
        samples[kv] = state_dict[kv]/N
    print("SAMPLES")
    print(samples)
    return samples, env, state_id


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
    samples = selection(collection, N, S)
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

if __name__ == '__main__':
    main()