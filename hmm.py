#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time: Part 1 - HMM
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White

import numpy as np
import random
import pandas as pd

def main():
    #Make a HMM by unpacking the returns from the Ex17 function
    bookHMM = HMM(*Ex17())
    weatherHMM = HMM(*weather())
    print("\n--------------------Demonstrating the HMM and accessing different values.--------------------")
    print("States in the HMM:", bookHMM.get_states())
    print("Number of states in the HMM:", bookHMM.num_states())
    print("Transitional probability of going to states from the 'enough_sleep' state:", bookHMM.t_prob['enough_sleep'])
    print("\n")
   
    #Get a random set of observations
    bookObs = gen_ev(bookHMM, 3)
    weatherObs = gen_ev(weatherHMM, 3)
    
    #Run viterbi, passing in the HMM and a sequence of observations. Returns the most likely sequence
    #print("OBS", bookObs, "\n")
    MLS = viterbi(bookHMM, bookObs)
    print("The most likely sequence given observations of", bookObs, "is:", MLS, "\n")

    #print("OBS", weatherObs, "\n")
    MLS = viterbi(weatherHMM, weatherObs)
    print("The most likely sequence given observations of", weatherObs, "is:", MLS, "\n")


class HMM:
    def __init__(self, prior_prob, obs, t_prob, o_prob, prior_prob_matrix, t_prob_matrix, o_prob_matrix):
        self.prior_prob = prior_prob
        self.obs = obs
        self.t_prob = t_prob
        self.o_prob = o_prob
        self.prior_prob_matrix = prior_prob_matrix
        self.t_prob_matrix = t_prob_matrix
        self.o_prob_matrix = o_prob_matrix

    def get_states(self):
        return list(self.prior_prob.keys())

    def num_states(self):
        return len(self.prior_prob.keys())       

#Sets the probability tables and observations variables per Exercise 17 in the textbook
def Ex17():
    #Prior Probability
    prior_prob = {'enough_sleep' : 0.7, 'not_enough_sleep' : 0.3}
    prior_prob_matrix = pd.DataFrame([prior_prob])

    #Observation States
    obs =[['red_eyes', 'no_red_eyes'], ['sleeping_in_class', 'not_sleeping_in_class']]
    
    #Transition Probability
    t_prob = {
        'enough_sleep' : {'enough_sleep' : 0.8, 'not_enough_sleep' : 0.2},
        'not_enough_sleep' : {'enough_sleep' : 0.3, 'not_enough_sleep' : 0.7 }
    }
    t_prob_matrix = pd.DataFrame(t_prob).T.fillna(0)
    
    #Observation Probability
    o_prob = {
        'enough_sleep' : {'red_eyes' : 0.2, 'no_red_eyes' : 0.8, 'sleeping_in_class' : 0.1, 'not_sleeping_in_class' : 0.9},
        'not_enough_sleep' : {'red_eyes' : 0.7, 'no_red_eyes' : 0.3, 'sleeping_in_class' : 0.3, 'not_sleeping_in_class' : 0.7}
    }
  
    o_prob_matrix = pd.DataFrame(o_prob).T.fillna(0)

    return prior_prob, obs, t_prob, o_prob, prior_prob_matrix, t_prob_matrix, o_prob_matrix

def weather():
    #Prior Probability
    prior_prob = {'hot' : 0.8, 'cold' : 0.2}
    prior_prob_matrix = pd.DataFrame([prior_prob])

    #Observation States
    obs = ('1', '2', '3')
    
    #Transition Probability
    t_prob = {
        'hot' : {'hot' : 0.6, 'cold' : 0.4},
        'cold' : {'hot' : 0.5, 'cold' : 0.5}
    }
    t_prob_matrix = pd.DataFrame(t_prob).T.fillna(0)

    #Observation Probability
    o_prob = {
        'hot' : {'1' : 0.2, '2' : 0.4, '3' :0.4},
        'cold' : {'1' : 0.5, '2' :0.4, '3' : 0.1}
    }
    o_prob_matrix = pd.DataFrame(o_prob).T.fillna(0)

    return prior_prob, obs, t_prob, o_prob, prior_prob_matrix, t_prob_matrix, o_prob_matrix

#Generate a random set of <num> evidence variables based on the HMM
def gen_ev(HMM, num):
    ev = []
    for i in range(num):
        group = []
        for ob in HMM.obs:
            #If we have multiple evidence variables, we need to get a random value for each of them and combine into a tuple
            if isinstance(ob, list):
                group.append(random.choice(ob))
            else:
                #If we have one evidence variable, just randomly choose a value
                group.append(random.choice(HMM.obs))
                break
        ev.append(group)
    return ev

#Returns the most likely sequence in a HMM given observations
def viterbi(myHMM, ev):
    #Initialize our mt[xt]. Pad with a "None" to account for t needing to start at 1.
    m = ["None"]
    #Initialize our at[xt] for storing our transitions through the state space. Pad with a "None".
    #Want a in the form where each index is the time slice, and formed in {from_state:{to_state_0 : prob, to_state_1 : prob}, from_state:{...}}
    a = ["None"]

    ''' Forward Pass '''
    #Go through all of our observations.
    for t in range(1, len(ev)+1):
        
        #Create a dictionary that holds the transition probabilities across time slices
        temporal_t_prob = {}
        #Want our m to be a 2d list, with each index having another list of all the domain state values
        tmp_m = []

        #Compute probability of each state in our domain.
        for state in myHMM.get_states():
            #If this is the first iter of loop, we'll pull from prior prob table.
            if (t == 1):
                #We want: P(obs, state) = P(state) * P(obs|state)
                #Where P(state) is pulled from prior_prob since we want the t-1 state
                val = 1
                for e in ev[t-1]:
                    val *= myHMM.o_prob[state][e]
                tmp_m.append(myHMM.prior_prob[state] * val)

            else:
                #Init a tmp list that will hold the {to_state : prob} dicts
                tmp = []
                #Get the probability of transitioning to each other state in the domain from the current state
                for st in myHMM.get_states():
                    #           to_state : P(obs|to_state)      * P(to_state | from_state)
                    val = 1
                    for e in ev[t-1]:
                        val *= myHMM.o_prob[st][e]
                    tmp.append({st : (val * myHMM.t_prob[state][st])})

                temporal_t_prob[state] = tmp
        
        #Add all these transition probabilities to a.
        a.append(temporal_t_prob)
        
        #Now calculate the values at each state
        if t is not 1:
            for state in myHMM.get_states():
            #Init a list that hold all the values
                all_probs = []
                for st in myHMM.get_states():
                    #Multiply the t-1 state times the transition probability to the t state
                    all_probs.append(m[t-1][myHMM.get_states().index(st)] * temporal_t_prob[st][myHMM.get_states().index(state)][state])
                
                #All probs holds the values from all states in the domain, to the t state. We only want the max.  
                tmp_m.append(max(all_probs))

        m.append(tmp_m)

    ''' Backward Pass '''
    #Most Likely Sequence
    MLS = []
    #Go through our observations
    for t in range(1, len(ev)+1):
        #Get the max value from m, then convert that index to the associated HMM domain state.
        MLS.append(myHMM.get_states()[np.argmax(m[t])])

    return MLS

if __name__ == '__main__':
    main()