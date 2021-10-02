#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White

import numpy as np

def main():
    #Make a HMM by unpacking the returns from the Ex17 function
    myHMM = HMM(*Ex17())
    print("\n--------------------Demonstrating the HMM and accessing different values.--------------------")
    print("States in the HMM:", myHMM.get_states())
    print("Number of states in the HMM:", myHMM.num_states())
    print("Transitional probability of going to states from the 'enough_sleep' state:", myHMM.t_prob['enough_sleep'])
    print()
   
    #Run viterbi, passing in the HMM and a sequence of observations
    viterbi(myHMM, ['red_eyes', 'red_eyes', 'sleeping_in_class'])


class HMM:
    def __init__(self, prior_prob, obs, t_prob, o_prob):
        self.prior_prob = prior_prob
        self.obs = obs
        self.t_prob = t_prob
        self.o_prob = o_prob

    def get_states(self):
        return list(self.prior_prob.keys())

    def num_states(self):
        return len(self.prior_prob.keys())       

#Sets the probability tables and observations variables per Exercise 17 in the textbook
def Ex17():
    #Prior Probability
    prior_prob = {'enough_sleep' : 0.7, 'not_enough_sleep' : 0.3}
    #Observation States
    obs = ('red_eyes', 'sleeping_in_class')
    #Transition Probability
    t_prob = {
        'enough_sleep' : {'enough_sleep' : 0.8, 'not_enough_sleep' : 0.2},
        'not_enough_sleep' : {'enough_sleep' : 0.3, 'not_enough_sleep' : 0.7 }
    }
    #Observation Probability
    o_prob = {
        'enough_sleep' : {'red_eyes' : 0.2, 'sleeping_in_class' : 0.1},
        'not_enough_sleep' : {'red_eyes' : 0.7, 'sleeping_in_class' : 0.3}
    }

    return prior_prob, obs, t_prob, o_prob

#Returns the most likely sequence in a HMM given observations
def viterbi(myHMM, ev):
    #Copy the ev for insertions/deletions and to recursively pass through correctly.
    ev = ev.copy()
    #Initialize our mt[xt]. Pad with a "None" to account for t needing to start at 1.
    m = ["None"]
    #Initialize our at[xt] for storing our transitions through the state space. Pad with 2 "None"s since at[xt] will not be filled until t=2+.
    a = ["None", "None"]
    #a = {[0.0] for dummy in range(len(ev) -1)}

    ''' Forward Pass '''
    #Go through all of our observations.
    for t in range(1, len(ev)):
        #Compute probability of each state in our domain.
        for state in myHMM.get_states():
            #If this is the first iter of loop, we'll pull from prior prob table.
            if (t == 1):
                #We want: P(obs, state) = P(state) * P(obs|state)
                #Where P(state) is pulled from prior_prob since we want the t-1 state
                m.append(myHMM.prior_prob[state] * myHMM.o_prob[state][ev[t-1]])
                
            else:
                #We need to take the argmax and store in our transition sequence. Init a tmp dict of {state : val}
                tmp = {}
                #For each state, multiply all the elements in the t_prob table by the mt-1 value
                for st in myHMM.get_states():
                    tmp[st] = myHMM.t_prob[state][st] * m[t-1]
                max_key = max(tmp, key=tmp.get)

                #Store [state, value] into the at[xt] list
                a.append([max_key, tmp[max_key]])

                #TODO:
                #m.apppend(myHMM.o_prob[state][ev[t-1]] * )

    ''' Backward Pass '''
               

    return 0

if __name__ == '__main__':
    main()