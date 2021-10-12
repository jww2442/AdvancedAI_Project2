#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time: Part 1 - HMM
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White

import numpy as np
import random

def main():
    #Make a HMM by unpacking the returns from the Ex17 function
    bookHMM = HMM(*Ex17())
    weatherHMM = HMM(*weather())
    flsHMM = Fixed_Lag_Smoothing()
    print("\n--------------------Demonstrating the HMM and accessing different values.--------------------")
    print("States in the HMM:", bookHMM.get_states())
    print("Number of states in the HMM:", bookHMM.num_states())
    print("Transitional probability of going to states from the 'enough_sleep' state:", bookHMM.t_prob['enough_sleep'])
    print("\n")
   
    #Get a random set of observations
    bookObs = gen_ev(bookHMM, 3)
    weatherObs = gen_ev(weatherHMM, 3)
    
    #Run FLS, passing in one set of observations at a time
    for ev in bookObs:
        state_prob = flsHMM.fixed_lag_smoothing(bookHMM, ev)
    print(state_prob)

    #Run viterbi, passing in the HMM and a sequence of observations. Returns the most likely sequence
    MLS = viterbi(bookHMM, bookObs)
    print("The most likely sequence given observations of", bookObs, "is:", MLS, "\n")

    #Run FLS, passing in one set of observations at a time
    for ev in weatherObs:
        state_prob = flsHMM.fixed_lag_smoothing(weatherHMM, ev)
    print(state_prob)

    #Run viterbi, passing in the HMM and a sequence of observations. Returns the most likely sequence
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

    def Normalize(self, dist):
        a = np.sum(dist)
        if a != 0:
            dist[0][0] /= a
            dist[1][1] /= a
        return dist  

#Sets the probability tables and observations variables per Exercise 17 in the textbook
def Ex17():
    #Prior Probability
    prior_prob = {'enough_sleep' : 0.7, 'not_enough_sleep' : 0.3}
    data = list(prior_prob.items())
    prior_prob_matrix = np.array(data)

    #Observation States
    obs =[['red_eyes', 'no_red_eyes'], ['sleeping_in_class', 'not_sleeping_in_class']]
    
    #Transition Probability
    t_prob = {
        'enough_sleep' : {'enough_sleep' : 0.8, 'not_enough_sleep' : 0.2},
        'not_enough_sleep' : {'enough_sleep' : 0.3, 'not_enough_sleep' : 0.7 }
    }
    data = list(t_prob.items())
    t_prob_matrix = np.array(data)
    
    #Observation Probability
    o_prob = {
        'enough_sleep' : {'red_eyes' : 0.2, 'no_red_eyes' : 0.8, 'sleeping_in_class' : 0.1, 'not_sleeping_in_class' : 0.9},
        'not_enough_sleep' : {'red_eyes' : 0.7, 'no_red_eyes' : 0.3, 'sleeping_in_class' : 0.3, 'not_sleeping_in_class' : 0.7}
    }
  
    data = list(o_prob.items())
    o_prob_matrix = np.array(data)
    
    return prior_prob, obs, t_prob, o_prob, prior_prob_matrix, t_prob_matrix, o_prob_matrix

def weather():
    #Prior Probability
    prior_prob = {'hot' : 0.8, 'cold' : 0.2}
    data = list(prior_prob.items())
    prior_prob_matrix = np.array(data)

    #Observation States
    obs = ('1', '2', '3')
    
    #Transition Probability
    t_prob = {
        'hot' : {'hot' : 0.6, 'cold' : 0.4},
        'cold' : {'hot' : 0.5, 'cold' : 0.5}
    }
    data = list(t_prob.items())
    t_prob_matrix = np.array(data)

    #Observation Probability
    o_prob = {
        'hot' : {'1' : 0.2, '2' : 0.4, '3' :0.4},
        'cold' : {'1' : 0.5, '2' :0.4, '3' : 0.1}
    }
    data = list(o_prob.items())
    o_prob_matrix = np.array(data)
    
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

class Deque:
    def __init__(self):
        self.queue = []
        self.count = 0

    def insert_init(self, data):
        if self.count == 0:
            self.queue = [data, ]
            self.count += 1
            return
        self.queue.insert(0, data)
        self.count += 1
        return

    def insert_end(self, data):
        if self.count == 0:
            self.queue = [data, ]
            self.count += 1
            return
        self.queue.append(data)
        self.count += 1
        return

    def remove_init(self):
        if self.count == 0:
            raise ValueError("Invalid Operation")
        x = self.queue.pop(0)
        self.count -= 1
        return x

    def remove_end(self):
        if self.count == 0:
            raise ValueError("Invalid Operation")
        x = self.queue.pop()
        self.count -= 1
        return x

    def get(self, index):
        if index > self.count | index < 0:
            raise ValueError("Index out of range")
        return self.queue[index-1]

    def size(self):
        return self.count

class Fixed_Lag_Smoothing():
    def __init__(self):
        self.e = Deque()
        self.t = 1
        self.f = None
        self.B = np.identity(2)
        self.d = 1

    def fixed_lag_smoothing(self, HMM, et):
        self.e.insert_end(et)
       
        Ot = self.get_prob_from_state(HMM, et)
        trans_matrix = self.get_prob_of_state(HMM)
        self.f = self.get_prior(HMM)

        if self.t > self.d:
            self.e.remove_init()
            Otmd = self.get_prob_from_state(HMM, self.e.get(0))
            self.f = Forward(self.f, Otmd, trans_matrix)
            self.B = np.linalg.inv(Otmd).dot(np.linalg.inv(trans_matrix)).dot(self.B).dot(trans_matrix).dot(Ot)
        else:
            self.B = self.B.dot(trans_matrix).dot(Ot)
        self.t += 1
        if self.t > (self.d + 1):
            return self.Normalize(self.f * self.B)
        else:
            return None

    def Normalize(self, dist):
        a = np.sum(dist)
        if a != 0:
            dist[0][0] /= a
            dist[1][1] /= a
        return dist

    def get_prob_from_state(self, HMM, state):
        st = HMM.get_states()
        z = np.identity(2)
        x = 1
        y = 1
        for i in range(len(state)):
            x *= HMM.o_prob[st[0]][state[i]]
            y *= HMM.o_prob[st[1]][state[i]]
        z[0][0] = x
        z[1][1] = y
        return z
 
    def get_prob_of_state(self, HMM):
        n = np.identity(2)
        st1 = HMM.get_states()[0]
        st2 = HMM.get_states()[1]
        n[0][0] = HMM.t_prob[st1][st1] * HMM.t_prob[st1][st2]
        n[1][1] = HMM.t_prob[st2][st1] * HMM.t_prob[st2][st2]
        return n

    def get_prior(self, HMM):
        n = np.identity(2)
        st1 = HMM.get_states()[0]
        st2 = HMM.get_states()[1]
        n[0][0] = HMM.prior_prob[st1]
        n[1][1] = HMM.prior_prob[st2]
        return n

def Forward(f, O, tprob):
    return O.dot(tprob.transpose().dot(f))

if __name__ == '__main__':
    main()