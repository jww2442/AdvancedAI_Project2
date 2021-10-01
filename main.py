#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White

def main():
    print("Hello")

    #Make a HMM by unpacking the returns from the Ex17 function
    myHMM = HMM(*Ex17())
    print(myHMM.get_states())
    print(myHMM.num_states())

    viterbi(0, 0)


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
    #Observations
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
    return 0

if __name__ == '__main__':
    main()