#Project 2 for the University of Tulsa'ss  CS-7313 Adv. AI Course
#Probabilistic Reasoning Over Time: Part 2 - DBN
#Professor: Dr. Sen, Fall 2021
#Noah Schrick, Noah Hall, Jordan White


from CS5313_Localization_Env import localization_env as le

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

    myDBN = DBN(1,1,1,"homogeneous")

class DBN:
    def __init__(self, prior_prob, t_model, sensor_model, topology):
        self.prior_prob = prior_prob
        self.t_model = t_model
        self.sensor_model = sensor_model
        self.topology = topology


#Returns the set of empty squares that are adjacent to i
def neighbors(i):
    raise (NotImplementedError)

if __name__ == '__main__':
    main()