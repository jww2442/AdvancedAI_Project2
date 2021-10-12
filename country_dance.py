import numpy as np

def main():

    f0 = np.array([
        0.7, 0.3
    ])

    transition_mx = np.array([
        [0.8, 0.2], 
        [0.3, 0.7]
    ])

    #these evidence matrices have been hard coded in a method and so aren't accessed here
    red_eyes_prob = np.array([
        [0.2, 0], 
        [0, 0.7]
    ])
    sleeping_in_class_prob = np.array([
        [0.1, 0],
        [0, 0.3]
    ])
   
    obs_vals_list = [
        {'ERR: THERE IS NO EVIDENCE ON DAY 0'}, #skip this line in code
        {'red_eyes': False, 'sleeping_in_class': False},
        {'red_eyes': True, 'sleeping_in_class': False},
        {'red_eyes': True, 'sleeping_in_class': True},
    ]

    fv, sv, b, ev_mxs = country_dance(transition_mx, obs_vals_list, f0)
    print(fv)
    print(sv)
    print(b)
    print(ev_mxs)


#forward-backward algorithm as in Figure 14.4 of textbook
def country_dance(trans_mx, ev_mx, first_state_mx):
    
    fv = [first_state_mx]
    ev_mxs = [None]
    for i in range(1, len(ev_mx)):
        ev_for_one_state = make_evidence_mx(ev_mx[i])
        ev_mxs.append(ev_for_one_state)
        fv.append(_normalize(_forward(ev_for_one_state, trans_mx, fv[i-1])))
        
    sv = []
    b = [np.array([1, 1])] * len(ev_mx)

    for i in reversed(range(1, len(ev_mx))):

        sv.insert(0, _normalize(np.multiply(fv[i], b[i])))

        b[i - 1] = _normalize(_backward(ev_mxs[i], trans_mx, b[i]))
        
    return fv, sv, b, ev_mxs
    
  
#helper function to analyze observation vars of a day and make an evidence matrix  
def make_evidence_mx(obs_vals, red_eyes_prob = 0, sleeping_in_class_prob = 0):

    if(red_eyes_prob == 0 and sleeping_in_class_prob == 0):
        red_eyes_prob = np.array([
            [0.2, 0], 
            [0, 0.7]
        ])
        sleeping_in_class_prob = np.array([
            [0.1, 0],
            [0, 0.3]
        ])

    elif(red_eyes_prob == 0 or sleeping_in_class_prob == 0): 
        print('err ONLY ONE MATRIX SET TO 0')
        exit()

    mx_to_dot = np.array([
        [1, 0],
        [0, 1]
    ])

    if(obs_vals.get('red_eyes')):
        mx_to_dot = mx_to_dot.dot(red_eyes_prob)
    else:
        mx_to_dot2 = np.array([
            [1-red_eyes_prob[0][0], 0],
            [0, 1-red_eyes_prob[1][1]]
        ])
        mx_to_dot = mx_to_dot.dot(mx_to_dot2)
    
    if(obs_vals.get('sleeping_in_class')):
        mx_to_dot = mx_to_dot.dot(sleeping_in_class_prob)
    else:
        mx_to_dot2 = np.array([
            [1-sleeping_in_class_prob[0][0], 0],
            [0, 1-sleeping_in_class_prob[1][1]]
        ])
        mx_to_dot = mx_to_dot.dot(mx_to_dot2)
    
    return mx_to_dot

#forward algorithm from equation 14.5 in book
def _forward(obs_mx, transition_mx, last_state):
    
    new_state_prob_not_normalized = obs_mx.dot(np.transpose(transition_mx)).dot(last_state)

    return new_state_prob_not_normalized

#backward algorithm from equation 14.9 in book
def _backward(obs_mx, transition_mx, after_state):
        
        last_state_prob_not_normed = transition_mx.dot(obs_mx).dot(after_state)

        return last_state_prob_not_normed
  
#rturns normalized list of two values. order is maintained
def _normalize(arr):

    l = np.array([
        arr[0]/(arr[0]+arr[1]), arr[1]/(arr[0]+arr[1])
    ])

    return l

if(__name__ == '__main__'):
    main()