from cmath import log
import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps      #Alphas
    forward_messages[0] = prior_distribution        #initial distribution
    backward_messages = [None] * num_time_steps     #Betas
    marginals = [None] * num_time_steps             #Alphas*Betas
    
    # TODO: Compute the forward messages
    #compute distribution for forward_messages[0]
    dict0 = rover.Distribution()
    obsModel = rover.observation_model(observations[0])   #may encounter an error here cause need to give some random state?
    for key, value in prior_distribution.items():
        newDictValue = value * obsModel[key[0:2]]
        if(newDictValue != 0):
            dict0[key] = newDictValue

    forward_messages[0] = dict0

    #compute distribution for forward messages[1:num_time_steps]
    for i in range(1, num_time_steps):
        newDict = rover.Distribution()
        allPossibleStates = rover.get_all_hidden_states()
        for testState in allPossibleStates:
            cumSum = 0
            obsModel = rover.observation_model(testState)
            for key, value in forward_messages[i-1].items():
                transModel = rover.transition_model(key)
                if(observations[i] == None):
                    cumSum += value * 1 * transModel[testState]
                else:
                    cumSum += value * obsModel[observations[i]] * transModel[testState]

            if(cumSum > 0):
                newDict[testState] = cumSum

        newDict.renormalize()
        forward_messages[i] = newDict
                   
    # TODO: Compute the backward messages
    #initialize betas of all states at end time to 1
    stateList = rover.get_all_hidden_states()
    initialBeta = rover.Distribution()
    for key in stateList:
        initialBeta[key] = 1
    backward_messages[-1] = initialBeta
    
    #do formula
    #Something not working here - figure that mf out mf mf mf mf mf mf 
    for i in range(num_time_steps - 2, -1, -1):
        newDict = rover.Distribution()
        allPossibleStates = rover.get_all_hidden_states()
        for testState in allPossibleStates:
            cumSum = 0
            transModel = rover.transition_model(testState)
            for key, value in backward_messages[i + 1].items():
                obsModel = rover.observation_model(key)
                if(observations[i+1] == None):
                    cumSum += value * 1 * transModel[key]
                else:
                    cumSum += value * obsModel[observations[i + 1]] * transModel[key]

            if(cumSum > 0):
                newDict[testState] = cumSum

        #print(newDict)
        newDict.renormalize()
        backward_messages[i] = newDict
    
    # TODO: Compute the marginals 
    for i in range(num_time_steps):
        marginalDict = rover.Distribution()
        for key in forward_messages[i]:
            marginalDict[key] = forward_messages[i][key]*backward_messages[i][key]
        marginalDict.renormalize()
        marginals[i] = marginalDict

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    #mf viterbi algorithm
    pathList = [None]*num_time_steps
    messages = [None]*num_time_steps
    maxList = [None]*num_time_steps
    argMaxList = [None]*num_time_steps

    #Based case
    dict0 = rover.Distribution()
    obsModel = rover.observation_model(observations[0])
    for key, value in prior_distribution.items():
        f1 = np.log(value) + np.log(obsModel[key[0:2]])
        dict0[key] = f1
    messages[0] = dict0

    for i in range(1, num_time_steps):

        dict = rover.Distribution()
        path = rover.Distribution()

        for prevState, value in messages[i-1].items():
            transModel = rover.transition_model(prevState)
            for state in transModel:
                obsModel = rover.observation_model(state)
                if observations[i] != None:
                    logProbability = value + np.log(transModel[state]) + np.log(obsModel[observations[i]])
                else:
                    logProbability = value + np.log(transModel[state]) + np.log(1)

                if logProbability > dict.get(state, -np.inf):
                    dict[state] = logProbability
                    path[state] = prevState

        messages[i] = dict
        pathList[i] = path

    maxState = max(messages[-1], key = messages[-1].get)

    estimated_hidden_states = []
    estimated_hidden_states.append(maxState)

    for i in range(num_time_steps - 1, 0, -1):
        maxState = pathList[i][maxState]
        estimated_hidden_states.append(maxState)

    estimated_hidden_states.reverse()
    return estimated_hidden_states

def countStateMisses(hiddenStateList, predictionList):
    numMisses = sum(hiddenStateList[i] != predictionList[i] for i in range(len(hiddenStateList)))
    probMissing = numMisses/len(hiddenStateList)
    return probMissing

def findInvalidStep(maxMarginals):
    for i in range(len(maxMarginals)):
        transModel = rover.transition_model(maxMarginals[i])
        if maxMarginals[i+1] not in transModel:
            return i, maxMarginals[i], maxMarginals[i + 1]

if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    timestep = 30 #num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    maxMarginals = []
    for i in range(len(marginals)):
        maxMarginals.append(max(marginals[i], key = marginals[i].get))

    print("Error probibility for marginals is: ", countStateMisses(hidden_states, maxMarginals))
    print("Error probibility for viterbi is: ", countStateMisses(hidden_states, estimated_states))

    indexOfImpossibility, startState, nextState = findInvalidStep(maxMarginals)

    print("Impossible step in marginal calculation: ")
    print("Index of step", indexOfImpossibility, "start step: ", startState, "next step: ", nextState)

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
