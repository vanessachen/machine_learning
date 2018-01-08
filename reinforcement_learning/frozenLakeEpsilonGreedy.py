import numpy as np
import gym

env = gym.make("FrozenLake-v0")

#just testing to get a feel for the states and actions in the game
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_states)
print(n_actions)

#env.render()
render = True

###########################################################how well it does just by taking random actions
state = env.reset()

reward = None
done = None

g = 0
episodes = 0
rewardTracker = []

while reward != 1:
    if render:
        env.render()
    state, reward, done, info = env.step(env.action_space.sample()) #finds the state,reward,done,info based on the step function for a possible action
    g += reward
    if done == True: #done is true if you win or if you die
        rewardTracker.append(g)
        state = env.reset() #reset state to beginning state
        episodes += 1
print("Reached goal (once) after {} episodes with a average return of {} with random chance".format(episodes, sum(rewardTracker)/len(rewardTracker)))


##############################add reward discounting and decaying epsilon for better choices
epsilon = 1 #these are for the complex functions in the epsilon greedy function
#exploration rate, which is the rate the agent does a random action vs a prediction
gamma = 0.95

rewardTracker = []
Q = np.zeros([n_states, n_states]) #returns an array of zeros with dimensions of the number of states
episodes = 5000
G = 0
alpha = 0.618

for episode in range(1,episodes+1):
    done = False
    G = 0
    reward = 0
    state = env.reset()
    while not done: #while running game
        # if render:
        #     env.render()
        if np.random.rand() > epsilon: #take an action based on Q matrix described above
            action = np.argmax(Q[state])
        else: #want to decrease epsilon after agent gets better at playing
            action = env.action_space.sample()
            epsilon -= 10**-3

        state2, reward, done, info = env.step(action)
        Q[state,action] += alpha * ((reward + gamma * (np.max(Q[state2])) -  Q[state,action])) #complex Q function that I looked up that finds the maximum reward given a state and an action
        G += reward #keeping track of reward
        state = state2 #moving to next state
    rewardTracker.append(G) #adding to reward

if (sum(rewardTracker[episode-100:episode])/100.0) > .78: #this means frozenLake has been Solved (avg > .78 over 100 episodes)
            print('-------------------------------------------------------')
            print('Solved after {} episodes with average return of {}'.format(episode-100, sum(rewardTracker[episode-100:episode])/100.0))

print("Average return of {} with decaying epsilon in {} episodes".format(sum(rewardTracker)/len(rewardTracker), episode)) #average return after running the above function (5000 episodes)
#this generally returns a reward of around .6

###################################################### Define better epsilon greedy policy for Q function

def e_greedy(eps, Q, state, episode):

    if np.random.rand() > eps:
        #optimal action based on Q function defined above if the random number between 0 and 1 was greater than epsilon (which starts at 1)
        action = np.argmax(Q[state,:]+np.random.randn(1, n_actions)/(episode/4))
    else:
        action = env.action_space.sample() #otherwise,take a random action (exploration)
        eps -= 10**-5 #decrease exploration rate

    return action, eps

#################################### Define Optimal Hyper policy_parameters
'''
General Idea of Q function:
- takes in a state and action like: Q(s,a) and returns the reward for taking an action in that state
-
'''
def learn_Q(alpha, gamma, eps, numTrainingEpisodes, numTrainingSteps):

    global Q_star
    Q = np.zeros([env.observation_space.n, env.action_space.n]) #this gives a matrix of 0's of dimensions states and actions
    rewardTracker = []

    for episode in range(1,numTrainingEpisodes+1): #train the Q function based on numTrainingEpisodes

        G = 0 #this var will keep track of the rewards
        state = env.reset()

        for step in range(1,numTrainingSteps): #train the model in this many steps
            action, eps = e_greedy(eps, Q, state, episode) #using the e_greedy function defined above that returns epsilon and action
            state2, reward, done, info = env.step(action) #for each action taken in the env, there are 4 variables provided to help train the agent
            Q[state,action] += alpha * (reward + gamma * np.max(Q[state2]) - Q[state,action]) #this is the Q function that tries to maximize reward given a state and action
            state = state2 #moving to next state
            G += reward #append to G

        rewardTracker.append(G)

        if episode % (numTrainingEpisodes*.01) == 0 and episode != 0: #update on average reward every 50 episodes c(before solving the game)
            render = True
            if render:
                env.render()
            print('Alpha {}  Gamma {}  Epsilon {:04.3f}  Episode {} of {}'.format(alpha, gamma, eps, episode, numTrainingEpisodes))
            print("Average Total Return: {}".format(sum(rewardTracker)/episode))
        else:
            render = False #don't want it to render during the entire training process
        if (sum(rewardTracker[episode-100:episode])/100.0) > .78: #this is when frozen lake is defined as solved
            render = True
            if render:
                env.render()
            print('-------------------------------------------------------')
            print('Solved after {} episodes with average return of {}'.format(episode-100, sum(rewardTracker[episode-100:episode])/100.0))
            Q_star = Q
            break
    Q_star = Q #setting Q_star equal to Q to evaluate the function found

# Alpha, Gamma, Eps, Episodes, Steps per Episode for learning the Q function
Alpha = 0.8
Gamma = 0.95
Epsilon = 0.1
Episodes = 5000
Steps_per_Episode = 300

learn_Q(Alpha, Gamma, Epsilon, Episodes, Steps_per_Episode)

#######################################################Evaluate Model After learning the Q Function
def evaluate(Q, numTrainingEpisodes, numTrainingSteps, render): #similar to the training step but it gives update on performance after 'learning' Q function
    print("After learning Q function: ")

    rewardTracker = []

    for episode in range(1,numTrainingEpisodes+1):

        G = 0
        state = env.reset()

        for step in range(1,numTrainingSteps):

            action = np.argmax(Q[state])

            state2, reward, done, info = env.step(action)
            state = state2
            G += reward
            if render == True:
                env.render()

            if done == True:
                break

        rewardTracker.append(G)

        if episode % (numTrainingEpisodes*.05) == 0 and episode != 0: #every 100 episodes

            #print("Average Total Return After {} Episodes: {:04.3f}".format(episode, sum(rewardTracker)/episode))
            print("Average Total Return After {} Episodes: {:04.4f}".format(episode, sum(rewardTracker[episode-100:episode])/100.0))

############################################
numTrainingEpisodes = 2000
numTrainingSteps = 500
evaluate(Q_star, numTrainingEpisodes, numTrainingSteps, False) #increasing numTrainingSteps greatly increases performance, but it is based on numTrainingEpisodes also
