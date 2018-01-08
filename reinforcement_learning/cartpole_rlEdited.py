import gym
import numpy as np

environment = gym.make('CartPole-v0')
environment.reset()
total_reward = 0
for step in range(1, 1001):
    environment.render()
    random_action = environment.action_space.sample()
    observation, reward, done, info = environment.step(random_action)
    total_reward += reward
    pos, veloc, angle, ang_veloc = observation
    if done: #if you either die : cart out of screen, pole tips more than 30(?) degrees
    #or if you win : when you keep your pole vertical
        print("The total reward was: {}".format(total_reward))
        print("This completed in {} steps".format(step))
        break

# This will create a vector of length 4 where each value is nitializing chosen randomly between [-1, 1].
weights = np.random.rand(4) * 2 - 1

#returns 0 (go left) or 1 (go right) depending on whether the weighted sum of weights*observation > 0
def determine_action(observation, weights):
	action = 0 if np.dot(observation, weights) < 0 else 1
	return action

#function that basically tells us how much a set of weights will reward us
#for a given set of weights we can calculate how "good" they are in our CartPole game
def run_episode(environment, weights):
	observation = environment.reset()
	totalreward = 0
	for step in range(200):
		action = determine_action(observation, weights)
		observation, reward, done, info = environment.step(action)
		totalreward += reward
		if done:
			break
	return totalreward

##############################################################################################
#straightforward strategy: keep trying random weights, and pick the one that performs the best
def find_best_weights(num_episodes):
    """
        This function runs a number of episodes and picks random weights for each one and evaluates
        the reward given by the weights.
        It returns the weights for the best episode.
    """
    best_weights = None
    best_reward = 0
    observation = environment.reset()
    for episode in range(num_episodes):
        weights = np.random.rand(4) * 2 - 1
        reward = run_episode(environment, weights)
        if reward > best_reward:
            best_weights = weights
            best_reward = reward
            print("Current Best Weights at episode #{} are {}".format(episode, best_weights))
    return best_weights, best_reward
       # CartPole is considered solved if the agent lasts 200 timesteps
       #if reward == 200:
           #break

##################################################################################################
# Now let's run 300 different weights and pick the best one (the one that gave the highest reward)
best_weights, best_reward = find_best_weights(num_episodes=300)
print("The best weights we've seen are: {}".format(best_weights))

# Now that we've found the best weights we've seen
# let's use our best weights to run our program
observation = environment.reset()
cumulative_reward = 0

for step in range(0, 200):
    environment.render()
    action = determine_action(observation, best_weights)
    observation, reward, done, info = environment.step(action)
    cumulative_reward += reward
    if done:
        print("Reward when done: {}".format(cumulative_reward))
        if cumulative_reward == 200:
            print("Congrats! You successfully solved Cartpole V-0!")
        else:
            print("Unfortunately, even the best weights weren't enough")
        break
################################################################################################
