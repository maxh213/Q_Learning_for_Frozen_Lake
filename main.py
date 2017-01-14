import gym
from gym import wrappers
import numpy as np
import secret_constants

NUMBER_OF_EPISODES = 5000
ATTEMPTS_AT_EACH_EPISODE = 1000
MOVE_UP = 3
MOVE_DOWN = 1
MOVE_RIGHT = 2
MOVE_LEFT = 0
LEARNING_RATE = 0.85
DISCOUNT_FACTOR = 0.99


def main():
	env = gym.make("FrozenLake-v0")
	env = wrappers.Monitor(env, "/tmp/gym-results", force=True)
	
	reward_list = []
	#16 * 4: 16 places on lake and 4 possible moves
	Q_TABLE = np.zeros([env.observation_space.n,env.action_space.n])

	for i in range(NUMBER_OF_EPISODES):
		state = env.reset()
		total_reward = 0
		for j in range(ATTEMPTS_AT_EACH_EPISODE):
			action = np.argmax(Q_TABLE[state,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
			env.render()
			new_state, reward, done, _ = env.step(action)
			#traing and update Q-table based on result
			ESTIMATE_OF_OPTIMAL_FUTURE_STATE_ACTION = np.max(Q_TABLE[new_state,:]) - Q_TABLE[state,action]
			Q_TABLE[state,action] = Q_TABLE[state,action] + LEARNING_RATE*(reward + (DISCOUNT_FACTOR*ESTIMATE_OF_OPTIMAL_FUTURE_STATE_ACTION))
			total_reward += reward
			state = new_state
			if done:
				env.render()
				break
		reward_list.append(total_reward)

	print("Average score per episode: " + str(sum(reward_list)/NUMBER_OF_EPISODES))
	print("Final Q-Table Values")
	print(Q_TABLE)

	env.close()
	gym.upload("/tmp/gym-results", api_key=secret_constants.API_KEY)

if __name__ == "__main__": main()


