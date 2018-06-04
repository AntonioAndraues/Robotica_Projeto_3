# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.optimizers import Adam

EPISODES = 5000


class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95	# discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.9999
		self.learning_rate = 0.0001
		self.model = self._build_model()

	# def baseline_model(grid_size,num_actions,hidden_size):
	# #seting up the model with keras
	#	 model = Sequential()
	#	 model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
	#	 model.add(Dense(hidden_size, activation='relu'))
	#	 model.add(Dense(num_actions))
	#	 model.compile(sgd(lr=.1), "mse")
	#	 return model
	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding="same", input_shape=(1000, 1000, 3),data_format="channels_first"))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding="same", data_format="channels_first"))
		model.add(Conv2D(64, (4, 4), activation='relu', padding="same",data_format="channels_first"))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(128, activation='relu'))
		# model.add(Dense(1024, input_dim=self.state_size, activation='relu'))
		# model.add(Dense(512, activation='relu'))
		# model.add(Dense(256, activation='relu'))
		# model.add(Dense(128, activation='relu'))
		# model.add(Dense(64, activation='relu'))
		# model.add(Dense(32, activation='relu'))
		# model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

import time as tempo

if __name__ == "__main__":
	env = gym.make('Breakout-v0')
	env._max_episode_steps = 5000
	state_size = 100800
	# print(env.observation_space)
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)
	# agent.load("./save/cartpole-dqn.h5")
	done = False
	batch_size = 32
	e = 0

	while True:
		contawin = 0
		state = env.reset()
		state = np.reshape(state, [1,100800])
		for time in range(10000):
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			if e%50 == 0:
				env.render()
				tempo.sleep(0.05)
			reward = reward if not done else -1
			contawin += reward
			# print(reward)
			next_state = np.reshape(next_state, [1,100800])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
					  .format(e, EPISODES, contawin, agent.epsilon))
				break
		if len(agent.memory) > batch_size:
			agent.replay(batch_size)
		if e % 500 == 0:
			agent.save("Breakout-dqn.h5")
		e += 1