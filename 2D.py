# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

BANDWIDTH = 7
EPISODES = 100000

class info:
    def __init__(self):
        self.imagem = []
        self.acoes = []
        self.recompensa = []
        self.nexts = []
        self.done = []

    def refresha(self,infos):
        if len(self.imagem) >= BANDWIDTH:
            self.imagem = self.imagem[1:]
            self.acoes = self.acoes[1:]
            self.recompensa = self.recompensa[1:]
            self.nexts = self.nexts[1:]
            self.done = self.done[1:]

        self.imagem.append(infos[0])
        self.acoes.append(infos[1])
        self.recompensa.append(infos[2])
        self.nexts.append(infos[3])
        self.done.append(infos[4])

    def remember(self):
        # print(np.array(self.imagem))
        # print(np.array(self.acoes))
        # print(np.array(self.recompensa))
        # print(np.array(self.nexts))
        # print(np.array(self.done))
        return np.array(self.imagem), np.array(self.acoes), np.array(self.recompensa), np.array(self.nexts), np.array(self.done)

    def rewarda(self):
        return sum(self.recompensa)





class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu',
                         input_shape=(210, 160, 3),
                         data_format="channels_last"))

        model.add(Conv2D(64, (4, 4), strides=(2, 2),
                         activation='relu',
                         data_format="channels_last"))

        model.add(Conv2D(64, (4, 4), activation='relu',
                         data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(150, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(80, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
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
            if not done[-1]:
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

caralho = info()
import time as temp

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    state_size = (3, 210, 160)#100800#env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("Breakout-5-essevai500.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        time = 0
        scorai = 0
        while time < 10000:
            time += 1

            if time < 15:
                if e%50 == 0:
                    env.render()
                    temp.sleep(0.05)
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -5
                scorai += reward
                # next_state = np.reshape(next_state, [1, state_size])
                caralho.refresha([state,action,reward,next_state,done])
                state = next_state
            else:
                if e%50 == 0:
                    env.render()
                    temp.sleep(0.05)
                action = agent.act(caralho.remember()[0])
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -5
                scorai += reward
                # next_state = np.reshape(next_state, [1, state_size])
                caralho.refresha([state,action,reward,next_state,done])
                a, b, c, d, e_ = caralho.remember()
                agent.remember(a, b, c, d, e_)
                state = next_state
                if done:
                    print("episode: {}/{}, time: {}, score: {}, e: {}"
                          .format(e, EPISODES, time, scorai, agent.epsilon))
                    break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 100 == 0 and e != 0:
            agent.save("Breakout-5-essevai{}.h5".format(e + 500))
