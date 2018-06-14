# -*- coding: utf-8 -*-
import random
import time as temp #
import gym
import cv2
import numpy as np
from collections import deque
# from matplotlib.pyplot import imshow as show
import collections
import itertools
import keras
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, dot
from keras.optimizers import Adam, RMSprop, Adadelta

BANDWIDTH = 7
futures = 3

def _build_model():
        Qfunc = Sequential()
        action = Input(shape=(4,), name='action')
        screen = Input(shape=(7, 80, 80), name='screen')
        Qfunc.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu',
                         input_shape=(BANDWIDTH, *(80,80)),
                         data_format="channels_first"))

        Qfunc.add(Conv2D(64, (4, 4), strides=(2, 2),
                         activation='relu',
                         data_format="channels_first"))

        Qfunc.add(Conv2D(64, (4, 4), activation='relu',
                         data_format="channels_first"))
        Qfunc.add(MaxPooling2D(pool_size=(2, 2)))
        Qfunc.add(Flatten())
        Qfunc.add(Dense(128, activation='relu'))
        Qfunc.add(Dense(128, activation='relu'))
        Qfunc.add(Dense(4))

        reward = Qfunc(screen)

        model = Model(inputs=[screen, action], outputs=dot([reward, action], axes=[1, 1]))

        model.compile(loss='mse', optimizer=RMSprop(lr=0.00001)) # was RMSprop(0.00001)

        return model, Qfunc

class DQNAgent:
    def __init__(self):
        self.state_size = (210, 160, 3)
        self.action_size = 4
        self.memory = deque(maxlen=140000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        # self.learning_rate = 0.001
        self.model, self.Qfunc = _build_model()
        self.avgQ = 0
        self.cacheQ = {}

    def restate(self, img):
        balls = cv2.resize(img, (80,80))
        balls = cv2.cvtColor(balls, cv2.COLOR_RGB2GRAY)
        balls = balls*0.1
        return balls

    def rewardcalc(self,rewardlist):
        comp = len(rewardlist[14:])
        comp2 = len(rewardlist[7:14])
        return (np.dot(rewardlist[14:], [(1/(comp - i)) for i in range(comp)])) + (np.dot(rewardlist[7:14], [(1/(comp2 - i)) for i in range(comp2)]))# np.sum(rewardlist[7:14])

    def remember(self, state, action, reward, done):
        reward = self.rewardcalc(reward)
        if len(action) == 7*(futures + 1):
            self.memory.append((state[:7], action[6], reward, done))


    def act(self, state, action):
        # if np.random.rand() <= self.epsilon:
        #     return np.random.rand(4)

        act_values = self.Qfunc.predict(state[None,-7:])[0]
        if np.array_equal(act_values, action):
            return np.random.rand(4)
        # print(act_values)
        if type(act_values[0]) == 'nan':
            print('ruim')
        # print(act_values)
        return act_values  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        lista = []
        lista2 = []
        lista3 = []
        for state, action, reward, isplaying in minibatch:
            lista.append(state)
            lista2.append(action)
            lista3.append(reward)# + np.argmax(self.Qfunc.predict(state[None, :]).ravel()))

        loss = self.model.train_on_batch({'screen': np.array(lista), 'action': np.array(lista2)}, np.array(lista3))
        count = math.sqrt(math.sqrt(loss))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = 0.2

        if len(self.memory) > 5000:
            while count < loss:
                loss = self.model.train_on_batch({'screen': np.array(lista), 'action': np.array(lista2)}, np.array(lista3))

            return
        print('Not yet')
        # print('treinou')

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class info:
    def __init__(self):
        self.imagem = deque(maxlen=BANDWIDTH*(futures+1))
        self.acoes = deque(maxlen=BANDWIDTH*(futures+1))
        self.recompensa = deque(maxlen=BANDWIDTH*(futures+1))
        self.done = deque(maxlen=BANDWIDTH*(futures+1))

    def refresha(self,infos):
        self.imagem.append(np.array(infos[0], dtype=np.uint8))
        self.acoes.append(infos[1])
        self.recompensa.append(infos[2])
        self.done.append(infos[3])

    def remember(self):
        return np.array(self.imagem), np.array(self.acoes), np.array(self.recompensa), np.array(self.done)

def scorecalc(vida,ponto):
    return vida + ponto

listamed = deque(maxlen=100)

def media(scoretotal):
    global listamed
    listamed.append(scoretotal)
    return sum(listamed)/100

informacoesinhas = info()

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent()
    batch_size = 32
    # gym.train(agent,10)
    agent.load("Breakout-joga8-Dv4.h5")
    e = 0
    while True:
        state = env.reset()
        time = 0
        scoretotal = 0
        initialvida = 5
        done = False
        reward = []
        while not done:
            time += 1
            # if (e%250 == 0) or (scoretotal >= 3 and agent.epsilon < 0.05):
            env.render()
            temp.sleep(0.05)

            if time < 15:
                action = np.random.rand(4)
                next_state, ponto, done, vidas = env.step(np.argmax(action))
                vida = (vidas["ale.lives"] - initialvida)
                mandascore = scorecalc(vida,ponto)
                initialvida += vida
                scoretotal += ponto
                informacoesinhas.refresha([agent.restate(state),action,mandascore,not done])
                a, b, c, d = informacoesinhas.remember()
                agent.remember(a,b,c,d)
                state = next_state
            else:
                action = agent.act(informacoesinhas.remember()[0], action)
                # print(action)
                next_state, ponto, done, vidas = env.step(np.argmax(action).ravel())
                vida = (vidas["ale.lives"] - initialvida)
                mandascore = scorecalc(vida,ponto)
                initialvida += vida
                scoretotal += ponto
                informacoesinhas.refresha([agent.restate(state),action,mandascore,not done])
                a, b, c, d = informacoesinhas.remember()
                agent.remember(a,b,c,d)
                state = next_state
                if done:
                    break
        print("episode: {}, time: {}, media: {}, score: {}, e: {}"
              .format(e, time, media(scoretotal), scoretotal, agent.epsilon))
        if (len(agent.memory) > batch_size):
            agent.replay(batch_size)
        if e % 50 == 0 and e != 0:
            agent.save("Breakout-joga7-Dv4.h5")
        e += 1
