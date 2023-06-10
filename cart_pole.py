from collections import deque
import gym
from keras.layers import *
from keras.models import *
import numpy as np


class Agent:
    def __init__(self, n_states, n_actions, epsilon,
                 min_epsilon, epsilon_decay, memory_size,
                 batch_size, gamma, update_interval):
        self.n_states = n_states
        self.n_actions = n_actions

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, 2 * self.n_states + 3)) # we have 2 states(state & next state), 3 for reward,action,done
        self.memory_counter = 0

        self.batch_size = batch_size
        self.gamma = gamma

        self.train_counter = 0
        self.update_interval = update_interval

        self.value_model = self.build_model(name='value_model', do_compile=True) # we update just value & fit it so compile it
        self.target_model = self.build_model(name='target_model', do_compile=False) # we don't updata target so not to compile

    def build_model(self, name, do_compile=True):  # create nueral network model
        state = Input(shape=(self.n_states,), name='states')
        x = state
        x = Dense(units=256, activation='relu')(x)  # mlp
        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=256, activation='relu')(x)
        s_a_value = Dense(units=self.n_actions, activation='linear')(x)  # outputs ('linear' : we want value)
        model = Model(state, s_a_value, name=name)
        model.summary()
        if do_compile:
            model.compile(loss='mse',
                          optimizer='adam')
        return model

    def update_weights(self):
        self.target_model.set_weights(self.value_model.get_weights()) # get weights of value model & set to target model

    def act(self, state):
        if np.random.random() < self.epsilon:  # 0 < np.random.random() < 1
            action = np.random.randint(0, self.n_actions)  # select 0 or 1 (left or right)
        else:
            action = np.argmax(self.value_model.predict(np.expand_dims(state, axis=0))) # argmax = index of max value (0 or 1)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory[self.memory_counter % self.memory_size] = np.array(list(state) + [action, reward] + list(next_state) + [done]) # concat
        self.memory_counter += 1

    def get_batch(self):
        random_indices = np.random.randint(0, min(self.memory_size, self.memory_counter), size=self.batch_size)
        data = self.memory[random_indices]
        return data

    def update_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        epsilon = epsilon if epsilon > self.min_epsilon else self.min_epsilon
        self.epsilon = epsilon

    def train(self):
        if self.memory_counter < self.batch_size:
            print('not yet!!')
            return  # end of def
        data = self.get_batch()
        states = data[:, :self.n_states]
        actions = data[:, self.n_states].astype('int')
        rewards = data[:, self.n_states + 1]
        next_states = data[:, self.n_states + 2:-1]
        dones = data[:, -1]

        x = states
        target_value = np.max(self.target_model.predict(next_states), axis=1) * (1 - dones)
        target_value = target_value * self.gamma + rewards

        y = self.value_model.predict(x)
        y[np.arange(self.batch_size), actions] = target_value

        self.value_model.train_on_batch(x, y)

        self.train_counter += 1
        if self.train_counter % self.update_interval == 0:
            self.update_weights()
        self.update_epsilon()

    def save_weights(self):
        self.target_model.save_weights('cart_pole_dql.h5')

    def load_weights(self):
        self.target_model.load_weights('cart_pole_dql.h5')
        self.value_model.load_weights('cart_pole_dql.h5')


if __name__ == '__main__':

    # hyper-parameters
    train = False
    env_name = 'CartPole-v0'
    max_episodes = 3000  # number of episode
    epsilon = 1
    min_epsilon = 0.1
    epsilon_decay = 0.999
    memory_size = 5000
    batch_size = 64
    gamma = 0.9
    update_interval = 10
    print_intervals = 100
    win_rewards = 180
    scores = deque(maxlen=print_intervals)

    env = gym.make(env_name)    # create environment
    n_states = env.observation_space.shape[0]  # number of features in one state (angle,...) (state is input of neural network)
    n_actions = env.action_space.n  # number of actions in one state (left or right)

    agent = Agent(n_states, n_actions, epsilon,
                  min_epsilon, epsilon_decay, memory_size,
                  batch_size, gamma, update_interval)

    print('n_states: ', n_states)
    print('n_actions: ', n_actions)

    if train:
        for episode in range(1, max_episodes + 1):
            state = env.reset()  # in each episode, we reset environment
            done = False
            total_rewards = 0
            while not done:   # continue untill arive to terminal state
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_rewards += reward
            agent.train()

            scores.append(total_rewards)
            mean_score = np.mean(scores)
            if mean_score >= win_rewards and episode >= print_intervals:
                print("Solved in episode %d: Mean survival = %0.2lf in %d episodes"
                      % (episode, mean_score, print_intervals))
                print("Epsilon: ", agent.epsilon)
                agent.save_weights()
                break
            if episode % print_intervals == 0:
                print("Episode %d: Mean survival = %0.2lf, epsilon = %0.2f in %d episodes" %
                      (episode, mean_score, agent.epsilon, print_intervals))
    else:
        agent.epsilon = 0
        agent.load_weights()
        agent.min_epsilon = 0
        state = env.reset()
        for _ in range(1000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if done:
                env.reset()
            state = next_state

    env.close()
