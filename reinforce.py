import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

np.random.seed(1)
tf.set_random_seed(1)

class REINFORCE:
    def __init__(self,env):
        self.env = env
        self.learning_rate = 0.02
        self.discount = 0.99
        self.states, self.actions, self.rewards = [], [], []

        self.build_net()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        self.tf_states = tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0]], name="states")
        self.tf_actions = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        init = tf.contrib.layers.xavier_initializer()

        self.l1 = tf.Variable(init([self.env.observation_space.shape[0],24]))
        self.dense = tf.nn.relu(tf.matmul(self.tf_states, self.l1))
        self.l2 = tf.Variable(init([24,self.env.action_space.n]))
        self.dense2 = tf.matmul(self.dense,self.l2)
        self.outputs_softmax = tf.nn.softmax(self.dense2)
        
        neg_log_probability = tf.reduce_sum(-tf.log(self.outputs_softmax)*tf.one_hot(self.tf_actions, self.env.action_space.n), axis=1)
        loss = tf.reduce_mean(neg_log_probability * self.tf_vt) 

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
    
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def choose_action(self, state):
        state = state[np.newaxis,:]
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.tf_states: state})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  
        return action

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.rewards)
        cumulative = 0
        for t in reversed(range(len(self.rewards))):
            cumulative = cumulative * self.discount + self.rewards[t]
            discounted_episode_rewards[t] = cumulative
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def learn(self):
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
             self.tf_states: np.vstack(self.states), 
             self.tf_actions: np.array(self.actions),  
             self.tf_vt: discounted_episode_rewards_norm, 
        })

        self.states, self.actions, self.rewards = [], [], []
        return discounted_episode_rewards_norm
     
env = gym.make('CartPole-v0').env
env.seed(1)     
reward_set = []
average_set = []
RL = REINFORCE(env)
for episode in range(500):
    t=0
    state = env.reset()
    reward_total = 0
    while True:
        #env.render()
        action = RL.choose_action(state)
        next_state, reward, done, info = env.step(action)
        if done: reward = -10
        reward_total += reward
        RL.store_transition(state, action, reward)
        t+=1
        if done or t>3000:
            reward_set.append(reward_total)
            average_set.append(sum(reward_set)/(episode+1))
            print('Episode:', episode, ' Reward:', reward_total)
            discounted_episode_rewards_norm = RL.learn()
            break
        state = next_state
#print('-----',len(reward_set))
episode = np.arange(0,500,1)
plt.plot(episode,reward_set)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()
episode = np.arange(0,500,1)
plt.plot(episode,average_set)
plt.xlabel('episode')
plt.ylabel('average_reward')
plt.show()