import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(2)
tf.set_random_seed(2)

class actor():
    def __init__(self,env,sess):
        self.sess = sess
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.learning_rate = 0.001
        self.discount = 0.9
        self.sta = tf.placeholder(tf.float32,[1, self.state_size],name="state")
        self.act = tf.placeholder(tf.int32, None, name="action")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")
        init = tf.contrib.layers.xavier_initializer()

        self.l1 = tf.Variable(init([self.state_size,24]))
        self.dense = tf.nn.relu(tf.matmul(self.sta, self.l1))
        self.l2 = tf.Variable(init([24,self.action_size]))
        self.dense2 = tf.matmul(self.dense,self.l2)
        self.action_probability = tf.nn.softmax(self.dense2)

        log_probability = tf.log(self.action_probability[0,self.act])
        self.loss = tf.reduce_mean(log_probability*self.td_error)
 
        self.train_a = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.loss)

    def choose_action(self,state):
        state = state[np.newaxis, :]
        probability = self.sess.run(self.action_probability, {self.sta: state})
        return np.random.choice(np.arange(probability.shape[1]),p=probability.ravel())
    
    def learn_a(self,state,action,td):
        state = state[np.newaxis,:]
        _,expected_value = self.sess.run([self.train_a,self.loss],{self.sta:state,self.act:action,self.td_error:td})
        return expected_value

class critic():
    def __init__(self,env,sess):
        self.sess = sess
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.learning_rate = 0.01
        self.discount = 0.9
        self.sta = tf.placeholder(tf.float32,[1, self.state_size],name="state")
        self.value = tf.placeholder(tf.float32,[1,1],name="NextValue")
        self.reward = tf.placeholder(tf.float32, None, name="reward")
        init = tf.contrib.layers.xavier_initializer()

        self.l1 = tf.Variable(init([self.state_size,24]))
        self.dense = tf.nn.relu(tf.matmul(self.sta, self.l1))
        self.l2 = tf.Variable(init([24,1]))
        self.value_ = tf.matmul(self.dense,self.l2)

        self.td_error = self.reward + self.discount * self.value - self.value_
        self.loss_next = tf.square(self.td_error)   
        
        self.train_c = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_next)
    
    def learn_c(self,state,reward,state_next):
        state, state_next=state[np.newaxis,:],state_next[np.newaxis,:]
        value_ = self.sess.run(self.value_,{self.sta:state_next})
        td_error,_=self.sess.run([self.td_error,self.train_c],{self.sta:state,self.value:value_,self.reward:reward})
        return td_error

env = gym.make('CartPole-v0').env
env.seed(1)

sess = tf.Session()
actor = actor(env,sess)
critic = critic(env,sess)
sess.run(tf.global_variables_initializer())
reward_set = []
average_set = []
for episode in range(500):
    state = env.reset()
    reward_total=0
    t=0
    while True:
        #env.render()
        action = actor.choose_action(state)
        state_next, reward, done, info = env.step(action)
        if done: reward=-10
        td_error = critic.learn_c(state, reward, state_next)
        reward_total += reward
        actor.learn_a(state, action, td_error)    
        t+=1 
        if done or t>3000:
            reward_set.append(reward_total)
            average_set.append(sum(reward_set)/(episode+1))
            print('Episode:', episode, ' Reward:', reward_total)
            break
        state = state_next
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