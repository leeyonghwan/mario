# original DQN 2015 source from  https://github.com/nalsil/kimhun_rl_windows/blob/master/07_3_dqn_2015_cartpole.py
# The code is updated to play super mario by Jinman Chang
# super mario game can be downloaded at https://github.com/ppaquette/gym-super-mario

# ##### is marked where is updated
# explanation for this code is at http://jinman190.blogspot.ca/2017/10/rl.html


###############################################################################super mario initialized
import gym
import ppaquette_gym_super_mario



from gym.envs.registration import register
from gym.scoreboard.registration import add_group
from gym.scoreboard.registration import add_task


register(
     id='SuperMarioBros-1-1-v0',
     entry_point='gym.envs.ppaquette_gym_super_mario:MetaSuperMarioBrosEnv',
)

add_group(
     id='ppaquette_gym_super_mario',
     name='ppaquette_gym_super_mario',
     description='super_mario'
)

add_task(
    id='SuperMarioBros-1-1-v0',
    group='ppaquette_gym_super_mario',
    summary="SuperMarioBros-1-1-v0"
)
#################################################################################




import numpy as np
import tensorflow as tf
import random
from collections import deque

from gym import wrappers

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')                                             #####update game title

# Constants defining our neural network
input_size = np.array([env.observation_space.shape[0], env.observation_space.shape[1], 15])#env.observation_space.shape[0]*env.observation_space.shape[1]*3        #####change input_size - 224*256*3 acquired from ppaquette_gym_super_mario/nes_env.py
output_size = 6                                                                     #####meaning of output can be found at ppaquette_gym_super_mario/wrappers/action_space.py
_skip = 4
dis = 0.9
REPLAY_MEMORY = 50000

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
#        with tf.variable_scope(self.net_name):
#            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
#
#            # First layer of weights
#            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
#                                 initializer=tf.contrib.layers.xavier_initializer())
#            layer1 = tf.nn.relu(tf.matmul(self._X, W1))
#
#            # Second layer of Weights1
#            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
#                                 initializer=tf.contrib.layers.xavier_initializer())
#
#            # Q prediction
#            self._Qpred = tf.matmul(layer1, W2)
#
        self._l_rate = tf.placeholder(tf.float32)
        self._X = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1], self.input_size[2]], name="input_x")
        self._P = tf.placeholder(tf.float32, [None, 2, 6, 5], name="input_p")

        with tf.variable_scope(self.net_name):
            self._l_rate = tf.placeholder(tf.float32)
            self._X = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1], self.input_size[2]], name="input_x")
            self._P = tf.placeholder(tf.float32, [None, 2, 6, 5], name="input_p")
    
            with tf.name_scope("VGG_Layer1"):
                VGG_Layer1_1 = tf.layers.conv2d(self._X, filters=int(64), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer1_conv1')
                VGG_Layer1_1 = tf.nn.relu(VGG_Layer1_1)
                VGG_Layer1_2 = tf.layers.conv2d(VGG_Layer1_1, filters=int(64), kernel_size=[3, 3], strides=[2, 2],padding='VALID', use_bias=False, name='VGG_Layer1_conv2')
                VGG_Layer1_2 = tf.nn.relu(VGG_Layer1_2)
            # shape (B, h, w, 64)->(B, h/2, w/2, 64)
            with tf.name_scope("VGG_Layer2"):
     ########################################################################################################
                VGG_Layer2_1 = tf.layers.conv2d(VGG_Layer1_2, filters=int(128), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer2_conv1')
                VGG_Layer2_1 = tf.nn.relu(VGG_Layer2_1)  # shape (B, h/2, w/2, 64)->(B, h/2, w/2, 128)
    ########################################################################################################
                VGG_Layer2_2 = tf.layers.conv2d(VGG_Layer2_1, filters=int(128), kernel_size=[3, 3], strides=[2, 2],padding='VALID', use_bias=False, name='VGG_Layer2_conv2')
                VGG_Layer2_2 = tf.nn.relu(VGG_Layer2_2)  # shape (B, h/2, w/2, 128)->(B, h/4, w/4, 128)
    ########################################################################################################
            with tf.name_scope("VGG_Layer3"):
    ########################################################################################################
                VGG_Layer3_1 = tf.layers.conv2d(VGG_Layer2_2, filters=int(256), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer3_conv1')
                VGG_Layer3_1 = tf.nn.relu(VGG_Layer3_1)  # shape (B, h/4, w/4, 128)->(B, h/4, w/4, 256)
    ########################################################################################################
                VGG_Layer3_2 = tf.layers.conv2d(VGG_Layer3_1, filters=int(256), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer3_conv2')
                VGG_Layer3_2 = tf.nn.relu(VGG_Layer3_2)  # shape (B, h/4, w/4, 256)->(B, h/4, w/4, 256)
    ########################################################################################################
                VGG_Layer3_3 = tf.layers.conv2d(VGG_Layer3_2, filters=int(256), kernel_size=[3, 3], strides=[2, 2],padding='VALID', use_bias=False, name='VGG_Layer3_conv3')
                VGG_Layer3_3 = tf.nn.relu(VGG_Layer3_3)  # shape (B, h/4, w/4, 256)->(B, h/8, w/8, 256)
    ########################################################################################################
    ########################################################################################################
            with tf.name_scope("VGG_Layer4"):
    ########################################################################################################
                VGG_Layer4_1 = tf.layers.conv2d(VGG_Layer3_3, filters=int(512), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer4_conv1')
                VGG_Layer4_1 = tf.nn.relu(VGG_Layer4_1)  # shape (B, h/8, w/8, 256)->(B, h/8, w/8, 512)
    ########################################################################################################
                VGG_Layer4_2 = tf.layers.conv2d(VGG_Layer4_1, filters=int(512), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer4_conv2')
                VGG_Layer4_2 = tf.nn.relu(VGG_Layer4_2)  # shape (B, h/8, w/8, 512)->(B, h/8, w/8, 512)
    ########################################################################################################
                VGG_Layer4_3 = tf.layers.conv2d(VGG_Layer4_2, filters=int(512), kernel_size=[3, 3], strides=[2, 2],padding='VALID', use_bias=False, name='VGG_Layer4_conv3')
                VGG_Layer4_3 = tf.nn.relu(VGG_Layer4_3)  # shape (B, h/8, w/8, 512)->(B, h/16, w/16, 512)
    ########################################################################################################
    ########################################################################################################
            with tf.name_scope("VGG_Layer5"):
    ########################################################################################################
                VGG_Layer5_1 = tf.layers.conv2d(VGG_Layer4_3, filters=int(512), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer5_conv1')
                VGG_Layer5_1 = tf.nn.relu(VGG_Layer5_1)  # shape (B, h/16, w/16, 512)->(B, h/16, w/16, 512)
    ########################################################################################################
                VGG_Layer5_2 = tf.layers.conv2d(VGG_Layer5_1, filters=int(512), kernel_size=[3, 3],padding='VALID', use_bias=False, name='VGG_Layer5_conv2')
                VGG_Layer5_2 = tf.nn.relu(VGG_Layer5_2)  # shape (B, h/16, w/16, 512)->(B, h/16, w/16, 512)
    ########################################################################################################
                VGG_Layer5_3 = tf.layers.conv2d(VGG_Layer5_2, filters=int(512), kernel_size=[3, 3], strides=[2, 2],padding='VALID', use_bias=False, name='VGG_Layer5_conv3')
                VGG_Layer5_3 = tf.nn.relu(VGG_Layer5_3)  # shape (B, h/16, w/16, 512)->(B, h/32, w/32, 512)
    ########################################################################################################
    ########################################################################################################
            with tf.name_scope("VGG_Qpred"):
                VGG_Layer6_1 = tf.layers.conv2d(VGG_Layer5_3, filters=100, kernel_size=[2, 3], strides=[1, 1], padding='VALID')
                VGG_Layer6_1 = tf.nn.relu(VGG_Layer6_1)
                VGG_Layer6_1 = tf.contrib.layers.flatten(VGG_Layer6_1)
                JOYSTICK_Layer = tf.contrib.layers.flatten(self._P)
                VGG_Layer6_2 = tf.concat([VGG_Layer6_1, JOYSTICK_Layer], axis=1)
                VGG_Layer6_2 = tf.layers.dense(VGG_Layer6_2, units=100, use_bias=False)
                VGG_Layer6_2 = tf.nn.relu(VGG_Layer6_2)
                VGG_Layer6_3 = tf.layers.dense(VGG_Layer6_2, units=50, use_bias=False)
                VGG_Layer6_3 = tf.nn.relu(VGG_Layer6_3)
                self._Qpred = tf.layers.dense(VGG_Layer6_3, units=self.output_size, use_bias=False)

#        self.inference = out
#        self._Qpred = tf.argmax(self.inference, 1)
        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size[0], self.input_size[1], self.input_size[2]])
        action_seq = np.reshape(action_seq, [1, 2, 6, 5])
        return self.session.run(self._Qpred, feed_dict={self._X: x, self._P: action_seq})

    # def update(self, x_stack, y_stack):
    #     return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
    def update(self, x_stack, y_stack, action_seq, l_rate = 1e-5):
        #x_stack = np.reshape(x_stack, (-1, self.input_size))
        x_stack = np.reshape(x_stack, (-1, self.input_size[0], self.input_size[1], self.input_size[2]))
        action_seq = np.reshape(action_seq, [-1, 2, 6, 5])
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack, self._P: action_seq,  self._l_rate: l_rate})

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predic(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack( [x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def ddqn_replay_train(mainDQN, targetDQN, train_batch, l_rate):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)
    action_stack = np.empty(0).reshape(0, 60)
    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        if state is None:                                                               #####why does this happen?
            print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
        else:
            Q = mainDQN.predict(state)

            # terminal?
            if done:
                Q[0, action] = reward
            else:
                # Double DQN: y = r + gamma * targetDQN(s')[a] where
                # a = argmax(mainDQN(s'))
                # Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
                Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state)) #####use normal one for now

            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])])   #####change shape to fit to super mario
            action_stack = np.vstack([action_stack, np.reshape(action_seq, (-1, 60))])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack, action_stack, l_rate = l_rate)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def _step(action):
    total_reward = 0.0
    done = None
    _obs_buffer = deque(maxlen=2)
    for _ in range(_skip):
        obs, reward, done, info = env.step(action)
        _obs_buffer.append(obs)
        total_reward += reward
        if done:
            break

    max_frame = np.max(np.stack(_obs_buffer), axis=0)
    return max_frame, total_reward, done, info


def main():
    max_episodes = 5000
    # store the previous observations in replay memory
    replay_buffer = deque()
    # saver = tf.train.Saver()
    
    max_distance = 0
    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / 20) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e or state is None or state.size == 1:           #####why does this happen?
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    #action = np.argmax(mainDQN.predict(state))
                    action = mainDQN.predict(state).flatten().tolist()                  #####flatten it and change it as a list
#                    for i in range(len(output)):                                        #####the action list has to have only integer 1 or 0
#                        if action[i] > 0.5 :
#                            action[i] = 1                                               #####integer 1 only, no 1.0
#                        else:
#                            action[i] = 0                                               #####integer 0 only, no 0.0

#                action = OutputToAction3(output)
#                print("random action:", output, "output " , action)

                # Get new state and reward from environment
                next_state, reward, done, info = _step(action)
                
                #print("Episode: {} ".format(next_state));
                
                if info['distance'] > max_distance:
                    max_distance = info['distance']
                
                if done: # Penalty
                    reward = -100
                    print("distance ", max_distance, " current distance ",info['distance'] )

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:   # Good enough. Let's move on
                    break

            print("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000:
                pass
                # break

            if episode % 10 == 1: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)

                    l_rate = (1e-5 - 1e-4) * (1 / max_episodes) * episode + 1e-4
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch, l_rate=l_rate)

                print("Loss: ", loss)
                # copy q_net -> target_net
                sess.run(copy_ops)

        # See our trained bot in action
        env2 = wrappers.Monitor(env, 'gym-results', force=True)
        
        #save_path = saver.save(sess, "./mario_model_1")
    
        for i in range(200):
            bot_play(mainDQN, env=env2)

        env2.close()
        # gym.upload("gym-results", api_key="sk_VT2wPcSSOylnlPORltmQ")

if __name__ == "__main__":
    main()






