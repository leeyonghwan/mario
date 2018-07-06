
# coding: utf-8

# In[1]:


import gym
import ppaquette_gym_super_mario
import dqn


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

import numpy as np
import tensorflow as tf
import random
from collections import deque

from gym import wrappers
import tensorflow.contrib.layers as layers

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')  

frame_history_len=4
# Constants defining our neural network
#input_size = env.observation_space.shape[0]*env.observation_space.shape[1]*3        #####change input_size - 224*256*3 acquired from ppaquette_gym_super_mario/nes_env.py
img_h, img_w, img_c = env.observation_space.shape

print("inpusize[0]", env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])

input_size = np.array([env.observation_space.shape[0], env.observation_space.shape[1], 15]) #width*height*3ch#(img_h, img_w, frame_history_len * img_c)
# set up placeholders
# placeholder for current observation (or state)
#obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
output_size = 6                                                                     #####meaning of output can be found at ppaquette_gym_super_mario/wrappers/action_space.py

_skip = 4
dis = 0.9
REPLAY_MEMORY = 50000

state_buffer = deque()
next_state_buffer = deque()
output_buffer = deque()


# In[2]:


a_0 = [0, 0, 0, 0, 0, 0]  # NOOP\n",
a_1 = [1, 0, 0, 0, 0, 0]  #Up\n",
a_2 = [0, 0, 1, 0, 0, 0] # Down\n",
a_3 = [0, 1, 0, 0, 0, 0] # Left\n"
a_4 = [0, 1, 0, 0, 1, 0] # Left + A\n",
a_5 = [0, 1, 0, 0, 0, 1]  # Left + B\n",
a_6 = [0, 1, 0, 0, 1, 1]  # Left + A + B\n",
a_7 = [0, 0, 0, 1, 0, 0] # Right\n",
a_8 = [0, 0, 0, 1, 1, 0] # Right + A\n",
a_9 = [0, 0, 0, 1, 0, 1]  # Right + A + B\n",
a_10 = [0, 0, 0, 1, 1, 1]  #Right + A + B\n",
a_11 = [0, 0, 0, 0, 1, 0]
a_12 = [0, 0, 0, 0, 0, 1]
a_13 = [0, 0, 0, 0, 1, 1]

def encode_action_rand():
    action = random.randrange(0,14)
    if action == 0:
        return a_0
    if action == 1:
        return a_1
    if action == 2:
        return a_2
    if action == 3:
        return a_3
    if action == 4:
        return a_4
    if action == 5:
        return a_5
    if action == 6:
        return a_6
    if action == 7:
        return a_7
    if action == 8:
        return a_8
    if action == 9:
        return a_9
    if action == 10:
        return a_10
    if action == 11:
        return a_11
    if action == 12:
        return a_12
    if action == 13:
        return a_13

def encode_action( action):
    if action == 0:
        return a_0
    if action == 1:
        return a_1
    if action == 2:
        return a_2
    if action == 3:
        return a_3
    if action == 4:
        return a_4
    if action == 5:
        return a_5
    if action == 6:
        return a_6
    if action == 7:
        return a_7
    if action == 8:
        return a_8
    if action == 9:
        return a_9
    if action == 10:
        return a_10
    if action == 11:
        return a_11
    if action == 12:
        return a_12
    if action == 13:
        return a_13

# def decode_action( input):
#     if a_0 == input:
#         return 0
#     if a_1 == input:
#         return 1
#     if a_2 == input:
#         return 2
#     if a_3 == input:
#         return 3
#     if a_4 == input:
#         return 4
#     if a_5 == input:
#         return 5
#     if a_6 == input:
#         return 6
#     if a_7 == input:
#         return 7
#     if a_8 == input:
#         return 8
#     if a_9 == input:
#         return 9
#     if a_10 == input:
#         return 10
#     if a_11 == input:
#         return 11
#     if a_12 == input:
#         return 12
#     if a_13 == input:
#         return 13



# In[3]:



def ddqn_replay_train(mainDQN, targetDQN, train_batch, l_rate):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
#     x_stack = np.empty(0).reshape(0, mainDQN.input_size)
#     y_stack = np.empty(0).reshape(0, mainDQN.output_size)

#     # Get stored information from the buffer
#     for state, action, reward, next_state, done in train_batch:
#         if state is None:                                                               #####why does this happen?
#             print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
#         else:
#             Q = mainDQN.predict(state, action)

#             # terminal?
#             if done:
#                 Q[0, action] = reward
#             else:
#                 # Double DQN: y = r + gamma * targetDQN(s')[a] where
#                 # a = argmax(mainDQN(s'))
#                 # Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
#                 Q[0, action] = reward + dis * targetDQN.predict(next_state, action_next_seq)[0, np.argmax(mainDQN.predict(next_state, action_next_seq))] # np.max(targetDQN.predict(next_state, action_seq)) #####use normal one for now

#             y_stack = np.vstack([y_stack, Q])
#             x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size)])   #####change shape to fit to super mario
#             action_stack = np.vstack([action_stack, np.reshape(action, (-1, 60))])
#     # Train our network using target and predicted Q values on each episode
#     return mainDQN.update(x_stack, y_stack, action_stack)
    x_stack = np.empty(0).reshape(0, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)
    action_stack = np.empty(0).reshape(0, 30)

    # Get stored information from the buffer
    for state, action_seq, action_next_seq, action, reward, next_state, done  in train_batch:
        Q = mainDQN.predict(state, action_seq)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] =  reward + dis * targetDQN.predict(next_state, action_next_seq)[0, np.argmax(mainDQN.predict(next_state, action_next_seq))]

        if state is None:
            print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
        else:
            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])])
            action_stack = np.vstack([action_stack, np.reshape(action_seq, (-1, 30))])
     #   y_stack = np.vstack([y_stack, Q])
    #    x_stack = np.vstack( [x_stack, state])

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



# In[4]:


def main():
    max_episodes = 5000
    # store the previous observations in replay memory
    replay_buffer = deque()
    # saver = tf.train.Saver()
    
    max_distance = 0
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        print(" input ", input_size[2])
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
                    output = encode_action_rand()#env.action_space.sample()
                    action = output
                else:
                    # Choose an action by greedily from the Q-network
                    #action = np.argmax(mainDQN.predict(state))
                    output = mainDQN.predict(acc_state, output_seq) 
                    output = np.argmax(output) #####flatten it and change it as a list .flatten().tolist()
                    action = encode_action( output)
#                    for i in range(len(output)):                                        #####the action list has to have only integer 1 or 0
#                        if action[i] > 0.5 :
#                            action[i] = 1                                               #####integer 1 only, no 1.0
#                        else:
#                            action[i] = 0                                               #####integer 0 only, no 0.0

#                next_action = OutputToAction3(output)
#                print("random action:", output) 

                # Get new state and reward from environment
                next_state, reward, done, info = _step(action)
                
                state_buffer.append(next_state)
                output_buffer.append(action)

                #print("Episode: {} ".format(next_state));
                
                if info['distance'] > max_distance:
                    max_distance = info['distance']
                
                if done: # Penalty
                    reward = -100
                    print("distance ", max_distance, " current distance ",info['distance'] )

                # Save the experience to our buffer
                
                if step_count>=10:
                    acc_state = [state_buffer[-2-k] for k in range(5)]

                    state_buffer.popleft()
                    acc_state = np.reshape(acc_state, (input_size[0], input_size[1], input_size[2]))

                    acc_next_state = [state_buffer[-1-k] for k in range(5)]
                    acc_next_state = np.reshape(acc_next_state, (input_size[0], input_size[1], input_size[2]))

                    output_seq = [output_buffer[-2-k] for k in range(5)]
                    output_next_seq = [output_buffer[-1-k] for k in range(5)]
                    output_buffer.popleft()
                                       
                    replay_buffer.append((acc_state, output_seq, output_next_seq, output, reward, acc_next_state, done))
                    
#                     if replay_buffer[-1][6]: #if done==true?
#                         for k in range(1, 5):
#                             replay_buffer[-1 - k] = tuple(replay_buffer[-1 - k][0:4] + (-pow(0.9, k),) + replay_buffer[-1 - k][5:])
                    
#                     if replay_buffer[-1][4] >= 2.0 and replay_buffer[-1][6] == False:
#                         for k in range(1, 5):
#                             replay_buffer[-1 - k] = tuple(replay_buffer[-1 - k][0:4] + (pow(0.9, k),) + replay_buffer[-1 - k][5:])
                            
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                    acc_state = acc_next_state
                    
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
                    l_rate =(1e-5 -1e-4)*(1/max_episodes)*episode + 1e-4
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch, l_rate)
                  
                        
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



