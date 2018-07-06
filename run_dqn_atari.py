import argparse
import os.path as osp
import random

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from gym import wrappers
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

import dqn
from atari_wrappers import *
from dqn_utils import *
import argparse

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        _P = tf.placeholder(tf.float32, [None, 2, 6, 5], name="input_p")
        out = img_in
#        with tf.variable_scope("convnet"):
#            # original architecture
#            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
#            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
#            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
#        out = layers.flatten(out)
#        with tf.variable_scope("action_value"):
#            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
#            out = layers.fully_connected(out, num_outputs=256,         activation_fn=tf.nn.relu)
#            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        with tf.name_scope("VGG_Layer1"):
            VGG_Layer1_1 = tf.layers.conv2d(out, filters=int(64), kernel_size=[3, 3],
            padding='VALID', use_bias=False, name='VGG_Layer1_conv1')
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
            JOYSTICK_Layer = tf.contrib.layers.flatten(_P)
            VGG_Layer6_2 = tf.concat([VGG_Layer6_1, JOYSTICK_Layer], axis=1)
            VGG_Layer6_2 = tf.layers.dense(VGG_Layer6_2, units=100, use_bias=False)
            VGG_Layer6_2 = tf.nn.relu(VGG_Layer6_2)
            VGG_Layer6_3 = tf.layers.dense(VGG_Layer6_2, units=50, use_bias=False)
            VGG_Layer6_3 = tf.nn.relu(VGG_Layer6_3)
            out = tf.layers.dense(VGG_Layer6_3, units=num_actions, use_bias=False)
        return out


def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([(0, 1e-2 * lr_multiplier), (num_iterations / 10,  1e-2 * lr_multiplier), (num_iterations / 2,  5e-3 * lr_multiplier), ], outside_value=5e-5 * lr_multiplier)

    optimizer = dqn.OptimizerSpec(constructor=tf.train.AdamOptimizer, kwargs=dict(epsilon=1e-4),lr_schedule=lr_schedule)


    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1000, 0.1),
            (num_iterations , 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        learning_starts=33,
        learning_freq=64,
        frame_history_len=4,
        target_update_freq=5000,
        grad_norm_clipping=10
    )
    env.close()


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)


def get_session():
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1,gpu_options=gpu_options)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus());
    return session


def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    print( " path " , expt_dir)
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env


def main():
    # Get Atari games.
#    benchmark = gym.benchmark_spec('Atari40M')
#
#    # Change the index to select a different game.
#    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
    env.seed(seed)
    
    
    session = get_session()
    atari_learn(env, session, num_timesteps=5000)


if __name__ == "__main__":
    main()
