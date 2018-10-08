import gym
import itertools
import numpy as np
import csv
import tensorflow as tf
from tensorflow.contrib import layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def similarityCheck(obs, ment):
    # Check if the new state s' is similar to any state visited by the mentor.
    # The similarity threshold is set heuristicaly to 20% difference.
    sim_thres_perc = 0.2

    for i in range(len(ment)):
        comparisons = []
        m_s = ment[i]

        for j in range(len(m_s)):
            comparisons.append((m_s[j] > (1-sim_thres_perc)*obs[j]) and (m_s[j] < (1+sim_thres_perc)*obs[j]))
        if (all(comparisons)):
            print("found")
            print(m_s)
            print(obs)
            return i

    return -1

def augmentReward(rew, obs, ment, index):
    bias = 2000
    newReward = rew + 2000
    
    return newReward

if __name__ == '__main__':
    with U.make_session():
        # Create the environment
        env = gym.make("CartPole-v0")

        mentor_tr = []
        mentor_tr_tmp = []

        _gamma = 0.99

        print("--Creating mentor_demonstrations buffer..")
        
        with open('mentor_demonstrations_NN_0.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for r in data_reader:
                
                mentor_tr.append([float(i) for i in r])
        with open('mentor_demonstrations_NN_1.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for r in data_reader:
                mentor_tr.append([float(i) for i in r])
        with open('mentor_demonstrations_NN_2.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for r in data_reader:
                mentor_tr.append([float(i) for i in r])
        with open('mentor_demonstrations_NN_3.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for r in data_reader:
                mentor_tr.append([float(i) for i in r])
        with open('mentor_demonstrations_NN_4.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for r in data_reader:
                mentor_tr.append([float(i) for i in r])

        # Size of mentor's transitions
        N = len(mentor_tr)

        print("--Initializing mentor_actions buffer..")
        mentor_tr_actions = [None] * N

        # Create all the functions necessary to train the model
        act, train, trainAugmented, update_target, debug = deepq.build_train_imitation(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            gamma=_gamma,
        )
        # Create the replay buffer
        print("--Initializing experience replay buffer..")
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()

        t = 0
        while True:

            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            env.render()

            # Check if the new state s' is similar to any state visited by the mentor.
            similarityIndex = similarityCheck(obs, mentor_tr)
            if similarityIndex != -1:
                rew = augmentReward(rew, obs, mentor_tr, similarityIndex)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))

            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    if similarityIndex != -1:
                        trainAugmented(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    else:
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                        print(debug)
                        exit()
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
            
            t += 1
