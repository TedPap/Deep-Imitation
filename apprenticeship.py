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

from my_imports.expert_feature_expectation import EFE
from my_imports.agent_feature_expectation import AFE
from my_imports.weight_optimization import OPT

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session():
        # Create the environment
        env = gym.make("CartPole-v0")

        exp_demo = []
        exp_demo_tmp = []
        N = 1000
        _gamma = 0.99

        e = 0.1
        # Apprenticeship counter
        i = 0
        # Apprenticeship reward function weights
        weights = []
        # Apprenticeship distance to expert
        hyperdistance = 100
        min_hyperdistance = 100
        # Agent list of policies
        policies = []
        # Agent feature expectations list
        afe_list = []
        # Agent feature expectations
        afe = AFE()
        # Expert feature expectations
        efe = EFE()
        # Projections of agent's feature expectations
        proj_afe_list = []
        # Optimization of weigths and t
        opt = OPT()

        with open('mentor_demonstrations_NN_0.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            exp_demo_tmp = ([r for r in data_reader])
            exp_demo.append(exp_demo_tmp)
        with open('mentor_demonstrations_NN_1.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            exp_demo_tmp = ([r for r in data_reader])
            exp_demo.append(exp_demo_tmp)
        with open('mentor_demonstrations_NN_2.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            exp_demo_tmp = ([r for r in data_reader])
            exp_demo.append(exp_demo_tmp)
        with open('mentor_demonstrations_NN_3.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            exp_demo_tmp = ([r for r in data_reader])
            exp_demo.append(exp_demo_tmp)
        with open('mentor_demonstrations_NN_4.csv', 'r', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            exp_demo_tmp = ([r for r in data_reader])
            exp_demo.append(exp_demo_tmp)
            N1 = len(exp_demo_tmp)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            gamma=_gamma,
        )
        # Create the replay buffer
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
            # Apprenticeship reward
            if t + 1 > 1000:
                rew = np.dot(weights, obs)
                if rew < 0:
                    rew = 0
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            env.render()
            

            policies.append(obs)

            if (((t + 1) % N == 0) and (t != 0) and (i == 0)) or (((t + 1) % (N * 50) == 0) and (t != 0) and (i != 0)):
                    # Step 1 OR Steps 5 &
                    print("Policies length: ", len(policies))
                    print("Policies indexes: ", t+1-N, t)
                    afe_list.append(afe.comp_afe(N-1, _gamma, policies[t+1-N:t]))
                    proj_afe_list.append(np.zeros(len(obs)))
                    i += 1
                    # Step 2
                    weights, hyperdistance = (opt.optimize(proj_afe_list, efe.comp_efe(5, N, _gamma, exp_demo), afe_list, i))
                    if hyperdistance < min_hyperdistance:
                        min_hyperdistance = hyperdistance
                    print("current hyperdistance: ", hyperdistance)
                    print("min hyperdistance: ", min_hyperdistance)
                    # Step 3
                    if hyperdistance < e:
                        print("termination achieved")
                        exit()

                    t = 0
                    policies.clear()
                

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            is_solved = False
            if is_solved:
                # Show off the result
                print("Total Number of Episodes: ", len(episode_rewards))
                print("t final value: ", t)
                break
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 100 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("episode reward", episode_rewards[-2])
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
            
            t += 1
