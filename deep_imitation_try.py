import gym
import itertools
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
import random
import keyboard as kbd
from tensorflow.contrib import layers as layers
from pickle import dumps,loads

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


oldErrors = []
oldErrorStates = []

def build_act_imitation(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)
        return act

def build_train_imitation(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
    double_q=False, scope="deepq", reuse=None, param_noise=False, param_noise_filter_func=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array```
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    if param_noise:
        act_f = build_act_with_param_noise(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse,
            param_noise_filter_func=param_noise_filter_func)
    else:
        act_f = build_act_imitation(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1) # Q(s,a;θi)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best # maxQ(s',a';θi-)

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

# -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-! OBSERVER !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

        # TED's set up placeholders
        ment_obs_t_input = make_obs_ph("ment_obs_t")
        ment_act_t_ph = tf.placeholder(tf.int32, [None], name="ment_action")
        ment_obs_tp1_input = make_obs_ph("ment_obs_tp1")
        old_error_ph = tf.placeholder(tf.float32, shape=None, name="old_error")

        # TED's q network evaluation
        aug_q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        aug_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # TED's target q network evalution
        aug_q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func", reuse=True)
        aug_target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # TED's q scores for actions which we know were selected in the given state.
        aug_q_t_selected = tf.reduce_sum(aug_q_t * tf.one_hot(act_t_ph, num_actions), 1) # Q(s,a;θi)

        aug_q_tp1_selected = tf.reduce_sum(q_tp1 * tf.one_hot(ment_act_t_ph, num_actions), 1) # Q(s',am;θi)
        aug_q_tp1_selected_masked = (1.0 - done_mask_ph) * aug_q_tp1_selected

        # TED's compute estimate of best possible value starting from state at t + 1
        if double_q:
            aug_q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            aug_q_tp1_best_using_online_net = tf.argmax(aug_q_tp1_using_online_net, 1)
            aug_q_tp1_best = tf.reduce_sum(aug_q_tp1 * tf.one_hot(aug_q_tp1_best_using_online_net, num_actions), 1)
        else:
            aug_q_tp1_best = tf.reduce_max(aug_q_tp1, 1)
        aug_q_tp1_best_masked = (1.0 - done_mask_ph) * aug_q_tp1_best # maxQ(s',a';θi-)

        # TED's compute RHS of bellman equation
        aug_q_t_selected_target = rew_t_ph + gamma * tf.maximum(aug_q_tp1_best_masked, aug_q_tp1_selected_masked)
        # aug_q_t_selected_target = rew_t_ph + gamma * aug_q_tp1_best_masked

        # TED's compute the error (potentially clipped)
        aug_td_error = aug_q_t_selected - tf.stop_gradient(aug_q_t_selected_target)
        aug_errors = U.huber_loss(aug_td_error)
        aug_weighted_error = tf.reduce_mean(importance_weights_ph * aug_errors)
        # aug_weighted_error = tf.Print(aug_weighted_error, [aug_weighted_error], "AGENT WEIGHTED ERROR: ")

        # TED's compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(aug_weighted_error, var_list=aug_q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            aug_optimize_expr = optimizer.apply_gradients(gradients)
        else:
            aug_optimize_expr = optimizer.minimize(aug_weighted_error, var_list=aug_q_func_vars)

# -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-! OBSERVER !-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

# -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- MENTOR -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

        # TED's mentor's q network evaluation
        ment_q_t = q_func(ment_obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        ment_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # TED's mentor's target q network evalution
        ment_q_tp1 = q_func(ment_obs_tp1_input.get(), num_actions, scope="target_q_func", reuse=True)
        ment_target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # TED's mentor's q scores for action am which we know was selected in the given state. 
        ment_q_t_selected = tf.reduce_sum(ment_q_t * tf.one_hot(ment_act_t_ph, num_actions), 1) # Q(sm,am;θi)

        ment_q_tp1_selected = tf.reduce_sum(ment_q_tp1 * tf.one_hot(ment_act_t_ph, num_actions), 1) # Q(sm',am;θi-)
        ment_q_tp1_selected_masked = (1.0 - done_mask_ph) * ment_q_tp1_selected

        # TED's compute estimate of best possible value starting from state at t + 1
        if double_q:
            ment_q_tp1_using_online_net = q_func(ment_obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            ment_q_tp1_best_using_online_net = tf.argmax(ment_q_tp1_using_online_net, 1)
            ment_q_tp1_best = tf.reduce_sum(ment_q_tp1 * tf.one_hot(ment_q_tp1_best_using_online_net, num_actions), 1)
        else:
            ment_q_tp1_best = tf.reduce_max(ment_q_tp1, 1)
        ment_q_tp1_best_masked = (1.0 - done_mask_ph) * ment_q_tp1_best # maxQ(sm',a';θi-)

        # TED's compute RHS of bellman equation
        ment_q_t_selected_target = rew_t_ph + gamma * tf.maximum(ment_q_tp1_best_masked, ment_q_tp1_selected_masked)

        # TED's compute the error (potentially clipped)
        ment_td_error = ment_q_t_selected - tf.stop_gradient(ment_q_t_selected_target)
        ment_errors = U.huber_loss(ment_td_error)
        ment_weighted_error = tf.reduce_mean(importance_weights_ph * ment_errors)
        # ment_weighted_error = tf.Print(ment_weighted_error, [ment_weighted_error], "MENTOR WEIGHTED ERROR: ")

        # TED's compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(ment_weighted_error, var_list=ment_q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            ment_optimize_expr = optimizer.apply_gradients(gradients)
        else:
            ment_optimize_expr = optimizer.minimize(ment_weighted_error, var_list=ment_q_func_vars)

# -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- MENTOR -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!- 
        
        def temp_func1():
            return aug_td_error, aug_optimize_expr
        def temp_func2():
            return ment_td_error, ment_optimize_expr
        
        final_td_error, final_optimize_expr = tf.cond(tf.greater((ment_weighted_error-old_error_ph)**2, (aug_weighted_error-old_error_ph)**2), temp_func1, temp_func2)


        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        # TED's create callable functions
        trainAugmented = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph,
                ment_obs_t_input,
                ment_obs_tp1_input,
                ment_act_t_ph,
                old_error_ph
            ],
            outputs=final_td_error,
            updates=[final_optimize_expr]
        )

        return act_f, train, trainAugmented, update_target, {'q_values': q_values}


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def readMentorExperieces():
    mentor_tr = []
    
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
    # with open('mentor_demonstrations_NN_5.csv', 'r', newline='') as csvfile:
    #     data_reader = csv.reader(csvfile, delimiter=',')
    #     for r in data_reader:
    #         mentor_tr.append([float(i) for i in r])
    # with open('mentor_demonstrations_NN_6.csv', 'r', newline='') as csvfile:
    #     data_reader = csv.reader(csvfile, delimiter=',')
    #     for r in data_reader:
    #         mentor_tr.append([float(i) for i in r])
    # with open('mentor_demonstrations_NN_7.csv', 'r', newline='') as csvfile:
    #     data_reader = csv.reader(csvfile, delimiter=',')
    #     for r in data_reader:
    #         mentor_tr.append([float(i) for i in r])
    # with open('mentor_demonstrations_NN_8.csv', 'r', newline='') as csvfile:
    #     data_reader = csv.reader(csvfile, delimiter=',')
    #     for r in data_reader:
    #         mentor_tr.append([float(i) for i in r])
    # with open('mentor_demonstrations_NN_9.csv', 'r', newline='') as csvfile:
    #     data_reader = csv.reader(csvfile, delimiter=',')
    #     for r in data_reader:
    #         mentor_tr.append([float(i) for i in r])
    return mentor_tr

def similarityCheck(obs, ment, ment_act):
    # Check if the new state s' is similar to any state visited by the mentor.
    # The similarity threshold is set heuristicaly to 20% difference.
    sim_thres_perc = 20
    similarities = []

    for i in range(len(ment)-1):
        comparisons = []
        m_s = ment[i]

        for j in range(len(m_s)):
            diff_perc = ((obs[j] - m_s[j]) / m_s[j]) * 100
            comp = abs(diff_perc) < sim_thres_perc
            comparisons.append(comp)
        if (all(comparisons)):
            similarities.append(i)

    return similarities

def updateMentorActions(obs, new_obs, ment, ment_act, simList, action, env):
    sim_thres_perc = 20
    mentor_actions_index_list = []

    for i in simList:
        comparisons = []
        m_s = ment[i+1]
        if(ment_act[i] == None):
            for j in range(len(m_s)):
                diff_perc = ((new_obs[j] - m_s[j]) / m_s[j]) * 100
                comp = abs(diff_perc) < sim_thres_perc
                comparisons.append(comp)
            if (all(comparisons)):
                # print("Mentor action found")
                ment_act[i] = action
                mentor_actions_index_list.append(i)
            else:
                comparisons = []
                env.close()
                env_backup = dumps(env)
                for act in range(env.action_space.n):
                    if act != action:
                        tmp_new_obs, tmp_rew, tmp_done, _ = env.step(act)
                        for j in range(len(m_s)):
                            diff_perc = ((tmp_new_obs[j] - m_s[j]) / m_s[j]) * 100
                            comp = abs(diff_perc) < sim_thres_perc
                            comparisons.append(comp)
                        if (all(comparisons)):
                            print("Mentor action found through emulation")
                            ment_act[i] = act
                            mentor_actions_index_list.append(i)
                        env = loads(env_backup)

    return mentor_actions_index_list

def updateOldErrors():

    return


def augmentReward(rew, obs, ment, index):
    bias = 20
    newReward = rew + bias
    
    return newReward


if __name__ == '__main__':
    try:
        with U.make_session():
            # Create the environment
            env = gym.make("CartPole-v0")
            
            _gamma = 0.99
            oscillating = False
            mode = "0"


            print("Press 1 to enable imitation, 2 otherwise")
            while mode != "1" and mode != "2":
                mode = input("--> ")

            mentor_tr = readMentorExperieces()

            # Size of mentor's transitions
            N = len(mentor_tr)

            print("--Initializing mentor_actions buffer..")
            mentor_tr_actions = [None] * N

            # Create all the functions necessary to train the model
            act, train, trainAugmented, update_target, debug = build_train_imitation(
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
            is_solved = False

            y = [0]

            while True:
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = env.step(action)

                # Check if the new state s' is similar to any state visited by the mentor.
                if mode == "1":
                    similarity_list = similarityCheck(obs, mentor_tr, mentor_tr_actions)
                    similarity = len(similarity_list) != 0
                    if similarity and not is_solved:
                        # rew = augmentReward(rew, obs, mentor_tr, similarity_list)
                        ment_index_list = updateMentorActions(obs, new_obs, mentor_tr, mentor_tr_actions, similarity_list, action, env)
                        if len(ment_index_list) > 0:
                            ment_index = random.choice(ment_index_list)
                            ment_obs = mentor_tr[ment_index]
                            ment_obs_tp1 = mentor_tr[ment_index+1]
                            ment_act = mentor_tr_actions[ment_index]
                        else:
                            ment_index = random.choice(similarity_list)
                            ment_obs = mentor_tr[ment_index]
                            ment_obs_tp1 = mentor_tr[ment_index+1]
                            ment_act = random.randint(0, env.action_space.n)
                    else:
                        ment_index = random.randint(0, len(mentor_tr)-2)
                        ment_obs = mentor_tr[ment_index]
                        ment_obs_tp1 = mentor_tr[ment_index+1]
                        ment_act = random.randint(0, env.action_space.n)
                else:
                    ment_obs = []
                    ment_obs_tp1 = []
                    ment_act = 0

                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done), ment_obs, ment_obs_tp1, ment_act)

                obs = new_obs
                episode_rewards[-1] += rew

                # Detect oscilation or stalling of reward
                if len(episode_rewards) > 200 and abs(np.mean(episode_rewards[-101:-1]) - np.mean(episode_rewards[-201:-101])) <= 5:
                    oscillating = True
                else:
                    oscillating = False

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
                        obses_t, actions, rewards, obses_tp1, dones, ment_obs, ment_obs_tp1, ment_act = replay_buffer.sample(32)
                        if mode == "1":
                            trainAugmented(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards), ment_obs, ment_obs_tp1, ment_act, 0.5)
                        else:
                            train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))


                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                if done and len(episode_rewards) % 10 == 0:

                    if oscillating:
                        print("OSCILLATING")
                    
                    y.append(np.mean(episode_rewards[-101:-1]))
                    cnt = 0
                    for actn in mentor_tr_actions:
                        if actn != None:
                            cnt += 1
                    logger.record_tabular("mentor actions", cnt)
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean episode reward", np.mean(episode_rewards[-101:-1]))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

                    x = np.linspace(0, len(episode_rewards), num=len(y))
                    xmax = len(episode_rewards)
                    
                    plt.clf()
                    plt.plot(x, y, label='mean episode reward')
                    plt.axis([0, xmax, 0, 210])
                    plt.xticks(np.arange(0, len(episode_rewards), step=50), rotation=45)
                    plt.yticks(np.arange(0, 210, step=10))
                    plt.xlabel('time')
                    plt.ylabel('reward')
                    plt.title("deep_imitation: mean episode reward")
                    plt.grid(linestyle="-", linewidth="1")
                    plt.legend()
                    plt.pause(0.05)
                
                t += 1
    except KeyboardInterrupt:
        try:
            x = np.linspace(0, len(episode_rewards), num=len(y))
            xmax = len(episode_rewards)
            print(len(x))
            print(len(y))

            plt.clf()
            plt.plot(x, y, label='mean episode reward')
            plt.axis([0, xmax, 0, 210])
            plt.xticks(np.arange(0, len(episode_rewards), step=50), rotation=45)
            plt.yticks(np.arange(0, 210, step=10))
            plt.xlabel('time')
            plt.ylabel('reward')
            plt.title("deep_imitation: mean episode reward")
            plt.grid(linestyle="-", linewidth="1")
            plt.legend()
            plt.show()

            input("Press Enter to exit...")
        except Exception:
            print("\nAn Exception occured...")
            print("Exiting")
    exit()
    plt.show()
