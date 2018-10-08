#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time, csv

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('CartPole-v0' if len(sys.argv)<2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

exp_demo = []
temp_list = []
start_exp_demp = False
end_exp_demo = False
N = 1000

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    # 0xff0d = ENTER
    if key==0xff0d: human_wants_restart = True
    # 32 = SPACE
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause, start_exp_demp, end_exp_demo
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        # if r != 0:
        #     print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()

        if (start_exp_demp == True):
            if len(exp_demo) < N:
                temp_list = list(obser)
                temp_list.append(done)
                temp_list.append(a)
                exp_demo.append(temp_list)
                print(len(exp_demo))
            else:
                end_exp_demo = True
                with open('expert_demonstrations_Human.csv', 'w', newline='') as csvfile:
                    data_writer = csv.writer(csvfile, delimiter=',')
                    for row in exp_demo:
                        data_writer.writerow(row)
                break

        if window_still_open==False: return False
        if done: break
        if human_wants_restart: 
            start_exp_demp = True
            break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    # print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open==False: break
    if end_exp_demo == True: 
        env.close()
        break
    
