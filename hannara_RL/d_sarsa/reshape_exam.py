import numpy as np
rewards = []


def set_reward(state, reward):
    state = [int(state[0]), int(state[1])]
    x = int(state[0])
    y = int(state[1])
    temp = {}
    if reward > 0:
        temp['reward'] = reward
        temp['figure'] = [1, 1]

    elif reward < 0:
        temp['direction'] = -1
        temp['reward'] = reward
        temp['figure'] = [2, 2]

    temp['coords'] = [2, 2]
    temp['state'] = state
    return rewards.append(temp)


set_reward([0, 1], -1)
set_reward([1, 2], -1)
set_reward([2, 3], -1)
set_reward([4, 4], 1)

# print(rewards)

location = [0, 0]
agent_x = location[0]
agent_y = location[1]

states = list()

for reward in rewards:  # 총4개 (장애물 3개 * 4, 목표 1개 * 3 = 15개)
    reward_location = reward['state']
    states.append(reward_location[0] - agent_x)
    states.append(reward_location[1] - agent_y)
    if reward['reward'] < 0:
        states.append(-1)
        states.append(reward['direction'])
    else:
        states.append(1)
print(states)
state = np.reshape(states, [1, 15]) #state_size = 15

print(state)