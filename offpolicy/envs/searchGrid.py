from typing import Optional
import gym
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
import random
import copy
# from gym.envs.classic_control import utils


class Drones(object):
    def __init__(self, pos, view_range, id, map_size):
        self.id = id
        self.pos = pos
        self.view_range = view_range
        self.area = None
        self.relative_pos = []
        self.relative_direction = []
        self.relatvie_coordinate = []
        self.individual_observed_zone = []
        self.observed_obs = []
        self.observed_drone = []
        self.individual_observed_obs = None
        self.unobserved = []
        self.communicate_rate = 0  # 添加了机器人通信频率奖励，使机器人在扩散探索的同时也注意信息的共享
        self.whole_map = np.zeros((4, map_size, map_size), dtype=np.float32)  # 每个机器人保存一个本地地图
        self.last_whole_map = None


class Human(object):
    def __init__(self, pos):
        self.pos = pos


class SearchGrid(gym.Env):
    def __init__(self):
        self.observation_space = [spaces.Box(low=0, high=1, shape=(4*50*50,)) for i in range (2)]
        self.share_observation_space = [spaces.Box(low=0, high=1, shape=(2*4*50*50,)) for i in  range(2)]
        self.action_space = [spaces.Discrete(4) for i in range(2)]
        print("adfadsfaefwefaewfewfaef")
        self.init_param()
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.drone_step(action)
        self.human_take_action()
        self.human_step(self.human_act_list)
        self.get_full_obs()
        self.get_joint_obs(self.MC_iter)
        observation, reward, done, info = self.state_action_reward_done()
        # if reward != -2:
        #     print(f"reward:{reward}, action:{action}")
        for i in range(len(observation)):
            observation[i]  = observation[i].flatten()
        return observation, reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None):
        self.init_param()
        self.get_full_obs()
        self.get_joint_obs(self.MC_iter)
        observation, _, _, info = self.state_action_reward_done()
        # print("observation",observation[0].shape)
        for i in range(len(observation)):
            observation[i]  = observation[i].flatten()
        return observation
        # return (observation[0])

    def render(self):
        pass

    def close(self):
        pass

    def drone_step(self, drone_act_list):
        # drone_act_list = [drone_act_list]
        for k in range(self.drone_num):
            if drone_act_list[k][0] == 1:
                self.drone_list[k].pos[0] = self.drone_list[k].pos[0] - 1
            elif drone_act_list[k][1] == 1:
                self.drone_list[k].pos[0] = self.drone_list[k].pos[0] + 1
            elif drone_act_list[k][2] == 1:
                self.drone_list[k].pos[1] = self.drone_list[k].pos[1] - 1
            elif drone_act_list[k][3] == 1:
                self.drone_list[k].pos[1] = self.drone_list[k].pos[1] + 1
            # print(self.drone_list[k].pos[0], self.drone_list[k].pos[1])

    def human_take_action(self):
        self.human_act_list = [0] * self.human_num
        for i in range(self.human_num):
            self.human_act_list[i] = random.randint(0, 3)

    def human_step(self, human_act_list):
        for k in range(self.human_num):
            # print(self.human_init_pos)
            # print([self.human_list[k].pos[0]-self.human_init_pos[k][0], self.human_list[k].pos[1]-self.human_init_pos[k][1]])
            if human_act_list[k] == 0:
                if self.human_list[k].pos[0] > 0 and (self.human_list[k].pos[0] - \
                                                      self.human_init_pos[k][0] - 1 > -self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0] - 1, self.human_list[k].pos[1]]
                    if free_space == 0:
                        self.human_list[k].pos[0] = self.human_list[k].pos[0] - 1
            elif human_act_list[k] == 1:
                if self.human_list[k].pos[0] < self.map_size - 1 and (self.human_list[k].pos[0] - \
                                                                      self.human_init_pos[k][
                                                                          0] + 1 < self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0] + 1, self.human_list[k].pos[1]]
                    if free_space == 0:
                        self.human_list[k].pos[0] = self.human_list[k].pos[0] + 1
            elif human_act_list[k] == 2:
                if self.human_list[k].pos[1] > 0 and (self.human_list[k].pos[1] - \
                                                      self.human_init_pos[k][1] - 1 > -self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] - 1]
                    if free_space == 0:
                        self.human_list[k].pos[1] = self.human_list[k].pos[1] - 1
            elif human_act_list[k] == 3:
                if self.human_list[k].pos[1] < self.map_size - 1 and (self.human_list[k].pos[1] - \
                                                                      self.human_init_pos[k][
                                                                          1] + 1 < self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] + 1]
                    if free_space == 0:
                        self.human_list[k].pos[1] = self.human_list[k].pos[1] + 1

    def get_full_obs(self):  # 这里是整个环境的信息
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.land_mark_map[i, j] == 1:  # [0,0,0]表示wall
                    obs[i, j, 0] = 0
                    obs[i, j, 1] = 0
                    obs[i, j, 2] = 0
                if self.land_mark_map[i, j] == 2:  # [0,1,0]表示tree
                    obs[i, j, 0] = 0
                    obs[i, j, 1] = 0
                    obs[i, j, 2] = 0

        for i in range(self.human_num):  # [1,0,0]表示human
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 0] = 1
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 1] = 0
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 2] = 0

        for i in range(self.drone_num):
            obs[self.drone_list[i].pos[0], self.drone_list[i].pos[1], 0] = 0.5
            obs[self.drone_list[i].pos[0], self.drone_list[i].pos[1], 1] = 0
            obs[self.drone_list[i].pos[0], self.drone_list[i].pos[1], 2] = 0.5
        return obs

    def get_drone_obs(self, drone):  # 获得无人机的观测，这里的drone是类
        drone.observed_obs = []
        drone.unobserved = []
        drone.individual_observed_obs = 0
        drone.observed_drone = []
        drone.communicate_rate = 0
        drone.last_whole_map = drone.whole_map.copy()
        index = random.randint(self.sensing_threshold[0], self.sensing_threshold[1])
        obs_size = 2 * drone.view_range - 1
        sensing_size = 2 * (drone.view_range + index) - 1
        obs = np.ones((obs_size, obs_size, 3))
        # 这里是给机器人感知其他机器人的位置加了波动

        drone.whole_map[0, drone.pos[0], drone.pos[1]] = self.memory_step  # 记录100步的信息
        for i in range(sensing_size):
            for j in range(sensing_size):
                x = i + drone.pos[0] - (drone.view_range + index) + 1
                y = j + drone.pos[1] - (drone.view_range + index) + 1
                for k in range(self.drone_num):  # 是否有其他机器人在观测范围内
                    if self.drone_list[k].pos[0] == x and self.drone_list[k].pos[1] == y:
                        if self.drone_list[k].id != drone.id:
                            drone.observed_drone.append([x, y])
                            drone.whole_map[
                                2, x, y] = self.memory_step  # add other agent's history positions to the map
                            drone.communicate_rate += 1

        # 这里循环的目的是构建障碍物地图
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1
                if 0 <= x <= self.map_size - 1 and 0 <= y <= self.map_size - 1:
                    if self.land_mark_map[x, y] == 2:
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0

        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1
                # if 0 <= x < 50 and 0 <= y < 50:
                #     drone.whole_map[1, x, y] = self.MC_iter  # Add cell's timestamp to an agent's whole map.
                for k in range(self.human_num):  # 是否有目标点在观测范围内
                    if self.human_list[k].pos[0] == x and self.human_list[k].pos[1] == y:
                        obs[i, j, 0] = 1
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0
                for k in range(self.drone_num):  # 是否有其他机器人在观测范围内
                    if self.drone_list[k].pos[0] == x and self.drone_list[k].pos[1] == y:
                        obs[i, j, 0] = 0.5
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0.5
                if 0 <= x <= self.map_size - 1 and 0 <= y <= self.map_size - 1:  # 是否有障碍物在观测范围内
                    if self.land_mark_map[x, y] == 1:
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0
                    if self.land_mark_map[x, y] == 2:  # 在发现障碍物后对观测进行处理
                        obs[i, j, 0] = 0
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0
                        drone.observed_obs.append([x, y])
                        gap = [drone.observed_obs[-1][0] - drone.pos[0], \
                               drone.observed_obs[-1][1] - drone.pos[1]]
                        gap_abs = [abs(drone.observed_obs[-1][0] - drone.pos[0]), \
                                   abs(drone.observed_obs[-1][1] - drone.pos[1])]
                        chosen_gap = max(gap_abs)
                        if chosen_gap < drone.view_range:
                            if gap[0] >= 0 and gap[1] > 0:
                                if gap[0] == 0:
                                    if obs[i + 1, j, 0] == 0 and obs[i + 1, j, 1] == 0 and \
                                            obs[i + 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num_1, j + num + 1])
                                    if obs[i - 1, j, 0] == 0 and obs[i - 1, j, 1] == 0 and \
                                            obs[i - 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num_1, j + num + 1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + num + 1, j + num + 1])
                            if gap[0] > 0 and gap[1] <= 0:
                                if gap[1] == 0:
                                    if obs[i, j + 1, 0] == 0 and obs[i, j + 1, 1] == 0 and \
                                            obs[i, j + 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num + 1, j + num_1])
                                    if obs[i, j - 1, 0] == 0 and obs[i, j - 1, 1] == 0 and \
                                            obs[i, j - 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num + 1, j - num_1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + num + 1, j - num - 1])
                            if gap[0] < 0 and gap[1] >= 0:
                                if gap[1] == 0:
                                    if obs[i, j + 1, 0] == 0 and obs[i, j + 1, 1] == 0 and \
                                            obs[i, j + 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num - 1, j + num_1])
                                    if obs[i, j - 1, 0] == 0 and obs[i, j - 1, 1] == 0 and \
                                            obs[i, j - 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num - 1, j - num_1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i - num - 1, j + num + 1])
                            if gap[0] <= 0 and gap[1] < 0:
                                if gap[0] == 0:
                                    if obs[i + 1, j, 0] == 0 and obs[i + 1, j, 1] == 0 and \
                                            obs[i + 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num_1, j - num - 1])
                                    if obs[i - 1, j, 0] == 0 and obs[i - 1, j, 1] == 0 and \
                                            obs[i - 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num_1, j - num - 1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i - num - 1, j - num - 1])

                else:  # 其他情况
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0.5
                    obs[i, j, 2] = 0.5

                # 这里是设置圆形观测区域
                if (drone.view_range - 1 - i) * (drone.view_range - 1 - i) + (drone.view_range - 1 - j) * (
                        drone.view_range - 1 - j) > drone.view_range * drone.view_range:
                    obs[i, j, 0] = 0.5
                    obs[i, j, 1] = 0.5
                    obs[i, j, 2] = 0.5

        for pos in drone.unobserved:  # 这里处理后得到的obs是能观测到的标志物地图
            obs[pos[0], pos[1], 0] = 0.5
            obs[pos[0], pos[1], 1] = 0.5
            obs[pos[0], pos[1], 2] = 0.5

        # 这里计算与其他机器人在时刻t的相对位置
        # temp_list = []
        # for i in range(self.drone_num):
        #     temp = ((self.drone_list[i].pos[0] - drone.pos[0]) ** 2 + \
        #             (self.drone_list[i].pos[1] - drone.pos[1]) ** 2) ** 0.5
        #     if i != drone.id:
        #         temp_list.append(temp)
        # drone.relative_pos = temp_list
        # # print("relative_pos:",drone.relative_pos)
        # # 这里计算与其他机器人在时刻t的相对方向
        # temp_list = []
        # for i in range(self.drone_num):
        #     temp = self.get_relative_direction(drone.pos[0], drone.pos[1], \
        #                                        self.drone_list[i].pos[0], self.drone_list[i].pos[1])
        #     if i != drone.id:
        #         temp_list.append(temp)
        # drone.relative_direction = temp_list

        drone.whole_map[0] = np.zeros(drone.whole_map[0].shape, dtype=np.float32)
        drone.whole_map[0, drone.pos[0], drone.pos[1]] = 1
        for i in range(self.map_size):  # 这里进行轨迹的衰减
            for j in range(self.map_size):
                if drone.whole_map[2, i, j] > 1 / self.t_u:
                    drone.whole_map[2, i, j] = drone.whole_map[2, i, j] - 1 / self.t_u
        # print("relative_direction:",drone.relative_direction)

        for i in range(sensing_size):  # 在观测范围内进行信息更新，更新时间戳地图1，轨迹地图2和障碍物地图3
            for j in range(sensing_size):
                x = i + drone.pos[0] - (drone.view_range + index) + 1
                y = j + drone.pos[1] - (drone.view_range + index) + 1
                for k in range(self.drone_num):
                    if self.drone_list[k].pos[0] == x and self.drone_list[k].pos[1] == y:
                        if self.drone_list[k].id != drone.id:
                            drone.whole_map[2, x, y] = max(self.drone_list[k].whole_map[2, x, y],
                                                           drone.whole_map[2, x, y])
                            drone.whole_map[1, x, y] = max(self.drone_list[k].whole_map[1, x, y],
                                                           drone.whole_map[1, x, y])
                            drone.whole_map[3, x, y] = max(self.drone_list[k].whole_map[1, x, y],
                                                           drone.whole_map[3, x, y])
        return obs

    def get_joint_obs(self, time_stamp):
        # Modification record which agent find the target
        # One target can be found by multi agents at the smae time point.
        # Target_per_agent defines how many targets are found by each agent at thie time point.
        self.target_per_agent = np.zeros(self.drone_num)
        human_del_list = []
        len_human_del_list = 0
        # Record obstacle gain of each agent
        self.obstacle_gain_per_agent = np.zeros(self.drone_num)
        len_obstacle_gain = 0

        self.obstacles_temp = copy.deepcopy(self.obstacles)
        self.per_observed_goal_num = 0
        self.time_stamp = time_stamp
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                obs[i, j, 0] = 0.5
                obs[i, j, 1] = 0.5
                obs[i, j, 2] = 0.5

        for k in range(self.drone_num):
            self.drone_list[k].individual_observed_obs = 0
            temp = self.get_drone_obs(self.drone_list[k])
            size = temp.shape[0]
            temp_list_individual = []

            for i in range(size):
                for j in range(size):
                    x = i + self.drone_list[k].pos[0] - self.drone_list[k].view_range + 1
                    y = j + self.drone_list[k].pos[1] - self.drone_list[k].view_range + 1
                    if_obs = True
                    # 如果一个位置根本没有被观测到，就不执行赋值
                    if temp[i, j, 0] == 0.5 and temp[i, j, 1] == 0.5 and temp[i, j, 2] == 0.5:
                        if_obs = False
                    if if_obs == True:
                        obs[x, y, 0] = temp[i, j, 0]
                        obs[x, y, 1] = temp[i, j, 1]
                        obs[x, y, 2] = temp[i, j, 2]
                        temp_list_individual.append([x, y])
                        # 这里为了判断观测中有多少障碍物，并更新障碍物地图
                        if temp[i, j, 0] == 0 and temp[i, j, 1] == 0 and temp[i, j, 2] == 0:
                            self.drone_list[k].individual_observed_obs += 1
                            self.obstacles.append([x, y])  # 所有机器人观测过的障碍物
                            self.drone_list[k].whole_map[
                                3, x, y] = 1  # add obstacle information to each agent's whole map
                        # 如果观测中有目标，则清除被观测到的目标
                        if obs[x, y, 0] == 1 and obs[x, y, 1] == 0 and obs[x, y, 2] == 0:
                            self.per_observed_goal_num += 1
                            for num, goal in enumerate(self.human_list):
                                if goal.pos[0] == x and goal.pos[1] == y:
                                    human_del_list.append(num)

            self.obstacle_gain_per_agent[k] = len(self.obstacles) - len_obstacle_gain
            len_obstacle_gain = len(self.obstacles)
            self.target_per_agent[k] = len(human_del_list)-len_human_del_list
            len_human_del_list = len(human_del_list)
            self.drone_list[k].individual_observed_zone = temp_list_individual
            # 这里计算观测区域去掉障碍物的面积
            self.drone_list[k].area = len(self.drone_list[k].individual_observed_zone) - \
                                      self.drone_list[k].individual_observed_obs
            # print(len(self.drone_list[k].individual_observed_zone))

        # Delete all targets found at this time point
        # 去掉重复检测到的target
        human_del_list = list(set(human_del_list))
        if len(human_del_list)>0:
            new_human_list = []
            new_human_init_pos = []
            for i in range(len(self.human_list)):
                if i in human_del_list:
                    self.human_num -= 1
                else:
                    new_human_list.append(self.human_list[i])
                    new_human_init_pos.append(self.human_init_pos[i])
            self.human_list = new_human_list
            self.human_init_pos = new_human_init_pos

        # print("处理前：", self.obstacles)
        # for pos in self.obstacles:   #这种方法可能指针会跳跃，导致输出的结果不一定正确
        #     while self.obstacles.count(pos) > 1:
        #         self.obstacles.remove(pos)
        temp_list_copy = copy.deepcopy(self.obstacles)
        for pos in temp_list_copy:
            while self.obstacles.count(pos) > 1:
                self.obstacles.remove(pos)
        for k in range(self.drone_num):  # 这里计算与观测区域内所有机器人的相对距离
            temp_list = []
            temp = [0, 0]
            for pos in self.drone_list[k].observed_drone:
                temp_list.append([abs(self.drone_list[k].pos[0] - pos[0]), \
                                  abs(self.drone_list[k].pos[1] - pos[1])])
            for pos in temp_list:
                temp[0] += pos[0] / len(temp_list)
                temp[1] += pos[1] / len(temp_list)
            self.drone_list[k].relative_coordinate = temp

        sensing_size = 2 * self.drone_list[0].view_range - 1
        index = random.randint(self.sensing_threshold[0], self.sensing_threshold[1])
        for k in range(self.drone_num):  # 更新观测范围内的时间戳地图
            for i in range(sensing_size):
                for j in range(sensing_size):

                    x = i + self.drone_list[k].pos[0] - (self.drone_list[k].view_range + index) + 1
                    y = j + self.drone_list[k].pos[1] - (self.drone_list[k].view_range + index) + 1
                    if 0 <= x < 50 and 0 <= y < 50:
                        self.drone_list[k].whole_map[1, x, y] = self.MC_iter

        return obs

    def state_action_reward_done(self):  # 这里返回状态值，奖励值，以及游戏是否结束
        reward = 0  # 合作任务，只设置单一奖励
        reward_list = np.zeros(self.drone_num,dtype=np.float32)
        ####################设置奖励的增益
        target_factor = 10
        information_gain = 2
        distance_factor = 0.001
        ####################
        # for i in range(self.drone_num):   #这里可以做智能信用分配
        #     reward += self.compute_reward(self.drone_list[i])
        done = False
        single_map_set = [self.drone_list[k].whole_map for k in range(self.drone_num)]
        if self.MC_iter > 0:
            # 这里对状态进行最值归一化，只需要操作观测历史层
            for single_map in single_map_set:
                max_value = np.max(single_map[1])
                single_map[1] = single_map[1] / max_value
        info = {}
        reward_list = [target_factor*i_agent for i_agent in self.target_per_agent] # 这里计算发现目标点的数量
        if self.human_num == 0:
            reward_list = list(map(lambda x:x+500, reward_list))
            done = True
            # info['0'] = "find all target"
        for i in range(self.drone_num - 1):  # 如果机器人发生碰撞
            for j in range(i + 1, self.drone_num):
                if self.drone_list[i].pos[0] == self.drone_list[j].pos[0] \
                        and self.drone_list[i].pos[1] == self.drone_list[j].pos[1]:
                    done = True
                    reward_list[i] -= 200
                    reward_list[j] -= 200
                    # info['0'] = "robot collision"
        for i in range(self.drone_num):  # 如果机器人和障碍物发生碰撞
            for j in range(len(self.obstacles)):
                if self.drone_list[i].pos[0] == self.obstacles[j][0] and \
                        self.drone_list[i].pos[1] == self.obstacles[j][1]:
                    done = True
                    reward_list[i] -= 200
                    # info['0'] = "obs collision"
        if self.time_stamp > self.run_time:  # 超时
            done = True
            # info['0'] = "exceed run time"
        if len(self.obstacles) == self.global_obs_num:
            done = True
            reward_list = list(map(lambda x:x+500, reward_list))
            # info['0'] = "construct the feature map"
        reward_list = list(map(lambda x: x -0.4, reward_list)) # 单步惩罚
        reward_list = [reward_list[i_agent] + information_gain * self.obstacle_gain_per_agent[i_agent] for i_agent in range(self.drone_num)]
        dis = 0
        # for i in range(self.drone_num):
        #     dis += ((self.drone_list[i].pos[0] - self.drone_list[0].pos[0]) ** 2 + \
        #             (self.drone_list[i].pos[1] - self.drone_list[0].pos[1]) ** 2) ** 0.5
        # reward += distance_factor * dis
        done_list  = [done for i_agent in range(self.drone_num)]
        # print("np.array reward lis",np.array(reward_list).reshape(self.drone_num,1))
        # print("-------reward list",reward_list)
        return single_map_set, reward_list, done_list, [info for i in range(self.drone_num)]

    def compute_reward(self, drone):  # s->a->r->s'
        pos_factor = 0.2
        direction_factor = 0.01
        target_factor = 300
        communicate_factor = 10
        time_factor = 1
        information_gain = 5 / 3

        # sum_1 = 0
        reward = 0
        # for i in drone.relative_pos:  # 这里计算相对位置的奖励
        #     sum_1 += i
        # sum_1 = pos_factor * sum_1
        # reward += sum_1 / 2 / self.drone_num
        # sum_2 = 0
        # sorted_relative_direction = copy.deepcopy(drone.relative_direction)
        # sorted_relative_direction.sort()
        # # print("sorted: ", sorted_relative_direction)
        # # print("original: ", drone.relative_direction)
        # for i in range(len(sorted_relative_direction) - 1):  # 这里计算相对方向的奖励
        #     sum_2 += sorted_relative_direction[i + 1] - sorted_relative_direction[i]
        # reward += direction_factor * sum_2 / self.drone_num

        # 这里增加通信频次更新信息的奖励
        # sum_4 = drone.communicate_rate / 2 / self.drone_num * communicate_factor
        # reward += sum_4

        return reward

    def init_param(self):
        self.MC_iter = 0
        self.run_time = 2000  # Run run_time steps per game
        self.map_size = 50
        self.drone_num = 2
        self.num_agents = self.drone_num # This is used in --n_rollout_threads, the number of parallel envs for training.
        self.view_range = 10
        self.tree_num = 3
        self.human_init_pos = []
        self.human_num = 5
        self.human_num_temp = 2
        self.human_num_copy = 2
        self.sensing_threshold = [3, 5]
        self.time_stamp = None
        self.observed_zone = {}  # 带有时序的已观测点
        self.global_reward = []
        self.global_done = []
        self.per_observed_goal_num = None
        self.obstacles = []  # 记录所有机器人观测到的障碍物
        self.obstacles_temp = []
        self.human_act_list = []
        self.drone_act_list = []
        # initialize trees
        self.land_mark_map = np.zeros((self.map_size, self.map_size))  # 地标地图
        self.memory_step = 1
        self.global_obs_num = 0
        self.t_u = 50
        self.move_threshold = 2
        self.random_pos_robot = False
        self.random_pos_target = True
        # intialize tree(随机生成块状障碍物)
        # for i in range(self.tree_num):
        #     tree_pos = []
        #     temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
        #     while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
        #         temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
        #     x, y = temp_pos
        #     if 1 < x < self.map_size - 2 and 1 < y < self.map_size - 2:
        #         tree_pos = [[x-1,y-1], [x-1,y], [x-1,y+1], [x,y-1], [x,y],\
        #         [x,y+1], [x+1,y-1], [x+1,y], [x+1,y+1]]
        #         # self.land_mark_map[temp_pos[0], temp_pos[1]] = 2  # tree
        #     for tree in tree_pos:
        #         self.land_mark_map[tree[0], tree[1]] = 2

        # 固定形状障碍物
        inverse_wall = [[16, 14], [16, 17], [13, 17], [10, 17], [10, 14],
                        [34, 36], [34, 33], [37, 33], [40, 33], [40, 36],
                        [16, 36], [16, 33], [13, 33], [10, 33], [10, 36],
                        [34, 14], [34, 17], [37, 17], [40, 17], [40, 14]]

        four_long_wall = [[14, 14], [14, 17], [36, 36], [36, 33], [14, 36], [14, 33],
                          [36, 14], [36, 17]]
        # for pos in inverse_wall:
        for pos in four_long_wall:
            tree_pos = []
            temp_pos = pos
            x, y = temp_pos
            if 1 < x < self.map_size - 2 and 1 < y < self.map_size - 2:
                tree_pos = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y], \
                            [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
                # self.land_mark_map[temp_pos[0], temp_pos[1]] = 2  # tree
            for tree in tree_pos:
                self.land_mark_map[tree[0], tree[1]] = 2
        # 随机生成墙体
        # for i in range(self.tree_num):
        #     tree_pos = []
        #     temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        #     while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
        #         temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        #     temp_pos = [[temp_pos[0], temp_pos[1] - 3], [temp_pos[0], temp_pos[1]], [temp_pos[0], temp_pos[1] + 3], \
        #                 [temp_pos[0] + 3, temp_pos[1] - 6], [temp_pos[0] + 6, temp_pos[1] - 3],
        #                 [temp_pos[0] + 6, temp_pos[1]], \
        #                 [temp_pos[0] + 6, temp_pos[1] + 3], [temp_pos[0] + 3, temp_pos[1] + 6], \
        #                 [temp_pos[0], temp_pos[1] - 6], [temp_pos[0], temp_pos[1] + 6], \
        #                 [temp_pos[0] + 6, temp_pos[1] - 6], [temp_pos[0] + 6, temp_pos[1] + 6]]
        #     del temp_pos[random.randint(0, len(temp_pos) - 1)]
        #     del temp_pos[random.randint(0, len(temp_pos) - 1)]
        #     for j in temp_pos:
        #         x = j[0]
        #         y = j[1]
        #         if 1 < x < self.map_size - 2 and 1 < y < self.map_size - 2:
        #             tree_pos = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y], \
        #                         [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
        #             # self.land_mark_map[temp_pos[0], temp_pos[1]] = 2  # tree
        #         for tree in tree_pos:
        #             self.land_mark_map[tree[0], tree[1]] = 2
        # 边缘加上围墙
        for i in range(self.map_size):
            self.land_mark_map[i, 0] = 2
            self.land_mark_map[0, i] = 2
            self.land_mark_map[self.map_size - 1, i] = 2
            self.land_mark_map[i, self.map_size - 1] = 2

        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.land_mark_map[i, j] == 2:
                    self.global_obs_num += 1

        # randomly initialize humans
        if self.random_pos_target:
            self.human_list = []
            for i in range(self.human_num):
                temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
                    temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                self.human_init_pos.append(temp_pos.copy())
                temp_human = Human(temp_pos)
                self.human_list.append(temp_human)
        # fixedly initialize humans
        else:
            self.human_list = []
            # temp_pos = [[16, 14], [34, 36], [16, 46], [40, 37], [48, 3]]
            temp_pos = [[12, 13], [38, 38], [16, 46], [35, 10], [48, 3]]
            for i in range(self.human_num):
                temp_human = Human(temp_pos[i])
                self.human_init_pos.append(temp_pos[i].copy())
                self.human_list.append(temp_human)
        # randomly initialize drones
        # self.start_pos = [self.map_size-1, self.map_size-1]
        if self.random_pos_robot:
            self.drone_list = []
            for i in range(self.drone_num):
                temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
                    temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                temp_drone = Drones(temp_pos, self.view_range, i, self.map_size)
                self.drone_list.append(temp_drone)
        # fixedly initialize robot
        else:
            self.drone_list = []
            temp_pos = [[25, 25], [25, 27], [25, 23], [23, 25], [27, 25]]
            for i in range(self.drone_num):
                temp_drone = Drones(temp_pos[i], self.view_range, i, self.map_size)
                self.drone_list.append(temp_drone)