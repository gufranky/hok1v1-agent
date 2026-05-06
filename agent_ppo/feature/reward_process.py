#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math

from agent_ppo.conf.conf import GameConfig
from agent_ppo.feature.raw_state import build_parsed_state, distance, subtype_eq


class RewardStruct:
    def __init__(self, weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, observation_or_frame):
        self.init_max_exp_of_each_hero()
        if "frame_state" in observation_or_frame:
            observation = observation_or_frame
        else:
            raise ValueError("reward manager expects observation with frame_state")
        self.frame_data_process(observation)
        self.get_reward(observation, self.m_reward_value)
        return self.m_reward_value

    def set_cur_calc_frame_vec(self, calc_map, observation, camp):
        state = build_parsed_state(observation)
        main_hero = state.my_hero if state.my_hero.unit.camp == camp else state.enemy_hero
        enemy_hero = state.enemy_hero if state.my_hero.unit.camp == camp else state.my_hero
        main_tower = state.my_tower if state.my_hero.unit.camp == camp else state.enemy_tower
        enemy_tower = state.enemy_tower if state.my_hero.unit.camp == camp else state.my_tower

        for reward_name, reward_struct in calc_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero.money_total
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(main_hero.unit.hp / max(main_hero.unit.max_hp, 1)))
            elif reward_name == "ep_rate":
                if main_hero.unit.max_ep == 0 or main_hero.unit.hp <= 0:
                    reward_struct.cur_frame_value = 0.0
                else:
                    reward_struct.cur_frame_value = main_hero.unit.ep / float(max(main_hero.unit.max_ep, 1))
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero.kill_cnt
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero.dead_cnt
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 0.0 if main_tower is None else main_tower.hp / max(main_tower.max_hp, 1)
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                for dead in state.dead_events:
                    if dead.killer_runtime_id == main_hero.unit.runtime_id and subtype_eq(dead.death_sub_type, 0, "ACTOR_SUB_SOLDIER"):
                        reward_struct.cur_frame_value += 1.0
                    elif dead.killer_runtime_id == enemy_hero.unit.runtime_id and subtype_eq(dead.death_sub_type, 0, "ACTOR_SUB_SOLDIER"):
                        reward_struct.cur_frame_value -= 1.0
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero.level, main_hero.exp)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero.unit, main_tower, enemy_tower)

    def calculate_exp_sum(self, level, exp):
        exp_sum = 0.0
        for index in range(1, level):
            exp_sum += self.m_each_level_max_exp[index]
        exp_sum += exp
        return exp_sum

    def calculate_forward(self, hero, main_tower, enemy_tower):
        if main_tower is None or enemy_tower is None:
            return 0.0
        dist_hero2enemy = distance(hero.location, enemy_tower.location)
        dist_main2enemy = distance(main_tower.location, enemy_tower.location)
        if hero.hp / max(hero.max_hp, 1) > 0.99 and dist_hero2enemy > dist_main2enemy:
            return (dist_main2enemy - dist_hero2enemy) / max(dist_main2enemy, 1.0)
        return 0.0

    def frame_data_process(self, observation):
        state = build_parsed_state(observation)
        main_camp = state.my_hero.unit.camp
        enemy_camp = state.enemy_hero.unit.camp
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, observation, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, observation, enemy_camp)

    def get_reward(self, observation, reward_dict):
        reward_dict.clear()
        frame_no = observation["frame_state"].get("frame_no", observation["frame_state"].get("frameNo", 0))
        state = build_parsed_state(observation)
        reward_sum = 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value if reward_struct.last_frame_value > 0 else 0.0
            elif reward_name == "exp":
                if state.my_hero.level >= 15:
                    reward_struct.value = 0.0
                else:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                if GameConfig.REMOVE_FORWARD_AFTER is not None:
                    reward_struct.value *= float(frame_no <= GameConfig.REMOVE_FORWARD_AFTER)
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            time_scale = 1.0
            if self.time_scale_arg > 0 and reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                time_scale = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

            reward_dict[f"{reward_name}_origin"] = reward_struct.value
            reward_dict[f"{reward_name}_weight"] = reward_struct.value * reward_struct.weight * time_scale
            reward_sum += reward_dict[f"{reward_name}_weight"]

        reward_dict["reward_sum"] = reward_sum
