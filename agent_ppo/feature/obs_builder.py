#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
from typing import List, Optional

import numpy as np

from agent_ppo.conf.conf import Args
from agent_ppo.feature.raw_state import ParsedState, Hero, Unit, Bullet, distance


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def sign_floor(value):
    return math.floor(value) if value > 0 else math.ceil(value)


class ObsBuilder:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.last_money = [None, None]
        self.next_cake_frame = [1778, 1778]

    def build_observation(self, state: ParsedState):
        self.state = state
        self.pos = state.my_hero.unit.location
        if self.last_money[0] is None:
            self.last_money = [state.my_hero.money_total, state.enemy_hero.money_total]

        hero_our = self.process_hero(state.my_hero, is_enemy=False, enemy_tower=state.enemy_tower)
        hero_enemy = self.process_hero(state.enemy_hero, is_enemy=True, enemy_tower=state.my_tower)
        soldier_our = self.process_soldiers(state.my_soldiers, state.enemy_tower)
        soldier_enemy = self.process_soldiers(state.enemy_soldiers, state.my_tower)
        tower_our = self.process_tower(state.my_tower, self.find_cake(state, state.player_camp), is_enemy=False)
        tower_enemy = self.process_tower(state.enemy_tower, self.find_cake(state, 1 - state.player_camp), is_enemy=True)
        bullets = self.process_bullets(state.bullets)

        feature = hero_our + hero_enemy + soldier_our + soldier_enemy + tower_our + tower_enemy + bullets
        assert len(feature) == Args.DIM_ALL, f"feature dim mismatch: {len(feature)} != {Args.DIM_ALL}"
        return np.array(feature, dtype=np.float32)

    def process_position(self, position):
        unit_size = Args.RELATIVE_DISTANCE_UNIT_SIZE
        max_size = Args.RELATIVE_DISTANCE_MAX_SIZE
        max_idx = max_size / unit_size + 1
        max_idx = max_idx / 2
        rel_x = int(clip(sign_floor((position[0] - self.pos[0]) / unit_size), -max_idx, max_idx) + max_idx)
        rel_z = int(clip(sign_floor((position[1] - self.pos[1]) / unit_size), -max_idx, max_idx) + max_idx)
        rel_dim = int(2 * max_idx + 1)
        rel_vec = [0.0] * (rel_dim * 2 + 1)
        rel_vec[rel_x] = 1.0
        rel_vec[rel_z + rel_dim] = 1.0
        rel_vec[-1] = clip(distance(position, self.pos) / (max_size / 2), 0.0, 1.0)

        unit_size = Args.WHOLE_DISTANCE_UNIT_SIZE
        max_size = Args.WHOLE_DISTANCE_MAX_SIZE
        whole_dim = int(max_size / unit_size)
        whole_x = int(clip(math.floor((position[0] + max_size / 2) / unit_size), 0, whole_dim - 1))
        whole_z = int(clip(math.floor((position[1] + max_size / 2) / unit_size), 0, whole_dim - 1))
        ratio_x = clip((position[0] + max_size / 2) / unit_size - whole_x, -1.0, 1.0)
        ratio_z = clip((position[1] + max_size / 2) / unit_size - whole_z, -1.0, 1.0)
        whole_vec = [0.0] * (whole_dim * 2 + 2)
        whole_vec[whole_x] = 1.0
        whole_vec[whole_z + whole_dim] = 1.0
        whole_vec[-2] = ratio_x
        whole_vec[-1] = ratio_z
        return rel_vec + whole_vec

    def process_unit(self, unit: Unit):
        hp_dim = Args.HP_MAX_SIZE // Args.HP_UNIT_SIZE + 2
        hp_vec = [0.0] * (1 + hp_dim)
        hp_vec[0] = unit.hp / max(unit.max_hp, 1)
        hp_index = 1 + min(int(math.ceil(unit.hp / Args.HP_UNIT_SIZE)), hp_dim - 1)
        hp_vec[hp_index] = 1.0

        mark_vec = [0.0] * Args.DIM_MARK
        if unit.buff.mark_ids:
            mark_vec[-1] = 1.0
        return hp_vec + mark_vec + self.process_position(unit.location)

    def process_skill(self, slot):
        cd_dim = Args.CD_MAX_SIZE // Args.CD_UNIT_SIZE + 2
        cd_vec = [0.0] * (2 + cd_dim)
        cd_max = max(slot.cooldown_max, 1)
        cd_vec[0] = slot.cooldown / cd_max
        cd_index = 1 + min(int(math.ceil(slot.cooldown / Args.CD_UNIT_SIZE)), cd_dim - 1)
        cd_vec[cd_index] = 1.0
        if not slot.usable and slot.level == 0:
            cd_vec[-1] = 1.0
        return cd_vec

    def process_money(self, money, is_enemy):
        idx = int(is_enemy)
        delta = money - self.last_money[idx]
        self.last_money[idx] = money
        money_dim = Args.MONEY_MAX_SIZE // Args.MONEY_UNIT_SIZE + 1
        money_vec = [0.0] * (2 + money_dim)
        if 0 < delta < 20:
            money_vec[-2] = 1.0
        else:
            bucket = max(min(int(math.floor(delta / Args.MONEY_UNIT_SIZE)), money_dim - 1), 0)
            money_vec[bucket] = 1.0
        money_vec[-1] = min(money / 10000, 1.0)
        return money_vec

    def process_hero(self, hero: Hero, is_enemy: bool, enemy_tower: Optional[Unit]):
        hero_id_vec = [float(Args.HERO_CONFIG_ID.index(hero.unit.config_id))] if hero.unit.config_id in Args.HERO_CONFIG_ID else [0.0]

        behave_vec = [0.0] * (len(Args.HERO_BEHAVE) + 1)
        behave_idx = len(Args.HERO_BEHAVE)
        if hero.unit.behave in Args.HERO_BEHAVE:
            behave_idx = Args.HERO_BEHAVE.index(hero.unit.behave)
        behave_vec[behave_idx] = 1.0

        ep_dim = Args.EP_MAX_SIZE // Args.EP_UNIT_SIZE + 1
        ep_vec = [0.0] * (1 + ep_dim)
        ep_vec[0] = hero.unit.ep / max(hero.unit.max_ep, 1)
        ep_idx = 1 + min(int(math.floor(hero.unit.ep / Args.EP_UNIT_SIZE)), ep_dim - 1)
        ep_vec[ep_idx] = 1.0

        slot_map = hero.skills
        skill_vec = []
        for slot_name in ["SLOT_SKILL_1", "SLOT_SKILL_2", "SLOT_SKILL_3", "SLOT_SKILL_5", "SLOT_SKILL_4"]:
            slot = slot_map.get(slot_name)
            if slot is None:
                class EmptySlot:
                    level = 0
                    usable = False
                    cooldown = 0
                    cooldown_max = 1
                slot = EmptySlot()
            skill_vec.extend(self.process_skill(slot))

        level_vec = [0.0] * Args.LEVEL_MAX
        level_vec[max(min(hero.level, Args.LEVEL_MAX), 1) - 1] = 1.0

        money_vec = self.process_money(hero.money_total, is_enemy)
        grass_vec = [float(hero.is_in_grass)]
        tower_vec = [0.0, 0.0]
        if enemy_tower is not None:
            tower_vec[0] = float(distance(hero.unit.location, enemy_tower.location) <= enemy_tower.attack_range)
            tower_vec[1] = float(hero.unit.runtime_id == enemy_tower.attack_target)

        buff_vec = [0.0] * Args.DIM_BUFF
        if hero.unit.buff.skill_ids:
            buff_vec[-1] = 1.0

        return hero_id_vec + behave_vec + ep_vec + skill_vec + level_vec + money_vec + grass_vec + tower_vec + buff_vec + self.process_unit(hero.unit)

    def process_soldiers(self, soldiers: List[Unit], enemy_tower: Optional[Unit]):
        soldiers = sorted(soldiers[: Args.SOLDIER_MAX_NUM], key=lambda unit: distance(unit.location, self.pos))
        feature = []
        for soldier in soldiers:
            soldier_vec = [0.0] * (Args.DIM_SOLDIER - Args.DIM_UNIT)

            behave_idx = len(Args.SOLDIER_BEHAVE)
            if soldier.behave in Args.SOLDIER_BEHAVE:
                behave_idx = Args.SOLDIER_BEHAVE.index(soldier.behave)
            soldier_vec[behave_idx] = 1.0

            base = len(Args.SOLDIER_BEHAVE) + 1
            for index, config_ids in enumerate(Args.SOLDIER_CONFIG_ID):
                if soldier.config_id in config_ids:
                    soldier_vec[base + index] = 1.0

            tower_base = base + len(Args.SOLDIER_CONFIG_ID)
            if enemy_tower is not None:
                soldier_vec[tower_base] = float(distance(soldier.location, enemy_tower.location) <= enemy_tower.attack_range)
                soldier_vec[tower_base + 1] = float(soldier.runtime_id == enemy_tower.attack_target)

            feature.extend(soldier_vec + self.process_unit(soldier))

        feature.extend([0.0] * (Args.DIM_SOLDIERS - len(feature)))
        return feature

    def process_tower(self, tower: Optional[Unit], cake, is_enemy: bool):
        if tower is None:
            return [0.0] * Args.DIM_ORGAN

        target_vec = [0.0] * 5
        target_type = self.state.id_to_unit_type.get(tower.attack_target)
        if tower.attack_target == 0:
            target_vec[0] = 1.0
        elif target_type == "hero":
            target_vec[1] = 1.0
        elif target_type == "soldier":
            target_vec[2] = 1.0

        side = int(is_enemy)
        target_vec[3] = float(cake is not None)
        if cake is not None:
            self.next_cake_frame[side] = self.state.frame_no + 76 * 30
            target_vec[4] = 0.0
        else:
            target_vec[4] = min((self.next_cake_frame[side] - self.state.frame_no) / (75 * 30), 1.0)

        return target_vec + self.process_unit(tower)

    def process_bullet(self, bullet: Bullet):
        slot_vec = [0.0] * (Args.DIM_BULLET - Args.DIM_DISTANCE)
        if bullet.slot_type in Args.BULLET_SLOT:
            slot_vec[Args.BULLET_SLOT.index(bullet.slot_type)] = 1.0
        else:
            slot_vec[-1] = 1.0
        return slot_vec + self.process_position(bullet.location)

    def process_bullets(self, bullets: List[Bullet]):
        enemy_bullets = [bullet for bullet in bullets if bullet.camp == 1 - self.state.player_camp]
        hero_bullets = [
            bullet for bullet in enemy_bullets
            if self.state.id_to_unit_type.get(bullet.source_actor) == "hero"
        ]
        organ_bullets = [
            bullet for bullet in enemy_bullets
            if self.state.id_to_unit_type.get(bullet.source_actor) == "organ"
        ]

        hero_bullets = sorted(hero_bullets, key=lambda bullet: distance(bullet.location, self.pos))[: Args.BULLET_MAX_NUM - 1]
        organ_bullets = sorted(organ_bullets, key=lambda bullet: distance(bullet.location, self.pos))[:1]

        feature = []
        for bullet in hero_bullets:
            feature.extend(self.process_bullet(bullet))
        while len(feature) < (Args.BULLET_MAX_NUM - 1) * Args.DIM_BULLET:
            feature.append(0.0)

        for bullet in organ_bullets:
            feature.extend(self.process_bullet(bullet))
        while len(feature) < Args.DIM_BULLETS:
            feature.append(0.0)
        return feature

    def find_cake(self, state: ParsedState, camp: int):
        for cake in state.cakes:
            if cake.camp == camp:
                return cake
        return None
