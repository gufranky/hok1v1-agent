#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np

from agent_ppo.feature.raw_state import ParsedState, distance


NO_ACTION = [0, 15, 15, 15, 15, 0]


class DebugAgent:
    def __init__(self):
        self.first_use_skill1 = False
        self.first_use_skill2 = False
        self.first_use_skill3 = False
        self.last_use_skill = False
        self.buff_dict = {"skills": set(), "marks": set()}

    def act(self, state: ParsedState, legal_actions=None, sub_action_mask=None):
        self.collect_buff_mark(state)
        dist = self.move_target(state.my_hero.unit.location, state.enemy_hero.unit.location)
        use_skill = False
        if dist < 1200 and state.frame_no > 200:
            if not self.first_use_skill1:
                use_skill = self.use_skill(0, legal_actions, sub_action_mask)
                if use_skill:
                    self.first_use_skill1 = True
            elif not self.first_use_skill2:
                use_skill = self.use_skill(1, legal_actions, sub_action_mask)
                if use_skill:
                    self.first_use_skill2 = True
            elif not self.first_use_skill3:
                use_skill = self.use_skill(2, legal_actions, sub_action_mask)
                if use_skill:
                    self.first_use_skill3 = True

        if dist < 1200 and not use_skill and not self.last_use_skill:
            action = self.normal_attack()
        else:
            action = self._move_action

        self.last_use_skill = use_skill
        return action

    def collect_buff_mark(self, state: ParsedState):
        for hero in [state.my_hero, state.enemy_hero]:
            self.buff_dict["skills"].update(hero.unit.buff.skill_ids)
            self.buff_dict["marks"].update(hero.unit.buff.mark_ids)

    def normal_attack(self):
        return [3, 0, 0, 0, 0, 1]

    def move_target(self, now, target):
        delta = self.delta_action_16x16(np.array(now, dtype=np.float32), np.array(target, dtype=np.float32))
        self._move_action = [2, int(delta[0]), int(delta[1]), 0, 0, 0]
        return distance(now, target)

    def use_skill(self, skill_index, legal_actions, sub_action_mask):
        if legal_actions is None or len(legal_actions) == 0:
            return False
        button = 4 + skill_index
        if len(legal_actions) <= button or legal_actions[button] == 0:
            return False
        mask = np.array(sub_action_mask[str(button)] if isinstance(sub_action_mask, dict) else sub_action_mask[button], dtype=np.int32)
        action = np.array([button, 0, 0, 8, 8, 1], dtype=np.int32)
        self._move_action = (action * mask).tolist()
        return True

    @staticmethod
    def delta_action_16x16(center, target):
        delta = target - center
        if np.max(np.abs(delta)) == 0:
            return np.array([8, 8], dtype=np.int32)
        return np.ceil(delta / np.max(np.abs(delta)) * 7).astype(np.int32) + np.array([8, 8], dtype=np.int32)
