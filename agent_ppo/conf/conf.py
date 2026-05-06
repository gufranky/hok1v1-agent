#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 10.0,
        "money": 4e-3,
        "exp": 4e-3,
        "death": -1.0,
        "last_hit": 0.5,
        "ep_rate": 0.75,
        "forward": 0.01,
        "kill": 0.0,
    }
    REMOVE_FORWARD_AFTER = 1000
    TIME_SCALE_ARG = 8000
    REWARD_WITHOUT_TIME_SCALE = set()
    MODEL_SAVE_INTERVAL = 1800
    CAMP_HEROES = [
        [112],
        [133],
    ]

    # DEBUG
    debug_agent = False
    debug_max_run_episodes = 1
    debug_frames = False
    debug_max_save_frame_no = 56 + 6 * 8000
    debug_total_frames = 4000


class Args:
    # Common unit features
    RELATIVE_DISTANCE_UNIT_SIZE = 600
    RELATIVE_DISTANCE_MAX_SIZE = 24600
    DIM_RELATIVE_DISTANCE = (RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2) * 2 + 1
    WHOLE_DISTANCE_UNIT_SIZE = 5000
    WHOLE_DISTANCE_MAX_SIZE = int(9e4)
    DIM_WHOLE_DISTANCE = (WHOLE_DISTANCE_MAX_SIZE // WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
    DIM_DISTANCE = DIM_RELATIVE_DISTANCE + DIM_WHOLE_DISTANCE

    HP_UNIT_SIZE = 100
    HP_MAX_SIZE = 2400
    MARK_ID_LAYERS = {}
    DIM_MARK = 1
    DIM_UNIT = DIM_DISTANCE + HP_MAX_SIZE // HP_UNIT_SIZE + 3 + DIM_MARK

    # Hero-specific features
    HERO_CONFIG_ID = [112, 133]
    HERO_BEHAVE = [
        "State_Dead",
        "State_Idle",
        "Direction_Move",
        "Normal_Attack",
        "State_Revive",
        "UseSkill_1",
        "UseSkill_2",
        "UseSkill_3",
    ]
    EP_UNIT_SIZE = 30
    EP_MAX_SIZE = 240
    CD_UNIT_SIZE = 1
    CD_MAX_SIZE = 10
    LEVEL_MAX = 15
    MONEY_UNIT_SIZE = 20
    MONEY_MAX_SIZE = 300
    BUFFS = []
    DIM_BUFF = len(BUFFS) + 1
    DIM_HERO = (
        DIM_UNIT
        + 1
        + len(HERO_BEHAVE)
        + 1
        + EP_MAX_SIZE // EP_UNIT_SIZE
        + 2
        + (CD_MAX_SIZE // CD_UNIT_SIZE + 4) * 5
        + LEVEL_MAX
        + MONEY_MAX_SIZE // MONEY_UNIT_SIZE
        + 3
        + 1
        + 2
        + DIM_BUFF
    )

    # Soldier-specific features
    SOLDIER_MAX_NUM = 4
    SOLDIER_BEHAVE = ["State_Dead", "Attack_Path"]
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]
    DIM_SOLDIER = DIM_UNIT + len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) + 2
    DIM_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM

    # Tower-specific features
    DIM_ORGAN = DIM_UNIT + 3 + 2
    DIM_ALL_UNITS = DIM_HERO * 2 + DIM_SOLDIERS * 2 + DIM_ORGAN * 2

    # Bullet features
    BULLET_MAX_NUM = 10
    BULLET_SLOT = ["SLOT_SKILL_0", "SLOT_SKILL_1", "SLOT_SKILL_2", "SLOT_SKILL_3", "SLOT_SKILL_VALID"]
    DIM_BULLET = len(BULLET_SLOT) + DIM_DISTANCE
    DIM_BULLETS = DIM_BULLET * BULLET_MAX_NUM
    DIM_ALL = DIM_ALL_UNITS + DIM_BULLETS


class DimConfig:
    DIM_OF_HERO_FRD = [Args.DIM_HERO]
    DIM_OF_HERO_EMY = [Args.DIM_HERO]
    DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * 4
    DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * 4
    DIM_OF_ORGAN_1 = [Args.DIM_ORGAN]
    DIM_OF_ORGAN_2 = [Args.DIM_ORGAN]
    DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * 9
    DIM_OF_BULLET_10 = [Args.DIM_BULLET]
    DIM_OF_FEATURE = [Args.DIM_ALL]


class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        Args.DIM_ALL + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(Args.DIM_ALL,), (85,)]
    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 5000
    BETA_START = 0.0
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(Args.DIM_ALL + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
