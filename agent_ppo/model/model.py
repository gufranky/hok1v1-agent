#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict
from typing import List

from agent_ppo.conf.conf import Args, Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.hero_dim = Args.DIM_HERO
        self.soldiers_dim = Args.DIM_SOLDIERS
        self.organ_dim = Args.DIM_ORGAN
        self.bullet_dim = Args.DIM_BULLETS

        self.hero_encoder = MLP([self.hero_dim, 256, 128], "hero_encoder", non_linearity_last=True)
        self.soldier_encoder = MLP([self.soldiers_dim, 256, 128], "soldier_encoder", non_linearity_last=True)
        self.organ_encoder = MLP([self.organ_dim, 128, 64], "organ_encoder", non_linearity_last=True)
        self.bullet_encoder = MLP([self.bullet_dim, 256, 128], "bullet_encoder", non_linearity_last=True)

        concat_dim = 128 + 128 + 128 + 128 + 64 + 64
        self.concat_mlp = MLP([concat_dim, 512, 512], "concat_mlp", non_linearity_last=True)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.lstm_and_public_mlp = MLP([512 + self.lstm_unit_size, 512, 512], "lstm_and_public", non_linearity_last=True)

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [512, 256, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(512, self.target_embed_dim)
        self.target_embed_mlp = make_fc_layer(32, self.target_embed_dim, use_bias=False)
        self.value_mlp = MLP([512, 256, 1], "hero_value_mlp")

    def _split_feature(self, feature_vec):
        idx = 0
        hero_our = feature_vec[:, idx : idx + self.hero_dim]
        idx += self.hero_dim
        hero_enemy = feature_vec[:, idx : idx + self.hero_dim]
        idx += self.hero_dim
        soldier_our = feature_vec[:, idx : idx + self.soldiers_dim]
        idx += self.soldiers_dim
        soldier_enemy = feature_vec[:, idx : idx + self.soldiers_dim]
        idx += self.soldiers_dim
        organ_our = feature_vec[:, idx : idx + self.organ_dim]
        idx += self.organ_dim
        organ_enemy = feature_vec[:, idx : idx + self.organ_dim]
        idx += self.organ_dim
        bullets = feature_vec[:, idx : idx + self.bullet_dim]
        return hero_our, hero_enemy, soldier_our, soldier_enemy, organ_our, organ_enemy, bullets

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        split = self._split_feature(feature_vec)
        hero_our = self.hero_encoder(split[0])
        hero_enemy = self.hero_encoder(split[1])
        soldier_our = self.soldier_encoder(split[2])
        soldier_enemy = self.soldier_encoder(split[3])
        organ_our = self.organ_encoder(split[4])
        organ_enemy = self.organ_encoder(split[5])
        bullets = self.bullet_encoder(split[6])

        public_feature = torch.cat([hero_our, hero_enemy, soldier_our, soldier_enemy, organ_our, organ_enemy, bullets], dim=1)
        public_feature = self.concat_mlp(public_feature)

        self.lstm_hidden_output = lstm_hidden_init.unsqueeze(0)
        self.lstm_cell_output = lstm_cell_init.unsqueeze(0)
        lstm_input = public_feature.unsqueeze(1)
        lstm_output, (lstm_hidden_output, lstm_cell_output) = self.lstm(
            lstm_input,
            (self.lstm_hidden_output, self.lstm_cell_output),
        )
        self.lstm_hidden_output = lstm_hidden_output
        self.lstm_cell_output = lstm_cell_output

        merged_feature = torch.cat([public_feature, lstm_output.squeeze(1)], dim=1)
        merged_feature = self.lstm_and_public_mlp(merged_feature)

        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            result_list.append(self.label_mlp[f"hero_label{label_index}_mlp"](merged_feature))

        tar_embedding = split[2].reshape(-1, Args.SOLDIER_MAX_NUM, Args.DIM_SOLDIER)[:, :, -Args.DIM_UNIT :]
        tar_embedding = tar_embedding[:, :, : self.target_embed_dim]
        target_query = self.lstm_tar_embed_mlp(merged_feature).reshape(-1, self.target_embed_dim, 1)
        target_key = self.target_embed_mlp(tar_embedding)
        target_logits = torch.bmm(target_key, target_query).squeeze(-1)
        target_logits = target_logits[:, : self.label_size_list[-1]]
        result_list.append(target_logits)

        value_result = self.value_mlp(merged_feature)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = usq_label_list[shape_index].reshape(
                -1, self.data_split_shape[3 + shape_index]
            ).long()

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [element.squeeze(dim=1) for element in usq_label_list]
        weight_list = [element.squeeze(dim=1) for element in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary
                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_exp_logits = legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))
                self.policy_cost = self.policy_cost + temp_policy_loss

        entropy_loss_list = []
        current_entropy_loss_index = 0
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                temp_entropy_loss = -torch.sum(
                    temp_entropy_loss * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0))
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0))

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element
        self.entropy_cost_list = entropy_loss_list
        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for index in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[index], fc_feat_dim_list[index + 1])
            self.fc_layers.add_module(f"{name}_fc{index + 1}", fc_layer)
            if index + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{index + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
