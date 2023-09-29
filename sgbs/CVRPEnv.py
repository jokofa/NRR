
"""
The MIT License

Copyright (c) 2022 SGBS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch

DTYPE = torch.float


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(
            self,
            device: torch.device,
            pomo_size: int = -1
    ):

        self._device = device
        # Const @INIT
        ####################################
        self.pomo_size = pomo_size

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(
            self,
            coords: np.ndarray,
            demands: np.ndarray,
            batch_size: Optional[int] = None,
            aug_factor: int = 1,
            **kwargs
    ):
        assert len(coords.shape) == 3
        self.batch_size = batch_size if batch_size is not None else coords.shape[0]
        # added generate_problems argument to generate problems from different distribution
        self.problem_size = coords.shape[1]-1
        depot_xy = torch.from_numpy(coords[:, 0]).to(self._device, dtype=DTYPE).unsqueeze(1)
        node_xy = torch.from_numpy(coords[:, 1:]).to(self._device, dtype=DTYPE)
        node_demand = torch.from_numpy(demands[:, 1:]).to(self._device, dtype=DTYPE)
        if self.pomo_size != 1:
            self.pomo_size = self.problem_size-1

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        # if (self.load < 0).any():
        #     raise RuntimeError
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][
            ~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


class E_CVRPEnv(CVRPEnv):

    def modify_pomo_size(self, new_pomo_size):
        self.pomo_size = new_pomo_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        self.selected_count = bs_env.selected_count
        self.current_node = bs_env.current_node.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.selected_node_list = bs_env.selected_node_list.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = bs_env.at_the_depot.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.load = bs_env.load.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)
        self.visited_ninf_flag = bs_env.visited_ninf_flag.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = bs_env.ninf_mask.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo, problem+1)
        self.finished = bs_env.finished.repeat_interleave(repeat, dim=1)
        # shape: (batch, pomo)

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = rollout_env.selected_count
        self.current_node = rollout_env.current_node.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.selected_count)
        self.selected_node_list = rollout_env.selected_node_list.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = rollout_env.at_the_depot.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        self.load = rollout_env.load.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)
        exp_gathering_index = gathering_index[:, :, None].expand(-1, -1, self.problem_size+1)
        self.visited_ninf_flag = rollout_env.visited_ninf_flag.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = rollout_env.ninf_mask.gather(dim=1, index=exp_gathering_index)
        # shape: (batch, pomo, problem+1)
        self.finished = rollout_env.finished.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo)

        if gathering_index.size(1) != self.pomo_size:
            self.modify_pomo_size(gathering_index.size(1))

    def merge(self, other_env):

        self.current_node = torch.cat((self.current_node, other_env.current_node), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.selected_node_list = torch.cat((self.selected_node_list, other_env.selected_node_list), dim=1)
        # shape: (batch, pomo1 + pomo2, 0~)

        self.at_the_depot = torch.cat((self.at_the_depot, other_env.at_the_depot), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.load = torch.cat((self.load, other_env.load), dim=1)
        # shape: (batch, pomo1 + pomo2)
        self.visited_ninf_flag = torch.cat((self.visited_ninf_flag, other_env.visited_ninf_flag), dim=1)
        # shape: (batch, pomo1 + pomo2, problem+1)
        self.ninf_mask = torch.cat((self.ninf_mask, other_env.ninf_mask), dim=1)
        # shape: (batch, pomo1 + pomo2, problem+1)
        self.finished = torch.cat((self.finished, other_env.finished), dim=1)
        # shape: (batch, pomo1 + pomo2)
