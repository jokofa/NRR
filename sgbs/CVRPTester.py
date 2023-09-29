
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

from logging import getLogger
import copy
import numpy as np
import torch

from .CVRPEnv import E_CVRPEnv


class CVRPTester:
    def __init__(
            self,
            env: E_CVRPEnv,
            model,
            tester_params,
            cuda: bool
    ):

        # save arguments
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')

        # cuda
        if cuda:
            assert torch.cuda.is_available()
            cuda_device_num = 0
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = env
        self.model = model.to(device=device)
        self._pomo_size = int(self.env.pomo_size)

        self._METHODS = {
            'greedy': self._test_one_batch_greedy,
            'sampling': self._test_one_batch_sampling,
            'obs': self._test_one_batch_original_beam_search,
            'mcts': self._test_one_batch_mcts,
            'sgbs': self._test_one_batch_simulation_guided_beam_search,
        }

    @torch.no_grad()
    def run(
            self,
            coords: np.ndarray,
            demands: np.ndarray,
            **kwargs
    ):
        assert len(coords.shape) == 3
        batch_size = coords.shape[0]
        gs = coords.shape[1]
        if self._pomo_size != 1:
            self._pomo_size = gs-1

        self.env.load_problems(
            coords=coords,
            demands=demands,
            aug_factor=self.tester_params['aug_factor'],
            **kwargs
        )

        # set test method
        inf_method = self._METHODS[self.tester_params['mode']]
        score, aug_score, sol = inf_method(batch_size)
        if batch_size == 1:
            aug_score = [aug_score]
        return aug_score, sol

    def _get_pomo_starting_points(self, model, env, num_starting_points):

        # Ready
        ###############################################
        model.eval()
        env.modify_pomo_size(self._pomo_size)
        env.reset()

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        # starting points
        ###############################################
        sorted_index = reward.sort(dim=1, descending=True).indices
        selected_index = sorted_index[:, :num_starting_points]
        selected_index = selected_index + 1  # depot is 0, and node index starts from 1
        # shape: (batch, num_starting_points)

        return selected_index

    def _get_aug_cfg(self, batch_size):
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        aug_batch_size = aug_factor * batch_size
        return aug_factor, aug_batch_size

    def _ready_setup(self, batch_size, aug_factor):
        self.model.eval()
        #with torch.no_grad():
            #self.env.load_problems(batch_size, aug_factor=aug_factor)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

    def _gather_results(self, reward, selected_all, aug_factor, batch_size):
        # best pomo start
        best_pomo_rew, best_pomo_idx = reward.max(dim=-1)
        best_pomo_sol = selected_all[torch.arange(reward.size(0), device=reward.device), best_pomo_idx]

        aug_reward = best_pomo_rew.reshape(aug_factor, batch_size).permute(1, 0)
        aug_sol = best_pomo_sol.reshape(aug_factor, batch_size, -1).permute(1, 0, 2)
        best_aug_rew, best_aug_idx = aug_reward.max(dim=-1)
        best_aug_sol = aug_sol[torch.arange(aug_reward.size(0), device=reward.device), best_aug_idx]

        no_aug_score = -aug_reward[0].cpu()
        aug_score = -best_aug_rew.cpu()

        # selected_all = selected_all.reshape(aug_factor, batch_size, self.env.pomo_size, -1)
        # aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # ###
        # max_pomo_reward, indices_pomo = aug_reward.max(dim=2)  # get best results
        # #all_routes = torch.cat(selected_all, dim=-1)
        # best_sol_pomo = selected_all.gather(
        #     index=indices_pomo[:, :, None].expand(-1, -1, selected_all.size(-1)),
        #     dim=1
        # ) #[:, indices_pomo[0], :]
        # no_aug_score = -max_pomo_reward[0, :].float() #.mean(0)  # negative sign to make positive value
        # max_aug_pomo_reward, index_augm = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # #best_sol_pomo_augm = best_sol_pomo[index_augm][0]
        # best_sol_pomo_augm = best_sol_pomo.permute(1,0,2)[
        #     torch.arange(batch_size, device=best_sol_pomo.device), index_augm
        # ]
        # aug_score = -max_aug_pomo_reward.float()    #.mean()
        return no_aug_score.cpu(), aug_score.cpu(), best_aug_sol

    def _test_one_batch_greedy(self, batch_size):

        aug_factor, aug_batch_size = self._get_aug_cfg(batch_size)
        self._ready_setup(batch_size, aug_factor)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        return self._gather_results(
            reward=reward,
            selected_all=self.env.selected_node_list,
            aug_factor=aug_factor,
            batch_size=batch_size
        )

    def _test_one_batch_sampling(self, batch_size):
        num_sampling = self.tester_params['sampling_num']
        aug_factor, aug_batch_size = self._get_aug_cfg(batch_size)
        self._ready_setup(batch_size, aug_factor)
        # POMO Starting Points
        starting_points = self._get_pomo_starting_points(
            self.model, self.env, self._pomo_size
        )
        num_repeat = (num_sampling // starting_points.size(1)) + 1
        pomo_starting_points = starting_points.repeat(1, num_repeat)[:, :num_sampling]

        # Sampling
        ###############################################
        self.env.modify_pomo_size(num_sampling)
        self.env.reset()

        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.env.pomo_size), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points
        state, _, done = self.env.step(pomo_starting_points)

        while not done:
            selected, _ = self.model(state, 'softmax')
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        return self._gather_results(
            reward=reward,
            selected_all=self.env.selected_node_list,
            aug_factor=aug_factor,
            batch_size=batch_size
        )

    def _test_one_batch_original_beam_search(self, batch_size):
        beam_width = self.tester_params['obs_bw']
        aug_factor, aug_batch_size = self._get_aug_cfg(batch_size)
        self._ready_setup(batch_size, aug_factor)
        # POMO Starting Points
        starting_points = self._get_pomo_starting_points(
            self.model, self.env, self._pomo_size
        )

        # Beam Search
        ###############################################
        # reset
        traj_log_prob_sum = torch.zeros(size=(aug_batch_size, starting_points.size(1)))
        self.env.modify_pomo_size(starting_points.size(1))
        self.env.reset()

        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.env.pomo_size), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points
        state, _, done = self.env.step(starting_points)

        # LOOP
        while not done:
            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(state)
            # shape: (aug_batch, beam_width, problem+1)

            traj_log_prob_sum_exp = traj_log_prob_sum[:, :, None] + probs.log()
            # shape: (aug_batch, beam, problem+1)

            ordered_prob, ordered_i = traj_log_prob_sum_exp.reshape(aug_batch_size, -1).sort(dim=1, descending=True)

            ordered_i_selected = ordered_i[:, :beam_width]
            # shape: (aug*batch, beam)

            beam_selected = ordered_i_selected // probs.size(2)
            # shape: (aug*batch, beam)

            action_selected = ordered_i_selected % probs.size(2)
            # shape: (aug*batch, beam)

            # traj_log_prob_sum
            ###############################################
            traj_log_prob_sum = ordered_prob[:, :beam_width]

            # BS Step
            ###############################################
            self.env.reset_by_gathering_rollout_env(self.env, gathering_index=beam_selected)
            state, reward, done = self.env.step(action_selected)

        return self._gather_results(
            reward=reward,
            selected_all=self.env.selected_node_list,
            aug_factor=aug_factor,
            batch_size=batch_size
        )

    def _test_one_batch_mcts(self, batch_size):
        assert batch_size == 1
        class Node:
            init_q = -100
            init_qn = 1.0
            cpuct = 2

            def __init__(self, node_action, parent_node, child_probs, ninf_mask, env, done):
                self.node_action = node_action
                self.P = child_probs.clone()
                self.ninf_mask = ninf_mask.clone()

                self.Q = torch.ones(size=self.P.size()) * Node.init_q
                self.N = torch.zeros(size=self.P.size())
                self.zeros = torch.zeros(size=self.P.size())
                self.init_qn = torch.ones(size=self.P.size()) * Node.init_qn

                self.parent_node = parent_node
                self.child_node = [None] * self.P.size(0)

                self.wp = torch.tensor(float('inf'))
                self.bp = torch.tensor(float('-inf'))

                self.env = copy.deepcopy(env)
                self.done = done

            def _get_uct(self):
                if self.N.sum() == 0:
                    return self.P

                Qn = (self.Q - self.wp) / (self.bp - self.wp) if self.bp - self.wp > 0 else self.zeros
                Qn = torch.where(self.Q == Node.init_q, self.init_qn, Qn)

                U = Node.cpuct * self.P * self.N.sum().sqrt() / (0.1 + self.N)

                return Qn + U + self.ninf_mask

            def select_next(self):
                uct = self._get_uct()
                idx = uct.argmax(dim=0)

                return self.child_node[idx], idx

            def set_child(self, idx, node):
                self.child_node[idx] = node

            def set_parent(self, parent_node):
                self.parent_node = parent_node

            def get_parent(self):
                return self.parent_node

            def update_q(self, idx, new_q):
                self.Q[idx] = max(self.Q[idx], new_q)
                self.N[idx] += 1

                self.bp = max(self.bp, new_q)
                self.wp = min(self.wp, new_q)

            def select_child_to_move(self):
                Qn_valid = self.Q + self.ninf_mask
                child_idx = Qn_valid.argmax(dim=0)

                return child_idx

            def get_child_to_move(self, child_idx):
                return self.child_node[child_idx]

            def get_env_state(self):
                return copy.deepcopy(self.env), self.env.step_state, self.done

        rollout_per_step = self.tester_params['mcts_rollout_per_step']
        aug_factor = 1  # no augs for mcts
        self._ready_setup(batch_size, aug_factor)

        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, 1)
        # shape: (aug*batch_size, 1)

        # MCTS
        ###############################################
        # reset
        self.env.modify_pomo_size(starting_points.size(1))
        self.env.reset()

        # the first step, depot
        selected = torch.zeros(size=(1, 1), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points
        state, _, done = self.env.step(starting_points)

        # MCTS Step > 1
        ###############################################

        # LOOP
        next_root = None
        while not done:
            node_root = next_root

            if node_root == None:
                probs = self.model.get_expand_prob(state)
                # shape: (aug_batch, 1, problem+1)

                node_root = Node(0, None,
                                 probs[0, 0],
                                 state.ninf_mask[0, 0],
                                 self.env, done)

            node_root.set_parent(None)

            for mcts_cnt in range(rollout_per_step):
                # selection
                ###############################################
                node_curr = node_root
                node_next, idx_next = node_curr.select_next()

                while node_next is not None:
                    node_curr = node_next
                    node_next, idx_next = node_curr.select_next()

                # expansion
                ###############################################
                simulation_env, sim_state, sim_done = node_curr.get_env_state()
                if sim_done:
                    continue

                sim_state, sim_reward, sim_done = simulation_env.step(idx_next[None, None])

                sim_probs = self.model.get_expand_prob(sim_state)

                node_exp = Node(idx_next, node_curr,
                                sim_probs[0, 0],
                                sim_state.ninf_mask[0, 0],
                                simulation_env, sim_done)
                node_curr.set_child(idx_next, node_exp)

                # simulation
                ###############################################
                while not sim_done:
                    selected, _ = self.model(sim_state)
                    sim_state, sim_reward, sim_done = simulation_env.step(selected)

                new_q = sim_reward[0, 0]

                # backprop
                ###############################################
                node_curr = node_exp
                node_parent = node_curr.get_parent()

                while node_parent is not None:
                    idx_curr = node_curr.node_action
                    node_parent.update_q(idx_curr, new_q)

                    node_curr = node_parent
                    node_parent = node_curr.get_parent()

            action = node_root.select_child_to_move()
            next_root = node_root.get_child_to_move(action)
            state, reward, done = self.env.step(action[None, None])

        #return reward[0, 0]
        return self._gather_results(
            reward=reward,
            selected_all=self.env.selected_node_list,
            aug_factor=aug_factor,
            batch_size=batch_size
        )

    def _test_one_batch_simulation_guided_beam_search(self, batch_size):
        beam_width = self.tester_params['sgbs_beta']
        expansion_size_minus1 = self.tester_params['sgbs_gamma_minus1']
        rollout_width = beam_width * expansion_size_minus1
        aug_factor, aug_batch_size = self._get_aug_cfg(batch_size)
        self._ready_setup(batch_size, aug_factor)
        # POMO Starting Points
        ###############################################
        starting_points = self._get_pomo_starting_points(self.model, self.env, beam_width)

        # Beam Search
        ###############################################
        self.env.modify_pomo_size(beam_width)
        self.env.reset()

        # the first step, depot
        selected = torch.zeros(size=(aug_batch_size, self.env.pomo_size), dtype=torch.long)
        state, _, done = self.env.step(selected)

        # the second step, pomo starting points
        state, _, done = self.env.step(starting_points)

        # BS Step > 1
        ###############################################

        # Prepare Rollout-Env
        rollout_env = copy.deepcopy(self.env)
        rollout_env.modify_pomo_size(rollout_width)

        # LOOP
        reward = None
        first_rollout_flag = True
        while not done:

            # Next Nodes
            ###############################################
            probs = self.model.get_expand_prob(state)
            # shape: (aug*batch, beam, problem+1)
            ordered_prob, ordered_i = probs.sort(dim=2, descending=True)

            greedy_next_node = ordered_i[:, :, 0]
            # shape: (aug*batch, beam)

            if first_rollout_flag:
                prob_selected = ordered_prob[:, :, :expansion_size_minus1]
                idx_selected = ordered_i[:, :, :expansion_size_minus1]
                # shape: (aug*batch, beam, rollout_per_node)
            else:
                prob_selected = ordered_prob[:, :, 1:expansion_size_minus1 + 1]
                idx_selected = ordered_i[:, :, 1:expansion_size_minus1 + 1]
                # shape: (aug*batch, beam, rollout_per_node)

            # replace invalid index with redundancy
            next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
            is_valid = (prob_selected > 0)
            next_nodes[is_valid] = idx_selected[is_valid]
            # shape: (aug*batch, beam, rollout_per_node)

            # Rollout to get rollout_reward
            ###############################################
            rollout_env.reset_by_repeating_bs_env(self.env, repeat=expansion_size_minus1)
            rollout_env_deepcopy = copy.deepcopy(rollout_env)  # Saved for later

            next_nodes = next_nodes.reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)

            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
            while not rollout_done:
                selected, _ = self.model(rollout_state)
                # shape: (aug*batch, rollout_width)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected)
            # rollout_reward.shape: (aug*batch, rollout_width)

            # mark redundant
            is_redundant = (~is_valid).reshape(aug_batch_size, rollout_width)
            # shape: (aug*batch, rollout_width)
            rollout_reward[is_redundant] = float('-inf')

            # Merge Rollout-Env & BS-Env (Optional, slightly improves performance)
            ###############################################
            if first_rollout_flag is False:
                rollout_env_deepcopy.merge(self.env)
                rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
                # rollout_reward.shape: (aug*batch, rollout_width + beam_width)
                next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
                # next_nodes.shape: (aug*batch, rollout_width + beam_width)
            first_rollout_flag = False

            # BS Step
            ###############################################
            sorted_reward, sorted_index = rollout_reward.sort(dim=1, descending=True)
            beam_reward = sorted_reward[:, :beam_width]
            beam_index = sorted_index[:, :beam_width]
            # shape: (aug*batch, beam_width)

            self.env.reset_by_gathering_rollout_env(rollout_env_deepcopy, gathering_index=beam_index)
            selected = next_nodes.gather(dim=1, index=beam_index)
            # shape: (aug*batch, beam_width)
            state, reward, done = self.env.step(selected)

        return self._gather_results(
            reward=reward,
            selected_all=self.env.selected_node_list,
            aug_factor=aug_factor,
            batch_size=batch_size
        )
