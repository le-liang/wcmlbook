import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Environment_meta
import os
from torch.distributions import Categorical
from collections import namedtuple

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]]
width = 750/2
height = 1298/2
label = 'sarl_model_meta_ppo'
n_veh = 4
n_RB = n_veh

# Environment Parameters
env = Environment_meta.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh)
env.new_random_game()   # initialize parameters in env

# Test Environment Parameters
env_test = Environment_meta.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh)
env_test.new_random_game()

n_episode = 200  # Outer Loops
in_episode = 20  # Inner Loops
n_step_per_episode = int(env.time_slow/env.time_fast)  # each episode consists of 100 steps
mini_batch_step = 10 * n_step_per_episode
num_batch_task = 20  # each outer loop sample 20 tasks for training
num_batch_test_task = 10 # each outer loop sample 10 tasks for testing
n_episode_test = 10  # test episodes
######################################################


def get_state(env, idx=(0, 0)):
    """ Get state from the environment """

    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :]
                - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining))


n_input_size = len(get_state(env=env))
n_output_size = n_RB * len(env.V2V_power_dB_List_new)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

# ------------------- PPO agent -------------------------------------
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc_1 = nn.Linear(n_input_size, 500)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(250, 120)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(120, n_output_size)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        action_prob = F.softmax(self.fc_4(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_1 = nn.Linear(n_input_size, 500)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(500, 250)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(250, 120)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(120, 1)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x


class Agent:
    def __init__(self):
        self.LAMBDA = 0.95
        self.discount = 0.99
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.K_epoch = 8 # here we update the network 8 times using the collected 10 trajectories
        self.actor_rate = 0.0001
        self.critic_rate = 0.0003

        self.actor = Actor().to(device)
        self.old_actor = Actor().to(device)  # Old policy network
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_rate)

        self.critic = Critic().to(device)
        self.old_critic = Critic().to(device)  # Old value network
        self.old_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_rate)
        self.loss_func = nn.MSELoss()

        self.data_buffer = []  # To store the experience
        self.counter = 0       # the number of experience tuple in data_buffer

    def choose_action(self, s_t):
        #  Return the action, and the probability to choose this action
        s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.old_actor(s_t)
        c = Categorical(action_prob)
        action = c.sample()
        a_log_prob = action_prob[:, action.item()]
        return action.item(), a_log_prob.item()

    def store_transition(self, transition):
        self.data_buffer.append(transition)
        self.counter = self.counter + 1

    def sample(self):  # Sample all the data
        l_s, l_a, l_a_p, l_r, l_s_ = [], [], [], [], []
        for item in self.data_buffer:
            s, a, a_prob, r, s_ = item
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.long))
            l_a_p.append(torch.tensor([[a_prob]], dtype=torch.float))
            l_r.append(torch.tensor([r], dtype=torch.float))
            l_s_.append(torch.tensor([[s_]], dtype=torch.float))
        s = torch.cat(l_s, dim=0).to(device)
        a = torch.cat(l_a, dim=0).to(device)
        a_prob = torch.cat(l_r, dim=0).unsqueeze(1).to(device)
        r = torch.cat(l_r, dim=0).unsqueeze(1).to(device)
        s_ = torch.cat(l_s_, dim=0).squeeze(1).to(device)
        self.data_buffer = []
        return s, a, a_prob, r, s_

    def update(self):
        s, a, a_old_prob, r, s_ = self.sample()
        for _ in range(self.K_epoch):
            with torch.no_grad():
                td_target = r + self.discount * self.old_critic(s_)
                td_error = r + self.discount * self.critic(s_) - self.critic(s)
                td_error = td_error.detach().cpu().numpy()
                advantage = []  # Advantage Function
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * self.LAMBDA * self.discount + td[0]
                    advantage.append(adv)
                advantage.reverse()
                advantage = torch.tensor(advantage, dtype=torch.float).reshape(-1, 1).to(device)
                # Trick: Normalization
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

            a_new_prob = self.actor(s).gather(1, a)
            ratio = a_new_prob / a_old_prob.detach()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            critic_loss = self.loss_func(td_target.detach(), self.critic(s))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

    def save_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.actor.state_dict(), model_path + '_a.ckpt')
        torch.save(self.critic.state_dict(), model_path + '_c.ckpt')

    def load_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        self.actor.load_state_dict(torch.load(model_path + '_a.ckpt'))
        self.critic.load_state_dict(torch.load(model_path + '_c.ckpt'))

# meta learner using Reptile
class Metalearner:
    def __init__(self):
        # meta learning rate
        self.meta_rate = 1e-4

        # update network
        self.meta_actor = Actor().to(device)
        self.meta_critic = Critic().to(device)

        # target network
        self.old_meta_actor = Actor().to(device)
        self.old_meta_critic = Critic().to(device)

        # load the weight
        self.old_meta_actor.load_state_dict(self.meta_actor.state_dict())
        self.old_meta_critic.load_state_dict(self.old_meta_critic.state_dict())

        # List of agents
        self.agents = []
        # initialize agents
        for index_agent in range(num_batch_task):
            print("Initializing inner agent", index_agent)
            agent = Agent()
            self.agents.append(agent)

    def choose_action(self, s_t):
        #  Return the action, and the probability to choose this action
        s_t = torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.old_meta_actor(s_t)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item()

    def load_weight(self):
        # Each outer loop need to execute once
        for i in range(num_batch_task):
            self.agents[i].actor.load_state_dict(self.meta_actor.state_dict())
            self.agents[i].critic.load_state_dict(self.meta_critic.state_dict())
            self.agents[i].old_actor.load_state_dict(self.agents[i].actor.state_dict())
            self.agents[i].old_critic.load_state_dict(self.agents[i].critic.state_dict())

    def meta_update(self):
        # Reptile
        for i in range(num_batch_task):
            for param, old_param, train_param in zip(self.meta_actor.parameters(), self.old_meta_actor.parameters(),
                                                     self.agents[i].actor.parameters()):
                param.data.copy_(param.data + self.meta_rate / (self.agents[i].actor_rate * num_batch_task)
                                 * (train_param.data - old_param.data))
            for param, old_param, train_param in zip(self.meta_critic.parameters(), self.old_meta_critic.parameters(),
                                                     self.agents[i].critic.parameters()):
                param.data.copy_(param.data + self.meta_rate / (self.agents[i].critic_rate * num_batch_task)
                                 * (train_param.data - old_param.data))

        # after updat, target networks load the parameters of the update network
        self.old_meta_actor.load_state_dict(self.meta_actor.state_dict())
        self.old_meta_critic.load_state_dict(self.meta_critic.state_dict())

    def save_models(self, model_path):
        # save the models
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.meta_actor.state_dict(), model_path + '_meta_a.ckpt')
        torch.save(self.meta_critic.state_dict(), model_path + '_meta_c.ckpt')


# ----------------------------------------------------------------------------
print('device:', device)
print('input size:', n_input_size, 'output size:', n_output_size)
meta_agent = Metalearner()
# ------------------------- Training -----------------------------
record_out_reward = np.zeros([n_episode, 1])  # Need to be considered
record_test_reward = np.zeros([n_episode_test, 1])

i_episode = 0  # Count the overall episodes for updating vehicles

# This two lists are used to generate different tasks
N = range(3)
N_ = range(2)

for out_episode in range(n_episode):
    ## ---------------Outer Loop ---------------------------
    # Sample a batch of tasks
    num_neighbor = np.random.choice(N, size=num_batch_task)
    V2I_type = np.random.choice(N, size=num_batch_task)
    V2V_type = np.random.choice(N, size=num_batch_task)
    speed = np.random.choice(N, size=num_batch_task)
    demand_size = np.random.choice(N, size=num_batch_task)
    # Initialize the weight
    meta_agent.load_weight()

    for i_task in range(num_batch_task):
        ## ----------------------- Inner Loop -------------------------
        env.set_parameters(num_neighbor[i_task] + 1, V2I_type[i_task], V2V_type[i_task], speed[i_task] + 1, demand_size[i_task] * 2 + 2)
        action_all_training = np.zeros([n_veh, env.n_neighbor, 2], dtype='int32')
        time_step = 0

        for inn_episode in range(in_episode):
            if i_episode % 100 == 0:
                env.renew_positions()  # update vehicle position
                env.renew_neighbor()
                env.renew_channel()  # update channel slow fading
                env.renew_channels_fastfading()  # update channel fast fading

            env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
            env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
            env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

            for i_step in range(n_step_per_episode):
                time_step = inn_episode * n_step_per_episode + i_step
                remainder = i_step % (n_veh * env.n_neighbor)
                i = int(np.floor(remainder / env.n_neighbor))
                j = remainder % env.n_neighbor
                state = get_state(env, [i, j])
                action, a_prob = meta_agent.agents[i_task].choose_action(state)
                action_all_training[i, j, 0] = action % n_RB  # chosen RB
                action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level

                action_temp = action_all_training.copy()
                train_reward = env.act_for_training(action_temp)
                # record_inn_reward[inn_episode] += train_reward

                env.renew_channels_fastfading()
                env.Compute_Interference(action_temp)

                state_new = get_state(env, [i, j])
                trans = Transition(state, action, a_prob, train_reward, state_new)
                meta_agent.agents[i_task].store_transition(transition=trans)

                # training this agent
                if time_step % mini_batch_step == mini_batch_step - 1:
                    # update the networks using PPO
                    meta_agent.agents[i_task].update()
            i_episode = i_episode + 1

    # update the meta parameters
    meta_agent.meta_update()

    # Sample a batch of test tasks
    # Note: If we choose different testing tasks per outer loop, the average cumulative reward may fluctuate.
    # You can fix the testing tasks to obtain a more smooth curve.
    num_neighbor_test = np.random.choice(N_, size=num_batch_test_task)
    V2I_type_test = np.random.choice(N_, size=num_batch_test_task)
    V2V_type_test = np.random.choice(N_, size=num_batch_test_task)
    speed_test = np.random.choice(N_, size=num_batch_test_task)

    for i_test_task in range(num_batch_test_task):

        env_test.set_parameters_test(num_neighbor_test[i_test_task] + 1, V2I_type_test[i_test_task], V2V_type_test[i_test_task], speed_test[i_test_task] + 1)
        action_all_training = np.zeros([n_veh, env_test.n_neighbor, 2], dtype='int32')

        # Here, for simplicity, we don't roll out the agent in the new environment for 20 episodes.
        # You can add the adaptation stage here to obtain an exact curve. The tendency should be the same.

        with torch.no_grad():
            for idx_episode in range(n_episode_test):
                env_test.renew_positions()
                env_test.renew_neighbor()
                env_test.renew_channel()
                env_test.renew_channels_fastfading()

                env_test.demand = env_test.demand_size * np.ones((env_test.n_Veh, env_test.n_neighbor))
                env_test.individual_time_limit = env_test.time_slow * np.ones((env_test.n_Veh, env_test.n_neighbor))
                env_test.active_links = np.ones((env_test.n_Veh, env_test.n_neighbor), dtype='bool')

                V2I_rate_per_episode = []

                for test_step in range(n_step_per_episode):
                    remainder = test_step % (n_veh * env_test.n_neighbor)
                    i = int(np.floor(remainder / env_test.n_neighbor))
                    j = remainder % env_test.n_neighbor
                    state = get_state(env_test, [i, j])
                    action = meta_agent.choose_action(state)
                    action_all_training[i, j, 0] = action % n_RB  # chosen RB
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level
                    action_temp = action_all_training.copy()
                    test_reward = env_test.act_for_training(action_temp)
                    record_test_reward[idx_episode] += test_reward

                    env_test.renew_channels_fastfading()
                    env_test.Compute_Interference(action_temp)

    # record the average reward
    record_out_reward[out_episode] = np.mean(record_test_reward) / num_batch_test_task
    record_test_reward.fill(0)
    print('[Episode:', out_episode, 'reward:', record_out_reward[out_episode], ']')

# after training, save the results.
print('Training Done. Saving models...')
model_path = label + '/agent'
meta_agent.save_models(model_path)

current_dir = os.path.dirname(os.path.realpath(__file__))
reward_path = os.path.join(current_dir, 'model/' + label + '/reward.mat')
scipy.io.savemat(reward_path, {'reward': record_out_reward})
