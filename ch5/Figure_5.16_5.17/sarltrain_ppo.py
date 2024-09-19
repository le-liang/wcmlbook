import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Environment_marl
import os
from torch.distributions import Categorical
from collections import namedtuple

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ################## SETTINGS ######################
# The configuration of the V2X environment
up_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]]
width = 750/2
height = 1298/2
label = 'sarl_model_ppo_4_1' # the label of the specific task
n_veh = 4
n_neighbor = 1
n_RB = n_veh
# Environment Parameters
env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()   # initialize parameters in env

n_episode = 6000
n_step_per_episode = int(env.time_slow/env.time_fast)  # each episode consists of 100 time steps

mini_batch_step = 10 * n_step_per_episode  # update once after collecting 10 trajectories

n_episode_test = 100  # test episodes
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

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining))


n_input_size = len(get_state(env=env))
n_output_size = n_RB * len(env.V2V_power_dB_List_new)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

# the actor network
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

# the critic network
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

# the PPO agent
class Agent:
    def __init__(self):
        self.LAMBDA = 0.95
        self.discount = 0.99
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.K_epoch = 8  # here we update the network 8 times using the collected 10 trajectories

        self.actor = Actor().to(device)  # target network
        self.old_actor = Actor().to(device)  # sample network
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)

        self.critic = Critic().to(device)
        self.old_critic = Critic().to(device)
        self.old_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0003)

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

    def sample(self):    # Sample all the data
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
        a_prob = torch.cat(l_a_p, dim=0).unsqueeze(1).to(device)
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
        self.actor.load_state_dict(torch.load(model_path + '_meta_a.ckpt'))
        self.critic.load_state_dict(torch.load(model_path + '_meta_c.ckpt'))


# --------------------------------------------------------------
print("Initializing agent...")
agent = Agent()
# ------------------------- Training -----------------------------
record_reward = np.zeros([n_episode, 1])
record_reward_rand = np.zeros([n_episode, 1])
action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
time_step = 0
for i_episode in range(n_episode):
    # Trick: fix the position of the vehicles for several episodes to improve learning.
    if i_episode % 100 == 0:
        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        env.renew_channel()  # update channel slow fading
        env.renew_channels_fastfading()  # update channel fast fading

    env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
    env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
    env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

    env.demand_rand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
    env.individual_time_limit_rand = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
    env.active_links_rand = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

    for i_step in range(n_step_per_episode):
        time_step = i_episode * n_step_per_episode + i_step  # the counter of the time step
        remainder = i_step % (n_veh * n_neighbor)
        i = int(np.floor(remainder / n_neighbor))  # index of the vehicle
        j = remainder % n_neighbor # the index of the V2V link of the selected vehicle
        state = get_state(env, [i, j])
        action, a_prob = agent.choose_action(state)
        action_all_training[i, j, 0] = action % n_RB  # chosen RB
        action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level

        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp)
        record_reward[i_episode] += train_reward

        # random baseline
        action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
        action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor])  # band
        action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor])  # power
        action_rand_temp = action_rand.copy()
        train_reward_rand = env.act_for_testing_rand(action_rand_temp)
        record_reward_rand[i_episode] += train_reward_rand

        env.renew_channels_fastfading()
        env.Compute_Interference(action_temp)

        state_new = get_state(env, [i, j])
        trans = Transition(state, action, a_prob, train_reward, state_new)
        agent.store_transition(transition=trans)

        # training this agent
        if time_step % mini_batch_step == mini_batch_step - 1:
            # PPO method to update the network parameters
            agent.update()
            if i_episode % 10 == 9:
                print('[Episode:', i_episode, 'reward:', record_reward[i_episode], ']')

print('Training Done. Saving models...')
model_path = label + '/agent'
agent.save_models(model_path)

current_dir = os.path.dirname(os.path.realpath(__file__))
reward_path = os.path.join(current_dir, 'model/' + label + '/reward.mat')
scipy.io.savemat(reward_path, {'reward': record_reward})

current_dir = os.path.dirname(os.path.realpath(__file__))
reward_path = os.path.join(current_dir, 'model/' + label + '/reward_rand.mat')
scipy.io.savemat(reward_path, {'reward_rand': record_reward_rand})
