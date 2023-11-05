import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from torch.distributions import Normal,Beta,MultivariateNormal

class ReplayBuffer():
    def __init__(self, input_shape, n_actions, max_size=int(1e6)):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros((self.mem_size,))
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states,actions,rewards,states_,dones
    
    def delete_first_n(self,n):
        """_summary_

        Args:
            n (_type_): _description_
        """        
        self.state_memory = np.delete(self.state_memory,np.s_[0:n],0)
        self.action_memory = np.delete(self.action_memory,np.s_[0:n],0)
        self.reward_memory = np.delete(self.reward_memory,np.s_[0:n],0)
        self.new_state_memory = np.delete(self.next_state_memory,np.s_[0:n],0)
        self.terminal_memory = np.delete(self.terminal_memory,np.s_[0:n],0)
        self.mem_cntr -= n

        # add n samples in the buffer to keep the size of the buffer constant
        self.state_memory = np.append((self.state_memory,np.zeros((n,*self.input_shape),dtype=np.float32)), axis=0)
        self.action_memory = np.append((self.action_memory,np.zeros((n,self.n_actions),dtype=np.float32)), axis=0)
        self.reward_memory = np.append((self.reward_memory,np.zeros((n,),dtype=np.float32)), axis=0)
        self.new_state_memory = np.append((self.new_state_memory,np.zeros((n,*self.input_shape),dtype=np.float32)), axis=0)
        self.terminal_memory = np.append((self.terminal_memory,np.zeros((n,),dtype=bool)), axis=0)


class ActorNetwork(nn.Module):
    def __init__(self,alpha,input_dims,n_actions,fc1_dims,fc2_dims,name,\
        chkpt_dir='tmp/sac',max_action=1,act_dist='beta'):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        name += '.pt'
        self.chkpt_file = os.path.join(chkpt_dir,name)
        self.reparam_noise=1e-6
        self.act_dist = act_dist

        self.gru = nn.GRU(self.input_dims[1],self.fc1_dims,batch_first=True)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims,self.n_actions)      
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.max_action = T.tensor(max_action).to(self.device)

    def forward(self,state):
        x = self.gru(state)[1][0]
        x = F.silu(self.fc2(x))
        if self.act_dist == 'beta':
            mu = 1+F.softplus(self.mu(x))
            sigma = 1+F.softplus(self.sigma(x))
        else:
            mu = self.mu(x)
            sigma = self.sigma(x)
            sigma = T.clamp(sigma,-20,2).exp()
        return mu,sigma

    def sample_action(self,state,deterministic=False):
        mu,sigma = self.forward(state)
        if deterministic is True:
            if self.act_dist == 'beta':
                T.distributions.beta.Beta(mu,sigma).mean, None
            else:
                return T.tanh(mu),None
        if self.act_dist == 'normal':
            pi_dist = Normal(mu,sigma)
            pi_action = pi_dist.rsample() 
            log_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
            corr = (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            log_pi -= corr
            pi_action = self.max_action*T.tanh(pi_action)
        elif self.act_dist == 'mv_normal':
            pi_dist = MultivariateNormal(mu,sigma)
            pi_action = pi_dist.rsample() 
            log_pi = pi_dist.log_prob(pi_action)
            pi_action = self.max_action*T.tanh(pi_action)
        elif self.act_dist == 'beta':
            pi_dist = Beta(mu,sigma)
            pi_action = pi_dist.rsample()
            log_pi = pi_dist.log_prob(pi_action).sum(axis=1)
            pi_action = self.max_action*pi_action
        return pi_action,log_pi
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(),self.chkpt_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.chkpt_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
            n_actions,name, chkpt_dir='./tmp/sac'):#
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.gru = nn.GRU(self.input_dims[1], self.fc1_dims, batch_first=True)
        self.fc2 = nn.Linear(self.fc1_dims+self.n_actions, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)  

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q = self.gru(state)[1][0]
        q = T.cat([q, action], dim=1)
        q = F.silu(self.fc2(q))
        q = self.q1(q)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
    

class SAC_Agent:
    def __init__(self, alpha,beta,tau,input_dims,n_actions,env_id,gamma=0.9,act_dist='beta',
                 max_size=int(1e6),fc1_dims=256,fc2_dims=256,batch_size=256,max_action=5):
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.actor = ActorNetwork(alpha=alpha,input_dims=input_dims,n_actions=n_actions,
                                  name=env_id+'_actor',fc1_dims=fc1_dims,fc2_dims=fc2_dims,
                                  act_dist=act_dist,max_action=max_action)
        self.critic_1 = CriticNetwork(beta=beta,input_dims=input_dims,n_actions=n_actions,
                                        name=env_id+'_critic_1',fc1_dims=fc1_dims,fc2_dims=fc2_dims)
        self.critic_2 = CriticNetwork(beta=beta,input_dims=input_dims,n_actions=n_actions,
                                        name=env_id+'_critic_2',fc1_dims=fc1_dims,fc2_dims=fc2_dims)
        self.target_critic_1 = CriticNetwork(beta=beta,input_dims=input_dims,n_actions=n_actions,
                                        name=env_id+'_target_critic_1',fc1_dims=fc1_dims,fc2_dims=fc2_dims)
        self.target_critic_2 = CriticNetwork(beta=beta,input_dims=input_dims,n_actions=n_actions,
                                        name=env_id+'_target_critic_2',fc1_dims=fc1_dims,fc2_dims=fc2_dims)
        self.memory = ReplayBuffer(input_dims,n_actions,max_size=max_size)

        self.update_network_parameters(tau=1)
        self.target_ent_coef = -1*n_actions
        self.log_ent_coef = T.log(T.ones(1, device=self.actor.device)).requires_grad_(True)
        self.ent_coef_optim = optim.Adam([self.log_ent_coef], lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    
    def choose_action(self,obs,deterministic=False):
        self.actor.eval()
        state = T.tensor(obs,dtype=T.float).to(self.actor.device)
        action,_ = self.actor.sample_action(state,deterministic)
        self.actor.train()
        return action.cpu().detach().numpy()[0]
    
    def remember(self,state,action,reward,state_,done):
        self.memory.store_transition(state,action,reward,state_,done)
    
    def mem_ready(self):
        return self.memory.mem_cntr > self.batch_size
    
    def learn(self):

        state,action,reward,new_state,done = \
                self.memory.sample_buffer(self.batch_size)
        
        state = T.tensor(state,dtype=T.float).to(self.device)
        action = T.tensor(action,dtype=T.float).to(self.device)
        reward = T.tensor(reward,dtype=T.float).to(self.device)
        new_state = T.tensor(new_state,dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)

        actions_,log_probs_ = self.actor.sample_action(new_state)
        actions,log_probs = self.actor.sample_action(state)

        ent_coef_loss = -(self.log_ent_coef * (log_probs + self.target_ent_coef).detach()).mean()
        ent_coef = self.log_ent_coef.exp().detach()

        # Critic Optimization
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_ = self.target_critic_1.forward(new_state,actions_)
        q2_ = self.target_critic_2.forward(new_state,actions_)

        q_ = T.min(q1_,q2_).view(-1)

        target = (reward.view(-1) + (1-done.int())*self.gamma*(q_ - ent_coef*log_probs_)).view(-1)

        q1 = self.critic_1.forward(state,action).view(-1)
        q2 = self.critic_2.forward(state,action).view(-1)

        q1_loss = 0.5*F.mse_loss(target,q1)
        q2_loss = 0.5*F.mse_loss(target,q2)

        q_loss = q1_loss + q2_loss
        cl = q_loss.item()        
        q_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Actor Optimization
        actions,log_probs = self.actor.sample_action(state)
        q1 = self.critic_1.forward(state,actions)
        q2 = self.critic_2.forward(state,actions)
        q = T.min(q1,q2).view(-1)
        actor_loss = (ent_coef*log_probs - q).mean()
        al = actor_loss.item()             
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor.optimizer.step()

        # Entropy Coefficient Optimization
        self.ent_coef_optim.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optim.step()

        self.update_network_parameters()
        return al,cl,ent_coef_loss.item(),ent_coef.item()

    def update_network_parameters(self,tau=None):
        if tau is None:
            tau = self.tau
        target_critic_1_params = self.target_critic_1.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_1_state_dict = dict(critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_2_state_dict = dict(critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()
        
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
    
    def save_models(self):
        self.actor.save_checkpoint()
        # self.critic_1.save_checkpoint()
        # self.critic_2.save_checkpoint()
        # self.target_critic_1.save_checkpoint()
        # self.target_critic_2.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        # self.critic_1.load_checkpoint()
        # self.critic_2.load_checkpoint()
        # self.target_critic_1.load_checkpoint()
        # self.target_critic_2.load_checkpoint()
