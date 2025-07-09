import os
from datetime import datetime
import csv
from gym import spaces
import numpy as np
from scipy.optimize import fsolve


seed = 0
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


class csv_handler:
    def __init__(self, filename, fieldnames):
        self.fieldnames = fieldnames
        file_exists = os.path.isfile(filename) 
        self.f = open(filename, 'a', newline='')
        self.csv_writer = csv.DictWriter(self.f, fieldnames=fieldnames, delimiter=',')
        if not file_exists:
            self.csv_writer.writeheader()

    def addLine(self, in_dict):
        if len(in_dict.keys()) != len(self.fieldnames):
            print(in_dict.keys())
            print(in_dict)
            print(self.fieldnames)
        
        assert len(in_dict.keys()) == len(self.fieldnames)
        self.csv_writer.writerow(in_dict)
        self.f.flush()

    def close_csv(self):
        self.f.close()



class EnvTank:
    def __init__(self, algo, noise, create_log=1, timestamp='', save_contexts=0, contexts_file='', ver=1):
        
        assert noise is not None
        self.noise_dist = noise['dist']
        self.noise_var = noise['var']
        self.create_log = create_log
        self.save_contexts = save_contexts
        self.algo = algo
        self.version = ver
        
        ### START KINEMATICS
        self.vel = 1000
        self.grav = -9.81

        self.elmin = np.deg2rad(0)
        velrgdistmin = self.vel * np.cos(self.elmin)
        velhtdistmin = self.vel * np.sin(self.elmin)
        t_distmin = np.roots([0.5 * self.grav, velhtdistmin, 0])[0]
        self.distmin = velrgdistmin * t_distmin

        self.elopt = np.deg2rad(45)
        velrgopt = self.vel * np.cos(self.elopt)
        velhtopt = self.vel * np.sin(self.elopt)
        self.t_distmax = np.roots([0.5 * self.grav, velhtopt, 0])[0]
        self.distmax = velrgopt * self.t_distmax

        self.elmax = np.deg2rad(90)
        velhtmax = self.vel * np.sin(self.elmax)
        self.t_max = np.roots([0.5 * self.grav, velhtmax, 0])[0]
        ### END KINEMATICS
        
        self.step_counter = 0
        min_action, max_action, action_dim = 0, 2 * self.elmax, 1
        # min_action, max_action, action_dim = 0, self.elmax, 1
        self.min_obs, self.max_obs = -0.5 * self.distmax, 0.5 * self.distmax
        self.obs_dim = 2 if self.version == 2 else 1
        
        self.action_space = spaces.Box(low=min_action, high=max_action, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(self.obs_dim,), dtype=np.float32)
        
        # self.obs = self.observation_space.sample()
        self.get_obs()
                
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S") if timestamp=='' else timestamp
        self.path = os.path.join(os.getcwd(), 'results', self.timestamp)
        
        if self.save_contexts:
            assert len(contexts_file) == 0
            fieldnames = ['step', 'context0', 'context1']
            filename = os.path.join(self.path, self.timestamp+'_context_log.csv')
            os.makedirs(self.path, exist_ok=True)
            self.csv_h_contexts = csv_handler(filename, fieldnames)
            
            # context2 = self.obs[2] if self.version == 2 else -1
            measure = {'step' : self.step_counter, 'context0' : self.obs[0], 'context1': self.obs[1]}
            self.csv_h_contexts.addLine(measure)
            
        self.reset(contexts_file)


    def get_obs(self):
        self.obs = self.observation_space.sample()
        obs0 = self.obs[0]
        obs1 = self.obs[1]
        if obs0 > obs1:
            self.obs[0] = obs1
            self.obs[1] = obs0
        
        gapmin = self.distmax * 0.1
        nogohalf = gapmin / 2
        gohalf = self.max_obs - nogohalf
        if obs1 - obs0 < gapmin:
            # self.obs[0] = (np.random.rand() * gapmin / 2) + self.min_obs
            self.obs[0] = -nogohalf - (gohalf * np.random.rand())
            # self.obs[1] = self.max_obs - (np.random.rand() * gapmin / 2)
            self.obs[1] = nogohalf + (gohalf * np.random.rand())


    def reset(self, contexts_file=''):
        
        if self.create_log:
            fieldnames = ['step', 'context', 'action', 'reward1', 'reward2', 'alpha']
            filename = os.path.join(self.path, self.timestamp+f'_{self.algo}_log.csv')
            os.makedirs(self.path, exist_ok=True)
            self.csv_h = csv_handler(filename, fieldnames)
        
        
        if len(contexts_file) > 0:
            with open(contexts_file) as in_file:
                self.n_lines = sum(1 for _ in in_file)
            
            self.f = open(contexts_file)
            self.contexts_file = csv.reader(self.f, delimiter=',')
            next(self.contexts_file)
            line = next(self.contexts_file)
            self.obs = np.zeros(self.obs_dim)
            
            for i in range(self.obs_dim):
                self.obs[i] = float(line[i+1])
        else:
            self.contexts_file = None
    
    def close(self):
        if self.create_log:
            self.csv_h.close_csv()
        if self.save_contexts:
            self.csv_h_contexts.close_csv()
        if self.contexts_file is not None:
            self.f.close()


    def compute_reward(self, action):
        # print('ENV_TANK action[0]:\n{}'.format(action[0]))
        el = action[0]
        # print('\nCOMPUTE_REWARD el:\n{}\nobs:\n{}\n'.format(el, obs))
        velht = self.vel * np.sin(el)
        veldist = self.vel * np.cos(el)
        self.t_impact = np.roots([0.5 * self.grav, velht, 0])[0]

        self.dist_impact = veldist * self.t_impact
        self.loc_impact = self.obs[0] + np.array(self.dist_impact)
        
        self.hit_miss_dist = np.subtract(self.obs[1], self.loc_impact)
        dreward = np.power(10 * np.abs(self.hit_miss_dist) / np.abs(self.obs[1] - self.obs[0]), 1/4)
        # print('DIAG hit_miss_dist: {}, diff: {}, dreward: {}'.format(self.hit_miss_dist, self.obs[1] - self.obs[0], dreward))
        # dreward = np.abs(self.hit_miss_dist)
        # print('el: {}, t_impact: {}, dreward: {}'.format(el, self.t_impact, dreward))
        treward = (10 + np.sqrt(np.abs(self.t_impact))) * np.power(dreward, 2)

        # if self.hit_miss_dist < (self.dist * self.hitthresh):
        #     reward = rewardbase + self.rewardmax
        # else:
        #     reward = rewardbase

        # return treward, dreward
        return dreward, treward


    def compute_truthscore(self):  # accessing from outside
        def t_hit_opt(x):
            z = x - self.dist_impact /\
                (self.vel * np.cos(np.arcsin(-0.5 * self.grav * x / self.vel)))

            return z

        t_opt = fsolve(t_hit_opt, self.t_distmax / 2)[0]
        tdiff = np.abs(self.t_impact - t_opt)
        truthscore = np.array([self.hit_miss_dist, tdiff])

        return truthscore, t_opt


    def step(self, action, alpha=None):

        m1, m2 = self.compute_reward(action)
        # print('m1: {}, m2: {}'.format(m1, m2))
        truthscore, t_opt = self.compute_truthscore()
        
        if self.noise_dist == 'norm':
            # m1 += np.random.normal(loc=0.0, scale=self.noise_var)
            # m2 += np.random.normal(loc=0.0, scale=self.noise_var)
            pert_m1 = np.random.normal(loc=0.0, scale=self.noise_var[0])
            if m1 + pert_m1 < 0:
                m1 -= pert_m1
            elif m1 + pert_m1 >= 0:
                m1 += pert_m1
            if self.version == 2:
                # m3 += np.random.normal(loc=0.0, scale=self.noise_var)
                pert_m2 = np.random.normal(loc=0.0, scale=self.noise_var[1])
                # print(pert_m1)
                if m2 + pert_m2 < 0:
                    m2 -= pert_m2
                elif m2 + pert_m2 >= 0:
                    m2 += pert_m2
            
        if self.create_log:
            measure = {'step' : self.step_counter, 'context' : self.obs, 'action': action, 'reward1' : m1, 'reward2': m2, 'alpha': alpha}
            self.csv_h.addLine(measure)
            
        if self.contexts_file is not None:
            if self.step_counter >= self.n_lines:
                raise Exception('Context file ended.')
            line = next(self.contexts_file)
            for i in range(self.obs_dim):
                self.obs[i] = float(line[i+1])
        else:
            # self.obs = self.observation_space.sample()
            self.get_obs()
        self.step_counter += 1
        
        if self.save_contexts:
            # context2 = self.obs[2] if self.version == 2 else -1
            measure = {'step' : self.step_counter, 'context0' : self.obs[0], 'context1': self.obs[1]}
            self.csv_h_contexts.addLine(measure)
        
        obs = np.expand_dims(self.obs, axis=0)
        reward = (m1, m2) if self.version == 2 else (m1)
        done = False
        info = {}

        return obs, reward, done, info
    
    
    
    def gen_dataset(self, n_points, actions_per_context, ver=2):
        if ver == 1:
            return self.gen_dataset_v1(n_points, actions_per_context)
        elif ver == 2:
            return self.gen_dataset_v2(n_points, actions_per_context)
        
    
    def gen_dataset_v1(self, n_points, actions_per_context):
        n_context = int(np.floor(n_points / actions_per_context))
        actual_points = int(n_context * actions_per_context)
        
        contexts_actions = np.zeros((actual_points, 3))
        m1 = np.zeros((actual_points, 1))
        m2 = np.zeros((actual_points, 1))
        
        idx = 0
        
        for i in range(n_context):
            obs = self.observation_space.sample()
            for j in range(actions_per_context):
                contexts_actions[idx, 0:2] = obs
                action = self.action_space.sample()
                contexts_actions[idx, 2] = action

                m1_i = obs[0]*action**2 + obs[1]*action 
                m2_i = obs[0]*action**2 - obs[1]*action 
                m1_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m2_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m1[idx] = m1_i
                m2[idx] = m2_i
                
                idx += 1
            
        return contexts_actions, m1, m2
    
    def gen_dataset_v2(self, n_points, actions_per_context):
        n_context = int(np.floor(n_points / actions_per_context))
        actual_points = int(n_context * actions_per_context)
        
        # contexts_actions = np.zeros((actual_points, 4))
        contexts_actions = np.zeros((actual_points, 3))
        m1 = np.zeros((actual_points, 1))
        # m23 = np.zeros((actual_points, 2))
        m2 = np.zeros((actual_points, 1))
        
        idx = 0
        
        for i in range(n_context):
            obs = self.observation_space.sample()
            for j in range(actions_per_context):
                contexts_actions[idx, 0:2] = obs
                action = self.action_space.sample()
                contexts_actions[idx, 2] = action

                # m1_i = obs[0]*action**2 + obs[1]*action 
                # m2_i = obs[0]*action**2 - obs[1]*action 
                m1_i, m2_i = self.compute_reward(action, obs)
                # m3_i = obs[0]*(action-obs[2])**2 - obs[1]*(action-obs[2])
                
                m1_i += np.random.normal(loc=0.0, scale=self.noise_var)
                m2_i += np.random.normal(loc=0.0, scale=self.noise_var)
                # m3_i += np.random.normal(loc=0.0, scale=self.noise_var)
                # print('GEN_DATASET_V2 m1_i:\n{}\n'.format(m1_i))
                m1[idx] = m1_i
                # m23[idx,:] = [m2_i, m3_i]
                m2[idx, :] = [m2_i]
                
                idx += 1
            
        # return contexts_actions, m1, m23
        return contexts_actions, m1, m2
