import numpy as np
from itertools import product
import warnings

class ActionSpace(object):

    def __init__(self, discvars, contexts, beta_val, safeset, constraint_thres, constr_greater, minimization=1):

        self.beta_val = beta_val
        self.constraint_thres = constraint_thres
        self.n_constraints = len(constraint_thres)
        self.constr_greater = constr_greater
        self.minimization = minimization
        # Get the name of the parameters
        self._action_keys = discvars.keys()
        self._context_keys = contexts.keys()
        
        allList = [discvars[k] for k in discvars.keys()]
        allActions = np.array(list(product(*allList)))
        self._allActions = allActions

        # preallocated memory for X and Y points
        self._context = np.empty(shape=(0, self.context_dim))
        self._action = np.empty(shape=(0, self.action_dim))
        self._context_action = np.empty(shape=(0, self.context_dim + self.action_dim))
        self._reward = np.empty(shape=(0))
        self._constraint = np.empty(shape=(0, self.n_constraints))
        self._lower = np.zeros((len(allActions), self.n_constraints+1))
        self._upper = np.zeros((len(allActions), self.n_constraints+1))

        
        self._S = np.zeros(len(allActions), dtype=bool)
        self._S0_idx = []
        # print('ACTIONSPACE safeset:\n{}'.format(safeset))
        # print('ACTIONSPACE allActions:\n{}'.format(allActions))
        for i in range(len(safeset)):
            # print('ACTIONSPACE safeset[i,:]:\n{}'.format(safeset[i,:]))
            idx = np.where((safeset[i,:] == allActions).all(axis=1))[0][0]
            self._S0_idx.append(idx)
            self._S[idx] = True
        self._S0_idx = np.array(self._S0_idx)
        self.empty_safeset = 1
        
    def __len__(self):
        assert len(self._action) == len(self._reward)
        assert len(self._action) == len(self._constraint)
        assert len(self._action) == len(self._context)
        return len(self._reward)

    @property
    def empty(self):
        return len(self) == 0
    
    @property
    def context(self):
        return self._context

    @property
    def action(self):
        return self._action
    
    @property
    def context_action(self):
        return self._context_action

    @property
    def reward(self):
        return self._reward
    
    @property
    def constraint_vals(self):
        return self._constraint
    
    @property
    def context_dim(self):
        return len(self._context_keys)

    @property
    def action_dim(self):
        return len(self._action_keys)

    @property
    def context_keys(self):
        return self._context_keys
    
    @property
    def action_keys(self):
        return self._action_keys

    @property
    def bounds(self):
        return self._bounds

    def action_to_array(self, action):
        try:
            assert set(action) == set(self._action_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(action)) +
                "not match the expected set of keys ({}).".format(self._action_keys)
            )
        return np.asarray([action[key] for key in self._action_keys])
    
    def context_to_array(self, context):
        try:
            assert set(context) == set(self._context_keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(context)) +
                "not match the expected set of keys ({}).".format(self._context_keys)
            )
        return np.asarray([context[key] for key in self._context_keys])

    def array_to_action(self, x):
        try:
            assert len(x) == len(self._action_keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._action_keys))
            )
        return dict(zip(self._action_keys, x))

    def array_to_context(self, x):
        try:
            assert isinstance(x, float) or isinstance(x, int) or len(x) == len(self._context_keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self._context_keys))
            )
        key_val = list(self._context_keys)
        if len(key_val) == 1:
            return {key_val : x}
        return dict(zip(key_val, x))
    
    def register(self, context, action, reward, constraint_val):

        c = self.context_to_array(context)
        a = self.action_to_array(action)
        ca = np.concatenate([c.reshape(1, -1), a.reshape(1, -1)], axis=1)
        
        self._context = np.concatenate([self._context, c.reshape(1, -1)])
        self._action = np.concatenate([self._action, a.reshape(1, -1)])
        self._reward = np.concatenate([self._reward, [reward]])
        self._constraint = np.concatenate([self._constraint, constraint_val.reshape(1, -1)])
        self._context_action = np.concatenate([self._context_action, ca.reshape(1, -1)])

    def random_sample_from_S(self):
        S_idx = np.where(self._S)[0]
        rand_idx = np.random.randint(len(S_idx))
        return self._allActions[S_idx[rand_idx],:]

    def random_sample_from_S_syn(self, context):
        print('----------Assited generation of safe set!-------------')
        actions = self._allActions.reshape(1,-1)[0] * 4 - 2
        # print('ACTION_SPACE actions:\n{}'.format(actions))
        # m2 = context[0]*actions**2 - context[1]*actions
        # m3 = context[0]*(actions-context[2])**2 - context[1]*(actions-context[2]) 
        el = actions[0]
        vel = 1000
        grav = 9.81
        velht = vel * np.sin(el)
        veldist = vel * np.cos(el)
        t_impact = np.roots([0.5 * grav, velht, 0])[0]

        m2 = t_impact
        
        # costs = np.maximum(0, self.constraint_thres[0] - m2) + np.maximum(0, self.constraint_thres[1] - m3)
        costs = np.maximum(0, self.constraint_thres[0] - m2)
        best_point_idx = np.argmax(costs)
        best_point = self._allActions[best_point_idx]
        
        return best_point




    def generate_safeset(self, context, gp_list):
        
        self.empty_safeset = 0
        
        gp_list[0].fit(self.context_action, self.reward)
        for i in range(self.n_constraints):
            gp_list[i+1].fit(self.context_action, self.constraint_vals[:,i])
       
        context_action = np.concatenate([np.tile(context, (len(self._allActions), 1)), self._allActions], axis=1)        
        for i, gp in enumerate(gp_list):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, std = gp.predict(context_action, return_std=True)

            # self._mean[:,i] = mean#########REMOVE
            self._upper[:,i] = mean + self.beta_val* std
            self._lower[:,i] = mean - self.beta_val* std
        
        self._S = np.ones(len(self._allActions), dtype=bool)

        for i in range(self.n_constraints):
            if self.constr_greater[i] == 1:
                S_aux = self._lower[:,i+1] > self.constraint_thres[i]
            else:
                S_aux = self._upper[:,i+1] < self.constraint_thres[i] 
            self._S = np.logical_and(self._S, S_aux)
            
            
            
        # We always add the initial safeset to the safeset    
        # self._S[self._S0_idx] = True
        
        
        ############################
        # We add the initial safeset to the safeset when the computed safeset is empty
        if np.sum(self._S) == 0:
            self.empty_safeset = 1
            value = 0
            for i in range(self.n_constraints):
                if self.constr_greater[i] == 1:
                    value += np.maximum(0, self.constraint_thres[i] - self._lower[:,i+1] )
                else:
                    value += np.maximum(0, self._upper[:,i+1] - self.constraint_thres[i])
                    
                best_point_idx = np.argmin(value)
                self._S[best_point_idx] = True
                

      

    def suggest_ucb(self, context, gp_list):
        if self.minimization == 1:
            selected_idx = np.argmin(self._lower[self._S,0])
        else:
            selected_idx = np.argmax(self._upper[self._S,0])
        return self._allActions[self._S][selected_idx]


    def res(self):
        """Get all reward values found and corresponding parametes."""
        context = [dict(zip(self._context_keys, p)) for p in self.context]
        action = [dict(zip(self._action_keys, p)) for p in self.action]

        return [
            {"reward": r, "constraint_val": p, "action": a, "context": c}
            for r, p, a, c in zip(self.reward, self.constraint_vals, action, context)
        ]
