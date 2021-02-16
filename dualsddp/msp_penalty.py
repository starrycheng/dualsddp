from msppy.msp import MSLP
from sp_penalty import StochasticModelPenalty
import numpy
from itertools import product

class MSLPPenalty(MSLP):

    def _set_up_model(self):
        self.models = [StochasticModelPenalty(name=str(t)) for t in range(self.T)]


    def _set_up_CTG(self):
        for t in range(self.T-1):
            # MC model may already do model copies
            M = (
                [self.models[t]]
                if type(self.models[t]) != list
                else self.models[t]
            )
            for m in M:
                if type(self.bound) == list:
                    if len(self.bound) < self.T-1:
                        raise Exception("initial dual bound does not match the number of stages!")
                    else:
                        m._set_up_CTG(discount=self.discount, bound=self.bound[t])
                else:
                    m._set_up_CTG(discount=self.discount, bound=self.bound)
                m.update()


    def _check_individual_stage_models(self):
        """Check state variables are set properly. Check stage-wise continuous
        uncertainties are discretized."""
        m = self.models[0] if type(self.models[0]) != list else self.models[0][0]
        if m.find_states == []:
            raise Exception("State variables must be set!")
        find_n_states = m.find_n_states
        for t in range(1, self.T):
            M = (
                self.models[t]
                if type(self.models[t]) == list
                else [self.models[t]]
            )
            for m in M:
                if m._type == "continuous":
                    if m._flag_discrete == 0:
                        raise Exception(
                            "Stage-wise independent continuous uncertainties "+
                            "must be discretized!"
                        )
                    self._individual_type = "discretized"
                else:
                    if m._flag_discrete == 1:
                        self._individual_type = "discretized"
                if m.find_n_states != find_n_states:
                    raise Exception(
                        "Dimension of state space must be same for all stages!"
                    )
        if self._type == "Markovian" and self._flag_discrete == 0:
            raise Exception(
                "Stage-wise dependent continuous uncertainties "+
                "must be discretized!"
            )
        self.find_n_states = [find_n_states] * self.T


    def _get_solution_sample_index(self, m, random_state):
        # sample one solution among scenarios 
        uniform_seed = random_state.random_sample()
        uniform_select = numpy.digitize(uniform_seed, m.cum_probability)
        uniform_select -= 1

        return uniform_select

    def _get_forward_solution(self, m, t, uniform_select=None):

        solution = [None for _ in m.find_states]
        # avoid numerical issues
        for idx, var in enumerate(m.find_states):
            if t > 0:
                var = var[uniform_select]
            elif t == 0:
                var = var[0]
            if var.vtype in ['B','I']:
                solution[idx] = int(round(var.X))
            else:
                if var.X < var.lb:
                    solution[idx] = var.lb
                elif var.X > var.ub:
                    solution[idx] = var.ub
                else:
                    solution[idx] = var.X

        return solution

    def _get_aggressive_forward_solution(self,m,t,last_state,minmax_flag):

        solution = [None for _ in m.find_states]
        for idx,var in enumerate(m.find_states):
            if t > 0:
                temp = numpy.array([abs(var_scen.X - last_state[idx]) for var_scen in var])
                if minmax_flag == 'min':
                    label = numpy.argmin(temp)
                if minmax_flag == 'max':
                    label = numpy.argmax(temp)
                if var[label].X < var[label].lb:
                    solution[idx] = var[label].lb
                elif var[label].X > var[label].ub:
                    solution[idx] = var[label].ub
                else:
                    solution[idx] = var[label].X
        m.scenario_index = label
        return solution,label

    def _get_stage_cost(self, m, t):
        if self.measure == "risk neutral":
            # the last stage model does not contain the cost-to-go function
            if t != self.T-1:
                return pow(self.discount,t) * (
                    m.objVal - self.discount*m.alpha.X
                )
            else:
                return pow(self.discount,t) * m.objVal
        else:
            return pow(self.discount,t) * m.getVarByName("stage_cost").X


    def _enumerate_sample_paths(self, T):
        """Enumerate all sample paths (three cases: pure stage-wise independent
        , pure Markovian, and mixed type)"""
        if self.n_Markov_states == 1:
            n_sample_paths = numpy.prod(
                [self.models[t].dual_n_samples for t in range(T + 1)]
            )
            sample_paths = list(
                product(
                    *[range(self.models[t].dual_n_samples) for t in range(T + 1)]
                )
            )
        
        return n_sample_paths, sample_paths



