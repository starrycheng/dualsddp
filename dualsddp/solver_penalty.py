from msppy.solver import SDDP
from msppy.utils.logger import LoggerSDDP,LoggerEvaluation,LoggerComparison
from msppy.utils.statistics import check_random_state,rand_int,compute_CI,allocate_jobs
from evaluation_penalty import Evaluation_penalty
import numpy
import time
import numpy
import multiprocessing
import gurobipy
import numbers
from collections import abc
import pandas

class SDDPPenalty(SDDP):
  
    def _add_and_store_pen_cuts(self, t, rhs, grad):
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            MSP.models[t-1]._add_cut_dual(rhs[0], grad[0])
            
        else:
            raise Exception('Do not consider stagewise-dependence')

    def _add_and_store_minimal_cuts(self, t, rhs, grad):
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            MSP.models[t-1]._add_minimal_cut_dual(rhs[0], grad[0])
            
        else:
            raise Exception('Do not consider stagewise-dependence')
    
    def _compute_time_idx(self,t):
        return t 


    def _select_trial_solution(self, random_state, forward_solution):
        return forward_solution, None

    def _add_cuts_additional_procedure(
        self, t, rhs, grad, cuts=None, cut_type=None,
        j=None
    ):
        pass

    def _regularize(self):
        if self.regularization_param == 0 or self.iteration == 0: return
        MSP = self.MSP
        for t in range(MSP.T):
            m = MSP.models[t]
            regularization = m.addVar(
                lb=0,
                obj=MSP.sense*self.regularization_param*0.99**self.iteration,
                name='regularization_{}'.format(self.iteration)
            )
            if self.regularization_type == 'L1':
                m.addConstrs(
                    (regularization >= gurobipy.quicksum(
                        m.find_states[i][j] *m.dual_probability[j] 
                        for j in range(m.dual_n_samples))
                        - self.forward_solution[t-1][i]
                    for i in range(m.find_n_states)),
                    name = 'regularization_{}'.format(self.iteration)
                )
            elif self.regularization_type == 'L2':
                m.addQConstr(
                    regularization -
                    gurobipy.QuadExpr(
                        gurobipy.quicksum([
                            gurobipy.quicksum(
                             m.find_states[i][j] *m.dual_probability[j] 
                             for j in range(m.dual_n_samples)) 
                            * gurobipy.quicksum(
                             m.find_states[i][j] *m.dual_probability[j] 
                             for j in range(m.dual_n_samples))
                            - gurobipy.quicksum(
                               m.find_states[i][j] *m.dual_probability[j] 
                               for j in range(m.dual_n_samples)) 
                              * 2 * self.forward_solution[t-1][i]
                            for i in range(m.find_n_states)
                        ])
                    )
                    >=0,
                    name = 'regularization_{}'.format(self.iteration)
                )
            else:
                raise NotImplementedError
            m.update()


    def _deregularize(self):
        if self.regularization_param == 0 or self.iteration == 0: return
        MSP = self.MSP
        for t in range(MSP.T):
            m = MSP.models[t]
            if self.regularization_type == 'L1':
                for i in range(m.find_n_states):
                    constr = m.getConstrByName(
                        'regularization_{}[{}]'.format(self.iteration-1,i))
                    if constr:
                        m.remove(constr)
            elif self.regularization_type == 'L2':
                constrs = m.getQConstrs()
                for constr in constrs:
                    m.remove(constr)
                var = m.getVarByName('regularization_{}'.format(self.iteration-1))
                if var:
                    var.obj = 0
            else:
                raise NotImplementedError
            m.update()

    def _SDDP_single(self):
        """A single serial SDDP step. Returns the policy value."""
        # random_state is constructed by number of iteration.
        random_state = numpy.random.RandomState(self.iteration)
        temp = self._forward(random_state)
        pv = temp['pv']
        forward_solution = temp['forward_solution']
        self._deregularize()
        self._backward(forward_solution)
        # regularization needs to store last forward_solution
        if self.regularization_param != 0:
            self.forward_solution = forward_solution
        if self.aggressive_search:
            self.forward_solution = forward_solution

        return [pv]

    def _forward(self, 
                 random_state,
                 sample_path_idx = None,
                 markovian_idx = None,
                 solve_true = False,
                 query = None,
                 query_dual = None,
                 query_stage_cost = None):
        """Single forward step. random_state generates random samples. Returns
        forward solution and corresponding policy value."""
        self.scenario_index = []
        MSP = self.MSP
        forward_solution = [None for _ in range(self.forward_T)]
        state = 0
        pv = 0
        query = [] if query is None else list(query)
        query_dual = [] if query_dual is None else list(query_dual)
        solution = {item: numpy.full(self.forward_T,numpy.nan) for item in query}
        solution_dual = {item: numpy.full(self.forward_T,numpy.nan) for item in query_dual}
        stage_cost = numpy.full(self.forward_T,numpy.nan)

        # time loop
        for t in range(self.forward_T):
            idx = self._compute_time_idx(t)

            if MSP._type == "stage-wise independent":
                m = MSP.models[idx]
            else:
                if t == 0:
                    m = MSP.models[idx][0]
                    state = 0
                else:
                    state = random_state.choice(
                        range(MSP.n_Markov_states[t]),
                        p=MSP.transition_matrix[t][state],
                    )
                m = MSP.models[idx][state]

            if t > 0:
                m._update_link_constrs(forward_solution[t - 1])
                if sample_path_idx is not None:
                    if MSP._type == "stage-wise independent":
                        uniform_select = sample_path_idx[t]
                    else:
                        raise NotImplementedError
                elif m._type == 'continuous' and solve_true:
                    raise NotImplementedError
                elif m._type == 'discrete' and m._flag_discrete == 1 and solve_true:
                    raise NotImplementedError
                m._update_expec_constrs(uniform_select, self.iteration,t)

            m.optimize()

            for var in m.getVars():
                if var.varName in query:
                    solution[var.varName][t] = var.X
            for constr in m.getConstrs():
                if constr.constrName in query_dual:
                    solution_dual[constr.constrName][t] = constr.Pi
            if query_stage_cost:
                stage_cost[t] = MSP._get_stage_cost(m, t)/pow(MSP.discount, t)
            
           
            if t == 0:
                uniform_select = 0
            else:
                uniform_select = MSP._get_solution_sample_index(m,random_state)
            forward_solution[t] = MSP._get_forward_solution(
                                        m,t,uniform_select=uniform_select
                                  )
            m.scenario_index = uniform_select
            pv += MSP._get_stage_cost(m, t)

            
        forward_solution,time_idx = self._select_trial_solution(
                                                               random_state, 
                                                               forward_solution
                                                            )                                                

        if query == [] and query_dual == [] and query_stage_cost is None:
            return {
                'forward_solution':forward_solution,
                'pv':pv
            }
        elif MSP._flag_infinity == 1:
            return {
                'solution':s_solution,
                'solution_dual':s_solution_dual,
                'stage_cost':s_stage_cost,
                'forward_solution':forward_solution,
                'pv':pv
            }
        else:
            return {
                'solution':solution,
                'solution_dual':solution_dual,
                'stage_cost':stage_cost,
                'forward_solution':forward_solution,
                'pv':pv
            }


    def _backward(self, forward_solution, j=None, lock=None, cuts=None):

        MSP = self.MSP
        for t in range(MSP.T-1, 0, -1):
            models = (
                [MSP.models[t]]
                if MSP.n_Markov_states == 1
                else MSP.models[t]
            )
            n_Markov_states = (
                1 if MSP.n_Markov_states == 1 else MSP.n_Markov_states[t]
            )
            objLP = [None] * n_Markov_states
            gradLP = [None] * n_Markov_states
            for k, m in enumerate(models):
                if MSP.n_Markov_states != 1:
                    raise NotImplementedError
                objDualLP, gradDualLP = m._solveDualLP()
                objLP[k], gradLP[k] = (
                    objDualLP,
                    gradDualLP,
                )
                objLP[k] -= numpy.matmul(gradLP[k], forward_solution[t-1])


            self._add_and_store_pen_cuts(t, objLP, gradLP) 
            self._add_cuts_additional_procedure(t,objLP, gradLP)
            
    
    def _remove_redundant_cut(self, clean_stages):
        for t in clean_stages:
            M = (
                [self.MSP.models[t]]
                if self.MSP.n_Markov_states == 1
                else self.MSP.models[t]
            )
            for m in M:
                m.update()
                for j in range(len(m.cuts)):
                    for key, cut in m.cuts[j].items():                       
                        if cut.sense == '>': cut.sense = '<'
                        elif cut.sense == '<': cut.sense = '>'
                        flag = 1
                        m.optimize()
                        if m.status == 4:
                            m.Params.DualReductions = 0
                            m.optimize()
                        if m.status not in [3,11]:
                            flag = 0
                        if flag == 1:
                            m._remove_cut(j,key)
                        else:
                            if cut.sense == '>': cut.sense = '<'
                            elif cut.sense == '<': cut.sense = '>'
                m.update()


    def solve(
            self,
            n_processes=1,
            n_steps=1,
            eval_n_processes = 1,
            max_iterations=10000,
            max_stable_iterations=10000,
            max_time=1000000.0,
            tol=0.001,
            freq_evaluations=None,
            percentile=95,
            tol_diff=float("-inf"),
            random_state=None,
            freq_evaluations_true=None,
            freq_comparisons=None,
            n_simulations=3000,
            n_simulations_true=3000,
            query=None,
            query_T=None,
            query_dual=None,
            query_stage_cost=False,
            query_policy_value=False,
            freq_clean=None,
            directory = None,
            regularization_type='L2',
            regularization_param=0,
            logFile=1,
            logToConsole=1,
            aggressive_search = False,
            aggressive_trial = False,
            warm_start = False,
            threshold = None,
            collect_CTG = False,
            seed = None):
        """Solve approximation model.

        Parameters
        ----------

        n_processes: int, optional (default=1)
            The number of processes to run in parallel. Run serial SDDP if 1.
            If n_steps is 1, n_processes is coerced to be 1.

        n_steps: int, optional (default=1)
            The number of forward/backward steps to run in each cut iteration.
            It is coerced to be 1 if n_processes is 1.

        max_iterations: int, optional (default=10000)
            The maximum number of iterations to run SDDP.

        max_stable_iterations: int, optional (default=10000)
            The maximum number of iterations to have same deterministic bound

        tol: float, optional (default=1e-3)
            tolerance for convergence of bounds

        freq_evaluations: int, optional (default=None)
            The frequency of evaluating gap on approximation model. It will be
            ignored if risk averse

        percentile: float, optional (default=95)
            The percentile used to compute confidence interval

        diff: float, optional (default=-inf)
            The stablization threshold

        freq_comparisons: int, optional (default=None)
            The frequency of comparisons of policies

        n_simulations: int, optional (default=10000)
            The number of simluations to run when evaluating a policy
            on approximation model

        freq_clean: int/list, optional (default=None)
            The frequency of removing redundant cuts.
            If int, perform cleaning at the same frequency for all stages.
            If list, perform cleaning at different frequency for each stage;
            must be of length T-1 (the last stage does not have any cuts).

        random_state: int, RandomState instance or None, optional (default=None)
            Used in evaluations and comparisons. (In the forward step, there is
            an internal random_state which is not supposed to be changed.)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used by numpy.random.

        logFile: binary, optional (default=1)
            Switch of logging to log file

        logToConsole: binary, optional (default=1)
            Switch of logging to console
        
        regularization is a heuristic that solves for the current states 
                       not far from the states in the last iteration

        regularization_param: the regularization term coefficient in the objective

        regularization_type: 'L2' or 'L1'

        collect_CTG: True if using trust bound strategy to restart 
        """
        MSP = self.MSP
        if freq_clean is not None:
            if isinstance(freq_clean, (numbers.Integral, numpy.integer)):
                freq_clean = [freq_clean] * (MSP.T-1)
            if isinstance(freq_clean, ((abc.Sequence, numpy.ndarray))):
                if len(freq_clean) != MSP.T-1:
                    raise ValueError("freq_clean list must be of length T-1!")
            else:
                raise TypeError("freq_clean must be int/list instead of {}!"
                .format(type(freq_clean)))
        if not MSP._flag_update:
            MSP._update()
        stable_iterations = 0
        total_time = 0
        a = time.time()
        gap = 1.0
        right_end_of_CI = float("inf")
        db_past = MSP.bound
        self.percentile = percentile
        self.regularization_type = regularization_type
        self.regularization_param = regularization_param
        if self.regularization_param != 0 and MSP._type != 'stage-wise independent':
            raise NotImplementedError
        self.collect_CTG = collect_CTG


        # collect optimal value of the CTG functions at each stage of each iteration
        if self.collect_CTG:
          self.optimalVal_CTG = numpy.zeros([max_iterations,(MSP.T)])

        pv_sim_past = None

        if n_processes != 1:
            self.n_steps = n_steps
            self.n_processes = min(n_steps, n_processes)
            self.jobs = allocate_jobs(self.n_steps, self.n_processes)

        logger_sddp = LoggerSDDP(
            logFile=logFile,
            logToConsole=logToConsole,
            n_processes=self.n_processes,
            percentile=self.percentile,
            directory=directory,
        )
        logger_sddp.header()
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation = LoggerEvaluation(
                n_simulations=n_simulations,
                percentile=percentile,
                logFile=logFile,
                logToConsole=logToConsole,
                directory=directory,
            )
            logger_evaluation.header()
        if freq_comparisons is not None:
            logger_comparison = LoggerComparison(
                n_simulations=n_simulations,
                percentile=percentile,
                logFile=logFile,
                logToConsole=logToConsole,
                directory=directory,
            )
            logger_comparison.header()
        try:
            while (
                self.iteration < max_iterations
                and total_time < max_time
                and stable_iterations < max_stable_iterations
                and tol < gap
                and tol_diff < right_end_of_CI
            ):
                start = time.time()

                self._compute_cut_type()

                if self.n_processes == 1:
                    pv = self._SDDP_single()
                else:
                    pv = self._SDDP_multiprocessesing()

                m = (
                    MSP.models[0]
                    if MSP.n_Markov_states == 1
                    else MSP.models[0][0]
                )
                m.optimize()

                # collect optimal value of the CTG functions at each stage of each iteration
                if self.collect_CTG:
                    for tau in range(MSP.T):
                        if tau == 0:
                            stage_model = MSP.models[tau]
                            self.optimalVal_CTG[self.iteration][tau] = stage_model.objBound
                        else:
                            stage_model = MSP.models[tau-1]
                            self.optimalVal_CTG[self.iteration][tau] = stage_model.alpha.X
               

                if m.status not in [2,11]:
                    m.write_infeasible_model(
                        "backward_" + str(m._model.modelName) + ".lp"
                    )
                db = m.objBound
                self.db.append(db)
                MSP.db = db
                if self.n_processes != 1:
                    CI = compute_CI(pv,percentile)
                self.pv.append(pv)
                self._regularize()

                if self.iteration >= 1:
                    if db_past == db:
                        stable_iterations += 1
                    else:
                        stable_iterations = 0

                self.iteration += 1
                db_past = db

                end = time.time()
                elapsed_time = end - start
                total_time += elapsed_time

                if self.n_processes == 1:
                    logger_sddp.text(
                        iteration=self.iteration,
                        db=db,
                        pv=pv[0],
                        time=elapsed_time,
                    )
                else:
                    logger_sddp.text(
                        iteration=self.iteration,
                        db=db,
                        CI=CI,
                        time=elapsed_time,
                    )
                if (
                    freq_evaluations is not None
                    and self.iteration%freq_evaluations == 0
                    or freq_comparisons is not None
                    and self.iteration%freq_comparisons == 0
                ):
                    start = time.time()
                    evaluation = Evaluation_penalty(MSP)
                    evaluation.run(
                        n_processes = eval_n_processes,
                        n_simulations=n_simulations,
                        random_state=random_state,
                        query_stage_cost=False,
                        percentile=percentile,
                    )
                    directory = '' if directory is None else directory
                    if query_policy_value:
                        pandas.DataFrame({'pv':evaluation.pv}).to_csv(directory+
                            "iter{}_dual_pv.csv".format(self.iteration))
                    if query is not None:
                        for item in query:
                            evaluation.solution[item].to_csv(directory+
                                "iter{}_dual_{}.csv".format(self.iteration, item))
                    if query_dual is not None:
                        for item in query_dual:
                            evaluation.solution_dual[item].to_csv(directory+
                                "iter{}_dual_{}.csv".format(self.iteration, item))
                    if query_stage_cost:
                        evaluation.stage_cost.to_csv(directory+
                            "iter{}_dual_stage_cost.csv".format(self.iteration))

                    elapsed_time = time.time() - start
                    gap = evaluation.gap
                    if n_simulations == -1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            pv=evaluation.epv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    elif n_simulations == 1:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            pv=evaluation.pv,
                            gap=gap,
                            time=elapsed_time,
                        )
                    else:
                        logger_evaluation.text(
                            iteration=self.iteration,
                            db=db,
                            CI=evaluation.CI,
                            gap=gap,
                            time=elapsed_time,
                        )
                if (
                    freq_comparisons is not None
                    and self.iteration%freq_comparisons == 0
                ):
                    start = time.time()
                    pv_sim = evaluation.pv
                    if self.iteration / freq_comparisons >= 2:
                        diff = MSP.sense*(numpy.array(pv_sim_past)-numpy.array(pv_sim))
                        if n_simulations == -1:
                            diff_mean = numpy.mean(diff)
                            right_end_of_CI = diff_mean
                        else:
                            diff_CI = compute_CI(diff, self.percentile)
                            right_end_of_CI = diff_CI[1]
                        elapsed_time = time.time() - start
                        if n_simulations == -1:
                            logger_comparison.text(
                                iteration=self.iteration,
                                ref_iteration=self.iteration-freq_comparisons,
                                diff=diff_mean,
                                time=elapsed_time,
                            )
                        else:
                            logger_comparison.text(
                                iteration=self.iteration,
                                ref_iteration=self.iteration-freq_comparisons,
                                diff_CI=diff_CI,
                                time=elapsed_time,
                            )
                    pv_sim_past = pv_sim
                if freq_clean is not None:
                    clean_stages = [
                        t
                        for t in range(1,MSP.T-1)
                        if self.iteration%freq_clean[t] == 0
                    ]
                    if len(clean_stages) != 0:
                        self._remove_redundant_cut(clean_stages)
                # self._clean()
        except KeyboardInterrupt:
            stop_reason = "interruption by the user"
        # SDDP iteration stops
        MSP.db = self.db[-1]
        if self.iteration >= max_iterations:
            stop_reason = "iteration:{} has reached".format(max_iterations)
        if total_time >= max_time:
            stop_reason = "time:{} has reached".format(max_time)
        if stable_iterations >= max_stable_iterations:
            stop_reason = "stable iteration:{} has reached".format(max_stable_iterations)
        if gap <= tol:
            stop_reason = "convergence tolerance:{} has reached".format(tol)
        if right_end_of_CI <= tol_diff:
            stop_reason = "stablization threshold:{} has reached".format(tol_diff)

        b = time.time()
        logger_sddp.footer(reason=stop_reason)
        if freq_evaluations is not None or freq_comparisons is not None:
            logger_evaluation.footer()
        if freq_comparisons is not None:
            logger_comparison.footer()
        self.total_time = total_time

        # collect optimal value of the CTG functions at each stage of each iteration
        if self.collect_CTG:
          pandas.DataFrame(self.optimalVal_CTG).to_csv('restart_alpha{}.csv'.format(seed))
           
class SDDPPenalty_infinity(SDDPPenalty):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_T = self.MSP.T
        self.cut_T = self.MSP.T
        self.period = self.MSP.T-1


    def _compute_time_idx(self,t):
        return t%self.period if (t%self.period != 0 or t == 0) else -1


    def _select_trial_solution(self, random_state, forward_solution):
        # if solving more than one signle period, only part of obtained solutions 
        # would be selected
        if self.forward_T > self.period + 1:
            indices = numpy.arange(0, self.forward_T, self.period)
            idx = indices[rand_int(k=len(indices), random_state=random_state)]
            if idx + self.period > self.forward_T:
                idx = idx - self.period
            for t in range(1, self.period+1):
                self.MSP.models[t]._update_link_constrs(forward_solution[idx+t-1])
            # self.period+1: is to update the second stage with the (m+1)^th stage solution
            time_idx = numpy.arange(idx,idx+self.period,1)
            new_forward_solution = forward_solution[idx:idx+self.period]
            return new_forward_solution, time_idx

        return forward_solution


    
    def _add_cuts_additional_procedure(
        self, t, rhs, grad, cuts=None, cut_type=None,
        j=None
    ):
        if t != 1: return
        MSP = self.MSP
        if MSP.n_Markov_states == 1:
            # add (m+2)^th stage cuts on the (m+1)^th model(the last stage model)
            MSP.models[-1]._add_cut_dual(rhs[0], grad[0]) 
            if cuts is not None:
                cuts[MSP.T-1][cut_type][j][:] = numpy.append(rhs, grad)

        else:
            raise NotImplementedError



    def solve(self, forward_T=None, *args, **kwargs):
        """Solve approximation model.

        Parameters
        ----------
        forward_T: int, optional 
            The number of stages to consider in forward passes and trial point 
            selection
        """
        if self.MSP._type == "Markov chain":
            raise NotImplementedError
        if forward_T: self.forward_T = forward_T
        if type(self.MSP.bound) == list:
            if self.MSP.T <= 2:
                self.MSP[-1]._set_up_CTG(discount=self.MSP.discount, 
                                      bound=self.MSP.bound[0])
            else:
                self.MSP[-1]._set_up_CTG(discount=self.MSP.discount, 
                                      bound=self.MSP.bound[1])
        else:
            self.MSP[-1]._set_up_CTG(discount=self.MSP.discount, 
                                     bound=self.MSP.bound)
        self.MSP._flag_infinity = 1
        super().solve(*args, **kwargs)

