from msppy.evaluation import Evaluation, EvaluationTrue
from msppy.utils.statistics import (rand_int,check_random_state,
                                          compute_CI, allocate_jobs)
import pandas
import time
import numpy
import multiprocessing

class Evaluation_penalty(Evaluation):

    def run_single(self,
                 pv,
                 jobs,
                 random_state = None,
                 query = None,
                 query_dual = None,
                 query_stage_cost = False,
                 stage_cost = None,
                 solution = None,
                 solution_dual = None
            ):
        random_state = check_random_state(random_state)

        for j in jobs:
            sample_path_idx = (self.sample_path_idx[j]
                if self.sample_path_idx is not None else None)
            state = 0
            result = self.solver._forward(
                       random_state = random_state,
                       sample_path_idx = sample_path_idx,
                       solve_true = self.solve_true,
                       query = query,
                       query_dual = query_dual,
                       query_stage_cost= query_stage_cost,
                )
            if query is not None:
                for item in query:
                    for i in range(len(solution[item][0])):
                        solution[item][j][i] = result['solution'][item][i]
            if query_dual is not None:
                for item in solution_dual:
                    for i in range(len(solution_dual[item][0])):
                        solution_dual[item][j][i] = result['solution_dual'][item][i]
            if query_stage_cost:
                for i in range(len(stage_cost[0])):
                    stage_cost[j][i] = result['stage_cost'][i]
            pv[j] = result['pv']


            



    def run(
            self,
            n_simulations,
            percentile=95,
            query=None,
            query_T = None,
            query_dual=None,
            query_stage_cost=False,
            random_state=None,
            n_processes = 1,):
        """Run a Monte Carlo simulation to evaluate the policy on the
        approximation model.

        Parameters
        ----------
        n_simulations: int/-1
            If int: the number of simulations;
            If -1: exhuastive evaluation.

        query: list, optional (default=None)
            The names of variables that are intended to query.

        query_dual: list, optional (default=None)
            The names of constraints whose dual variables are intended to query.

        query_stage_cost: bool, optional (default=False)
            Whether to query values of individual stage costs.

        percentile: float, optional (default=95)
            The percentile used to compute the confidence interval.

        random_state: int, RandomState instance or None, optional
            (default=None)
            If int, random_state is the seed used by the random number
            generator;
            If RandomState instance, random_state is the random number
            generator;
            If None, the random number generator is the RandomState
            instance used by numpy.random.
        """
        from solver_penalty import SDDPPenalty, SDDPPenalty_infinity
        MSP = self.MSP
        query_T = query_T if query_T else MSP.T
        if not MSP._flag_infinity:
            self.solver = SDDPPenalty(MSP)
            stage = query_T
        else:
            self.solver = SDDPPenalty_infinity(MSP)
            self.solver.forward_T = query_T
            stage = MSP.T-1

        self.n_simulations = n_simulations
        random_state = check_random_state(random_state)
        query = [] if query is None else list(query)
        query_dual = [] if query_dual is None else list(query_dual)
        MSP = self.MSP
        if n_simulations == -1:
            self.n_sample_paths, self.sample_path_idx = MSP._enumerate_sample_paths(query_T-1)
        else:
            self.n_sample_paths = n_simulations
            self.sample_path_idx = None

        self.pv = numpy.zeros(self.n_sample_paths)
        stage_cost = solution = solution_dual = None
        if query_stage_cost:
            stage_cost = [
                multiprocessing.RawArray("d",[0] * (stage))
                for _ in range(self.n_sample_paths)
            ]
        if query is not None:
            solution = {
                item: [
                    multiprocessing.RawArray("d",[0] * (stage))
                    for _ in range(self.n_sample_paths)
                ]
                for item in query
            }
        if query_dual is not None:
            solution_dual = {
                item: [
                    multiprocessing.RawArray("d",[0] * (stage))
                    for _ in range(self.n_sample_paths)
                ]
                for item in query_dual
            }
        n_processes = min(self.n_sample_paths, n_processes)
        jobs = allocate_jobs(self.n_sample_paths, n_processes)
        pv = multiprocessing.Array("d", [0] * self.n_sample_paths)
        procs = [None] * n_processes
        for p in range(n_processes):
            procs[p] = multiprocessing.Process(
                target=self.run_single,
                args=(pv,jobs[p],random_state,query,query_dual,query_stage_cost,stage_cost,
                    solution,solution_dual)
            )
            procs[p].start()
        for proc in procs:
            proc.join()
        if self.n_simulations != 1:
            self.pv = [item for item in pv]
        else:
            self.pv = pv[0]
        if self.n_simulations == -1:
            self.epv = numpy.dot(
                pv,
                [
                    MSP._compute_weight_sample_path(self.sample_path_idx[j])
                    for j in range(self.n_sample_paths)
                ],
            )
        if self.n_simulations not in [-1,1]:
            self.CI = compute_CI(self.pv, percentile)
        self._compute_gap()
        if query is not None:
            self.solution = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution.items()
            }
        if query_dual is not None:
            self.solution_dual = {
                k: pandas.DataFrame(
                    numpy.array(v)
                ) for k, v in solution_dual.items()
            }
        if query_stage_cost:
            self.stage_cost = pandas.DataFrame(numpy.array(stage_cost))


        