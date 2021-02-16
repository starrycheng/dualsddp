from msppy.sp import StochasticModel
from msppy.utils.statistics import check_random_state
import gurobipy
import numpy
import math

class StochasticModelPenalty(StochasticModel):

    def __init__(self, name=""):
      super().__init__(name)
      self.dual_n_samples=0
      self.dual_probability=None
      self.cum_probability=None
      self.find_states=[]
      self.find_n_states=0
      self.expec_constrs=[]
      self.expec_past=[]
      self.expec_uncertainty_rhs=[]
      self.expec_uncertainty_past=[]
      self.phi = None
      self.p = 1.015 #penalty adjustment parameter

    
    def set_dual_probability(
           self, 
           t,
           n_samples=1,
           random_state=None,
           probability=None
          ):

      if t > 0:
        if self.dual_n_samples == 0:
          self.dual_n_samples = n_samples

        if self.dual_probability is None:
          if probability is None:
            probability = []
            for _ in range(n_samples):
              probability += [1/n_samples]
          self.dual_probability = list(probability)

        if self.cum_probability is None:
          self.cum_probability = []
          self.cum_probability += [0]
          self.cum_probability += list(numpy.cumsum(self.dual_probability)) 

      if t == 0:
        if self.dual_n_samples == 0:
          self.dual_n_samples = n_samples

        if self.dual_probability is None and n_samples == 1:
          self.dual_probability = [1]

        if self.dual_probability is None and n_samples > 1:
          if probability is None:
            probability = []
            for _ in range(n_samples):
              probability += [1/n_samples]
          self.dual_probability = list(probability)

    def addDualStateVar(   
            self,
            t,
            lb=0.0,    
            ub=gurobipy.GRB.INFINITY,
            obj=0.0,
            vtype=gurobipy.GRB.CONTINUOUS,
            uncertainty=None,
            random_state=None,
            name="",
    ):
      """
      Add state variable

      Parameters:
      ----------
      random_state: integer or numpy.random.RandomState instance
      """
      if t == 0:
        if random_state is None:
          state = self._model.addVar(
            lb=lb, ub=ub, obj=obj, 
            vtype=vtype, name=name
            )
          local_copy = self._model.addVar(
              name="{}_local_copy".format(name), lb=lb, ub=ub,
          )
          self._model.update()
          self.states += [state]
          self.local_copies += [local_copy]
          self.n_states += 1
          self.find_states += [[state]]
          self.find_n_states += 1

      if t > 0 or random_state is not None:
        
        random_state = check_random_state(random_state)
        
        if uncertainty is not None:
          uncertainty = self._check_uncertainty(uncertainty,0,1)
         
          if callable(uncertainty):
            samples = []
            probability = []
            for _ in range(self.dual_n_samples):
                samples.append(uncertainty(random_state))
                probability.append(1/self.dual_n_samples)
            self.dual_probability = probability
          else:
            samples = list(uncertainty)
      
        #set up obj values: with or without uncertainty
        if uncertainty is not None:
          obj = [
                  samples[i] * self.dual_probability[i] 
                  for i in range(self.dual_n_samples)
                ]
        if uncertainty is None and len(list([obj])) == 1:
          obj = [ 
                  obj * self.dual_probability[i]
                  for i in range(self.dual_n_samples)
          ]

        state = self._model.addVars(
            self.dual_n_samples, 
            lb=lb, ub=ub, obj=obj, 
            vtype=vtype, name=name
            )
        local_copy = self._model.addVar(
            name="{}_local_copy".format(name), lb=lb, ub=ub,
        )

        self._model.update()
        self.states += state.values()
        self.local_copies += [local_copy]
        self.n_states += self.dual_n_samples
        self.find_states += [state.values()]
        self.find_n_states += 1


      return state, local_copy

    def addDualVar(
                  self,
                  lb = 0.0,
                  ub=gurobipy.GRB.INFINITY,
                  obj=0.0,
                  vtype=gurobipy.GRB.CONTINUOUS,
                  uncertainty=None,
                  random_state=None,
                  name="",
                  ):

      random_state = check_random_state(random_state)
      #check uncertainty
      if uncertainty is not None:
        uncertainty = self._check_uncertainty(uncertainty,0,1)
        if callable(uncertainty):
          samples = []
          probability = []
          for _ in range(self.dual_n_samples):
              samples.append(uncertainty(random_state))
              probability.append(1/self.dual_n_samples)
          self.dual_probability = probability#force continous uncertainty to uniform discretization
        else:
          samples = list(uncertainty)
    
      #set up obj values: with or without uncertainty
      if uncertainty is not None:
        obj = [
                samples[i] * self.dual_probability[i] 
                for i in range(self.dual_n_samples)
              ]
      if uncertainty is None:
        obj = [
                obj * self.dual_probability[i]
                for i in range(self.dual_n_samples)
        ]

      var = self._model.addVars(
          self.dual_n_samples, 
          lb=lb, ub=ub, obj=obj, 
          vtype=vtype, name=name 
          )
      self._model.update()

      return var


    def addExpecConstr(
                      self,
                      past=None,
                      now=None,
                      var=None,
                      past_coefficient = 0.0,
                      now_coefficient = 0.0,
                      var_coefficient = 0.0,
                      rhs = 0.0,
                      plus_penalty_coefficient = 0.0,
                      minus_penalty_coefficient = 0.0,
                      uncertainty={'rhs':None,'past':None,'var':None,'now':None},
                      random_state=None,
                      sense=None,
                      name='',
                      constant_penalty = None,
                      p1 = 1.015,
                      p2 = 1.3
                      ):
      """
      Add constraint with expectations of variables

      Parameters
      ----------
      past: a list of var(s)
      past_coefficient: a list of coef(s) of past variable(s)
      uncertainty: dict(default=dict)
      """
      self.p1 = p1
      self.p2 = p2
      self.constant_penalty = constant_penalty

      random_state = check_random_state(random_state)
      #check uncertainty
      for key, value in uncertainty.items():
        if value is not None:
          value = self._check_uncertainty(value,0,1)
        if type(key) == str and key.lower() == 'rhs':
          if value is not None:
            samples = self._discretize_dual_uncertainty(
                            'rhs',
                            value,
                            random_state
                            )
          if value is None:
            samples = [rhs for _ in range(self.dual_n_samples)]
          self.expec_uncertainty_rhs += [samples]

        if type(key) == str and key.lower() == 'past':
          if value is not None:
            samples = self._discretize_dual_uncertainty(
                            'past',
                            value,
                            random_state
                            )
          if value is None:
            samples = [past_coefficient for _ in range(self.dual_n_samples)] 
          self.expec_uncertainty_past += [samples]

        if type(key) == str and key.lower() == 'now':
          if value is not None:
            now_coefficient = self._discretize_dual_uncertainty(
                            'now',
                            value,
                            random_state
                          )
          if value is None and now_coefficient is not None:
            now_coefficient = [
                  now_coefficient * self.dual_probability[i]
                  for i in range(self.dual_n_samples)
                 ]
        if type(key) == str and key.lower() == 'var':
          if value is not None:
            var_coefficient = self._discretize_dual_uncertainty(
                            'var',
                            value,
                            random_state
                          )
          if value is None and var_coefficient is not None:
            var_coefficient = [
                  var_coefficient * self.dual_probability[i]
                  for i in range(self.dual_n_samples)
                 ]
       
      
      if var is not None and now is not None:
        lhs = gurobipy.LinExpr(
                  gurobipy.quicksum([
                     now_coefficient[i] * now[i]
                   + var_coefficient[i] * var[i]
                     for i in range(self.dual_n_samples) 
                     ]
                    )
                  + gurobipy.quicksum([1 * past[i] for i in range(len(past))])
                  )
      if var is None and now is not None:
        lhs = gurobipy.LinExpr(
                  gurobipy.quicksum([
                     now_coefficient[i] * now[i]
                     for i in range(self.dual_n_samples) 
                     ]
                    )
                  + gurobipy.quicksum([1 * past[i] for i in range(len(past))])
                  )
      if var is not None and now is None:
        lhs = gurobipy.LinExpr(
                  gurobipy.quicksum([
                     var_coefficient[i] * var[i]
                     for i in range(self.dual_n_samples) 
                     ]
                    )
                  + gurobipy.quicksum([1 * past[i] for i in range(len(past))])
                  )
      #add penalty
      if plus_penalty_coefficient!= 0.0:
        plus_penalty = self._model.addVar(
            lb = 0.0, obj = -abs(plus_penalty_coefficient), 
            name='plus_penalty_'+name
          )
      if minus_penalty_coefficient != 0.0:
        minus_penalty = self._model.addVar(
          lb = 0.0, obj = -abs(minus_penalty_coefficient), 
          name='minus_penalty_'+name
          )
      
      if plus_penalty_coefficient != 0.0 and minus_penalty_coefficient != 0.0:
        lhs = gurobipy.LinExpr(lhs + plus_penalty - minus_penalty)
      elif plus_penalty_coefficient == 0.0 and minus_penalty_coefficient != 0.0:
        lhs = gurobipy.LinExpr(lhs - minus_penalty)
      elif plus_penalty_coefficient != 0.0 and minus_penalty_coefficient == 0.0:
        lhs = gurobipy.LinExpr(lhs + plus_penalty)
      else:
        lhs = gurobipy.LinExpr(lhs)

      
      constr = self._model.addConstr(
                  lhs=lhs, sense=sense, rhs=0.0, 
                  name = name
                  )
      self._model.update()

    
      self.expec_constrs += [constr]
      self.expec_past += [past]

      return constr



    def addScenConstr(
                      self,
                      now=None,
                      var=None,
                      now_coefficient=0.0,
                      var_coefficient=0.0,
                      rhs=0.0,
                      uncertainty_now=None,                      
                      uncertainty_var=None,
                      uncertainty_rhs=None,
                      random_state=None,
                      name='',
                      sense=None
                      ):
      #TODO
      now_coefficient = self._discretize_dual_uncertainty(
                            uncertainty_now,
                            now_coefficient,
                            random_state
                          )
      var_coefficient = self._discretize_dual_uncertainty(
                            uncertainty_var,
                            var_coefficient,
                            random_state
                          )

      rhs = self._discretize_dual_uncertainty(
                            uncertainty_rhs,
                            rhs,
                            random_state
                          )

      lhs = [None] * self.dual_n_samples
      if var is not None and now is not None:
        for i in range(self.dual_n_samples):
          lhs[i] = self._model.LinExpr(
                    var[i] * var_coefficient[i] + now[i] * now_coefficient[i]
                    )
      if var is None and now is not None:
        for i in range(self.dual_n_samples):
          lhs[i] = self._model.LinExpr(
                      now[i] * now_coefficient[i]
                      )
      if var is not None and now is None:
        for i in range(self.dual_n_samples):
          lhs[i] = self._model.LinExpr(
                    var[i] * var_coefficient[i])

      constrs = [None] * self.dual_n_samples
      for i in range(dual_n_samples):
        constrs[i] = self._model.addConstr(
                    lhs=lhs[i],rhs=rhs[i],sense=sense,name=name
                  )
      self._model.update()

      return constrs

    

    def _set_up_link_constrs(self):
      if self.link_constrs == []:
        self.link_constrs = list(
            self._model.addConstrs(
                (var == var.lb for var in self.local_copies),
                name = 'link_constrs',
              ).values()
          )
      

    def _update_link_constrs(self, fwdSoln):
        self._model.setAttr("RHS", self.link_constrs, fwdSoln)

    def _update_state_bound(self,t):
      if t != 0:
        return 
      if t == 0:
        find_states = [s[0] for s in self.find_states]
        self._model.setAttr('LB', find_states,
                            [-1e6 for _ in find_states]
                          )
        self._model.update()


    def _update_expec_constrs(self,uniform_select,cur_iteration,t):

      self._model.setAttr(
                     'RHS', self.expec_constrs, 
                     [
                      rhs_el[uniform_select] 
                      for rhs_el in self.expec_uncertainty_rhs
                     ]
                    )
      for i in range(len(self.expec_constrs)):
        ctr = self.expec_constrs[i]

        if self.constant_penalty:
            penalty_value = self.constant_penalty
        else:
            penalty_value = min(1e9,1e4*self.p2*(self.p1**t)**(cur_iteration))
            
        row_ctr = self._model.getRow(ctr)
        for l in range(row_ctr.size()):
          if row_ctr.getVar(l).VarName == 'plus_penalty':
            penalty_plus = row_ctr.getVar(l)
            self._model.setAttr('Obj', [penalty_plus], [-penalty_value])
          if row_ctr.getVar(l).VarName == 'minus_penalty':
            penalty_minus = row_ctr.getVar(l)
            self._model.setAttr('Obj', [penalty_minus], [-penalty_value])

        for j in range(len(self.expec_past[i])):
          var = self.expec_past[i][j]
          value = self.expec_uncertainty_past[i][uniform_select][j]
          self._model.chgCoeff(ctr, var, value)
      
    
    def _set_up_CTG(self, discount, bound):
      self.bound = bound
      # set up ctg: scenario-wise and aggregate
      if self.modelsense == 1:
        if self.phi is None:
          self.phi = [None] * self.dual_n_samples
          self.phi = self._model.addVars(
                  self.dual_n_samples,
                  lb = bound,
                  ub = gurobipy.GRB.INFINITY,
                  obj = 0.0,
                  name = 'phi'
          )
        if self.alpha is None:
          self.alpha = self._model.addVar(
                    lb=bound,
                    ub=gurobipy.GRB.INFINITY,
                    obj=discount,
                    name="alpha",
                )

      if self.modelsense == -1:
        if self.phi is None:
          self.phi = [None] * self.dual_n_samples
          self.phi = self._model.addVars(
                  self.dual_n_samples,
                  lb = -gurobipy.GRB.INFINITY,
                  ub = self.bound,
                  obj = 0.0,#self.dual_probability,
                  name = 'phi'
          )
        if self.alpha is None:
          self.alpha = self._model.addVar(
                    lb=-gurobipy.GRB.INFINITY,
                    ub=bound,
                    obj=discount,
                    name="alpha",
                  )

    def _add_cut_dual(self, rhs, gradient):
        if self.phi is None:
          self.phi = [None] * self.dual_n_samples
          self.phi = self._model.addVars(
                  self.dual_n_samples,
                  lb = self.bound,
                  ub = gurobipy.GRB.INFINITY,
                  obj = 0.0,
                  name = 'phi'
          )
        #Add #scenario cuts
        temp = [None] * self.dual_n_samples
        for j in range(self.dual_n_samples):
          temp[j] = gurobipy.LinExpr(
                                gradient, 
                                [
                                  find_states[j] 
                                  for find_states in self.find_states
                                ]
                              )
        # set up J self.alphas
        self.cuts.append(
            self._model.addConstrs(
                self.modelSense * (self.phi[j] - temp[j] - rhs) >= 0
                for j in range(self.dual_n_samples)
            )
        )
        self.cuts.append(
                self._model.addConstr(
                  self.alpha 
                   - gurobipy.quicksum(#model sense
                     [
                        self.dual_probability[i] * self.phi[i]
                        for i in range(self.dual_n_samples)
                      ]
                    )
                   <= 0
                )  
              )    
        self._model.update()


    def _remove_cut(self, cut_idx, sample_idx):
        self._model.remove(self.cuts[cut_idx][sample_idx])
        del self.cuts[cut_idx][sample_idx]
        self._model.update()



    def _solveDualLP(self):
      self.optimize()
      objDualLP = self.objVal
      gradDualLP = self.getAttr('Pi', self.link_constrs)
      return objDualLP, gradDualLP


    def _discretize_dual_uncertainty(self, type, uncertainty,random_state):
      if uncertainty is not None:
        if callable(uncertainty):
          samples = []
          for _ in range(self.dual_n_samples):
            samples.append(uncertainty(random_state))
        else:
          samples = list(uncertainty)  

        if (type == 'past' and numpy.array(samples).ndim > 1) or type == 'rhs':   
          coef_or_rhs = [
                samples[i] for i in range(self.dual_n_samples)
               ]

        if (type == 'past' and numpy.array(samples).ndim == 1) :   
          coef_or_rhs = [
                [samples[i]] for i in range(self.dual_n_samples)
               ]

        if type == 'now' or type == 'var':
          coef_or_rhs = [
                samples[i]* self.dual_probability[i] for i in range(self.dual_n_samples)
               ]

      return coef_or_rhs








    
