# %% [markdown]
# ### Pyomo flow optim stuff

# %%
import scipy.optimize as spopt
import scipy as sp
import numpy as np
import math
import time
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as pyo


def get_dicts_from_numpy(nparray):

    if nparray.ndim == 1:
        array_dict = {(i+1):val for i,val in enumerate(nparray)}

    elif nparray.ndim == 2:
        array_dict = {(i+1,j+1):nparray[i,j] \
                      for j in range(nparray.shape[1])\
                      for i in range(nparray.shape[0])}

    else:
        return

    return array_dict

def init_model(max_flows,flow_cost=5,ens_cost=100):
    model = pyo.AbstractModel()
    
    model = init_transmission(model,max_flows)
    
    model.initial_residuals = pyo.Param(model.nodes)
    model.flows = pyo.Var(model.edges, domain=pyo.NonNegativeReals)
    model.unserved_energy = pyo.Var(model.nodes, domain=pyo.NonNegativeReals)

    model.obj = pyo.Objective(rule=o_rule, sense=pyo.minimize)
    model.nodal_residual_constraint = pyo.Constraint(model.nodes,rule=nodal_residual_cons_rule)
    model.flow_constraint = pyo.Constraint(model.edges,rule=flow_constraint_rule)

    model.flow_cost = pyo.Param(initialize=flow_cost)
    model.ens_cost = pyo.Param(initialize=ens_cost)
    
    return model

def init_transmission(model,max_flows):            
    all_regions = ['CAMX', 'Desert_Southwest', 'NWPP_Central', 'NWPP_NE', 'NWPP_NW']
    N = len(all_regions)
    nodes = np.arange(N)

    edges = np.arange(N*(N-1))

    edge_assoc = []
    for i in range(N):
        for j in range(N):
            if i!=j:
                edge_assoc.append((i,j))

    nodal_constraint = np.zeros((N,N*(N-1)))

    for i in range(N):
        for j in range(N*(N-1)):
            if i == edge_assoc[j][0]:
                nodal_constraint[i,j] = 1
            elif i == edge_assoc[j][1]:
                nodal_constraint[i,j] = -1

    model.nodes = pyo.RangeSet(N)
    model.edges = pyo.RangeSet(N*(N-1))

    model.max_flows = pyo.Param(model.edges, initialize=get_dicts_from_numpy(max_flows))
    model.nodal_constraint_matrix = pyo.Param(model.nodes, model.edges, 
                                              initialize=get_dicts_from_numpy(nodal_constraint))
    
    return model

def o_rule(model):
    return model.flow_cost*sum(model.flows[i] for i in model.edges) + model.ens_cost*sum(model.unserved_energy[j] for j in model.nodes) 

def nodal_residual_cons_rule(model,i):
    return (sum(model.nodal_constraint_matrix[i,j]*model.flows[j] for j in model.edges) 
            - model.unserved_energy[i] <= model.initial_residuals[i])
    
def flow_constraint_rule(model,i):
    return (model.flows[i] <= model.max_flows[i])

def create_instance_getoutput(model,row):
    instance = model.create_instance({None:{'initial_residuals':get_dicts_from_numpy(row)}})

    result = pyo.SolverFactory('glpk').solve(instance)

    flow = np.array([instance.flows[i]() for i in model.edges])
    ens = np.array([instance.unserved_energy[i]() for i in model.nodes])
    
    return flow,ens

