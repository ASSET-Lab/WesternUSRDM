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

def get_dicts_from_dataframe(dframe):
    names = dframe.columns.to_list()

    dframe_np = dframe.to_numpy().T
    param_set = {(i + 1, t + 1): dframe_np[i, t] \
                 for t in range(dframe_np.shape[1]) \
                 for i in range(dframe_np.shape[0]) \
                 }

    return names, param_set

def init_model(max_flows,time_steps,daylims_months,
               demand,
               RE_params,nonRE_nonHD_params,
               hydro_params,
               cost_params,co2_cap=None):
    
    model = pyo.ConcreteModel()
    
    model.time = pyo.RangeSet(time_steps[0], time_steps[1])
    model.months = pyo.RangeSet(1,time_steps[1]*12//365)
    model.daylims_months = pyo.Param(model.months, pyo.RangeSet(2), daylims_months)
    
    model = init_transmission(model,max_flows)

    model.flow_cost = pyo.Param(initialize=cost_params['flow_cost'])
    model.ens_cost = pyo.Param(initialize=cost_params['ens_cost'])

    model.RE_generation = pyo.Var(model.nodes, model.time, domain=pyo.NonNegativeReals)
    model.RE_generationcap = pyo.Param(model.nodes, model.time, initialize=RE_params['re_generation'])
    
    model.nonRE_nonHD_plants = pyo.RangeSet(1,nonRE_nonHD_params['num_nonre_nonHD_plants'])
    model.nonRE_nonHD_costs = pyo.Param(model.nonRE_nonHD_plants,initialize=nonRE_nonHD_params['nonre_nonHD_costs'])
    model.nonRE_nonHD_generationcap = pyo.Param(model.nonRE_nonHD_plants, model.time,
                                              initialize=nonRE_nonHD_params['nonre_nonHD_generation'])
    if co2_cap is not None:
        model.co2_cap = pyo.Param(initialize=co2_cap)
        model.nonRE_nonHD_co2emis = pyo.Param(model.nonRE_nonHD_plants,initialize=nonRE_nonHD_params['nonre_nonHD_co2emis'])

    model.nonRE_nonHD_generation = pyo.Var(model.nonRE_nonHD_plants, model.time, domain=pyo.NonNegativeReals)
    
    model.hydro_generation = pyo.Var(model.nodes, model.time, domain=pyo.NonNegativeReals)
    model.hydro_capacitycap = pyo.Param(model.nodes, initialize=hydro_params['hydro_capacitycap'])
    model.hydro_gen_cap = pyo.Param(model.nodes, model.months, initialize=hydro_params['hydro_gen_cap'])
    model.hydro_CF = pyo.Param(model.nodes, model.time, initialize=hydro_params['hydro_CF'])
    
    model.demand = pyo.Param(model.nodes, model.time, initialize=demand)
        
    model.flows = pyo.Var(model.edges, model.time,  domain=pyo.NonNegativeReals)
    model.unserved_energy = pyo.Var(model.nodes, model.time,  domain=pyo.NonNegativeReals)
        
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
                nodal_constraint[i,j] = -0.95

    model.nodes = pyo.RangeSet(N)
    model.edges = pyo.RangeSet(N*(N-1))

    model.max_flows = pyo.Param(model.edges, initialize=get_dicts_from_numpy(max_flows))
    model.nodal_constraint_matrix = pyo.Param(model.nodes, model.edges, 
                                              initialize=get_dicts_from_numpy(nodal_constraint))
    
    return model

def o_rule(model):
    
    return (sum(model.nonRE_nonHD_costs[i]*model.nonRE_nonHD_generation[i,t]\
                for i in model.nonRE_nonHD_plants for t in model.time)\
            + model.flow_cost*sum(model.flows[j,t] for j in model.edges for t in model.time)\
            + model.ens_cost*sum(model.unserved_energy[i,t] for i in model.nodes for t in model.time)
           ) 

def nodal_balance_constraint(model,nonRE_nonHD_plant_indexrange):
    
    model.residual_constraint = pyo.ConstraintList()
    
    for t in model.time:
        for i in model.nodes:
            nonRE_nonHD_plants_in_region = pyo.RangeSet(nonRE_nonHD_plant_indexrange[(i, 1)], 
                                                        nonRE_nonHD_plant_indexrange[(i, 2)])
            
            model.residual_constraint.add(
                   model.demand[i,t] - (model.RE_generation[i,t]
                                        + sum(model.nonRE_nonHD_generation[p,t] for p in nonRE_nonHD_plants_in_region)
                                        + model.hydro_generation[i,t]
                                        - sum(model.nodal_constraint_matrix[i,j]*model.flows[j,t] for j in model.edges)
                                       )
                   == model.unserved_energy[i,t]
            )
    
    return
    
def flow_constraint_rule(model):
    model.flow_constraint = pyo.ConstraintList()

    for t in model.time:
        for j in model.edges:
            model.flow_constraint.add(model.flows[j,t] <= model.max_flows[j])
        
    return

def set_hydro_constraints(model,daylims_months):
    model.hydro_constraint = pyo.ConstraintList()

    for t in model.time:
        for i in model.nodes:
            model.hydro_constraint.add(model.hydro_generation[i,t]
                                       <= model.hydro_capacitycap[i]*model.hydro_CF[i,t])

    for month in model.months:
        for i in model.nodes:
            days_in_month = pyo.RangeSet(daylims_months[(month, 1)], daylims_months[(month, 2)])
            expr_month_generation = sum(model.hydro_generation[i, t] for t in days_in_month)
            model.hydro_constraint.add(expr_month_generation <= model.hydro_gen_cap[i, month])

    return

def set_RE_constraints(model):
    
    model.RE_constraint = pyo.ConstraintList()
    
    for t in model.time:
        for i in model.nodes:
            model.RE_constraint.add(model.RE_generation[i,t] <= model.RE_generationcap[i,t])
    
    return

def set_nonRE_nonHD_constraints(model,co2_cap):
    
    model.nonRE_nonHD_constraint = pyo.ConstraintList()
    
    for t in model.time:
        for i in model.nonRE_nonHD_plants:
            model.nonRE_nonHD_constraint.add(model.nonRE_nonHD_generation[i,t] <= model.nonRE_nonHD_generationcap[i,t])
    
    if co2_cap is not None:
        model.co2_cap_constraint = pyo.ConstraintList()

        total_co2 = sum(sum(model.nonRE_nonHD_generation[i,t] for t in model.time)\
            *model.nonRE_nonHD_co2emis[i] for i in model.nonRE_nonHD_plants)
        total_co2 = total_co2*24

        model.co2_cap_constraint.add(total_co2<=model.co2_cap)

    return