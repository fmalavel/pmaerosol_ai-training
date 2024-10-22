import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LogNorm
import iris
import iris.plot as iplt
from FLO_utils import extract_cube_area, add_bounds, __callback_add_forecast_day
from RH_wrt_water import RH_wrt_water_from_cubes


AQUM_PATH = "/scratch/fmalavel/mass_retrievals/u-df278/"
AQUM_files = "*202201*.pp"

# Thermodynamics Constraints
t_con = iris.AttributeConstraint(STASH='m01s00i004')
q_con = iris.AttributeConstraint(STASH='m01s00i010')
p_con = iris.AttributeConstraint(STASH='m01s00i408')

theta = iris.load_cube(AQUM_PATH + '/' + AQUM_files, 
                       t_con,
                       callback=__callback_add_forecast_day)

print(f"\n{theta}")

fc1_cons = iris.Constraint(forecast_day=1)
theta_fc1 = theta.extract(fc1_cons)
print(f"\n{theta_fc1}")
