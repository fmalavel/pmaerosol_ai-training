# %% [markdown]
# ## Importing stuff

# %%
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
from FLO_utils import extract_cube_area, add_bounds, add_forecast_day
from RH_wrt_water import RH_wrt_water_from_cubes


# %% [markdown]
# ## Loading Data

# %%
AQUM_PATH = "/scratch/fmalavel/mass_retrievals/u-df278/tmp"
AQUM_files = "prods_op_aqum_20190921_18.012.pp"

# Thermodynamics Constraints
t_con = iris.AttributeConstraint(STASH='m01s00i004')
q_con = iris.AttributeConstraint(STASH='m01s00i010')
p_con = iris.AttributeConstraint(STASH='m01s00i408')

# PM Constraints
pm10wet_con = iris.AttributeConstraint(STASH='m01s38i560')
pm2p5wet_con = iris.AttributeConstraint(STASH='m01s38i561')
pm10dry_con = iris.AttributeConstraint(STASH='m01s38i562')
pm2p5dry_con = iris.AttributeConstraint(STASH='m01s38i563')

# GLOMAP MMR Constraints
ait_ins_bc_con = iris.AttributeConstraint(STASH='m01s34i120')
ait_ins_om_con = iris.AttributeConstraint(STASH='m01s34i121')
ait_sol_so4_con = iris.AttributeConstraint(STASH='m01s34i104')
ait_sol_bc_con = iris.AttributeConstraint(STASH='m01s34i105')
ait_sol_om_con = iris.AttributeConstraint(STASH='m01s34i106')
ait_sol_no3_con = iris.AttributeConstraint(STASH='m01s34i137')
ait_sol_nh4_con = iris.AttributeConstraint(STASH='m01s34i133')
acc_sol_so4_con = iris.AttributeConstraint(STASH='m01s34i108')
acc_sol_bc_con = iris.AttributeConstraint(STASH='m01s34i109')
acc_sol_om_con = iris.AttributeConstraint(STASH='m01s34i110')
acc_sol_ss_con = iris.AttributeConstraint(STASH='m01s34i111')
acc_sol_no3_con = iris.AttributeConstraint(STASH='m01s34i138')
acc_sol_nh4_con = iris.AttributeConstraint(STASH='m01s34i134')
coa_sol_so4_con = iris.AttributeConstraint(STASH='m01s34i114')
coa_sol_bc_con = iris.AttributeConstraint(STASH='m01s34i115')
coa_sol_om_con = iris.AttributeConstraint(STASH='m01s34i116')
coa_sol_ss_con = iris.AttributeConstraint(STASH='m01s34i117')
coa_sol_no3_con = iris.AttributeConstraint(STASH='m01s34i139')
coa_sol_nh4_con = iris.AttributeConstraint(STASH='m01s34i135')

# GLOMAP Number Constraints
ait_sol_number_con = iris.AttributeConstraint(STASH='m01s34i103')
acc_sol_number_con = iris.AttributeConstraint(STASH='m01s34i107')
coa_sol_number_con = iris.AttributeConstraint(STASH='m01s34i113')
ait_ins_number_con = iris.AttributeConstraint(STASH='m01s34i119')

# Loading Data

# Thermodynamics
theta = iris.load_cube(AQUM_PATH + '/' + AQUM_files, t_con)
p = iris.load_cube(AQUM_PATH + '/' + AQUM_files, p_con)
q = iris.load_cube(AQUM_PATH + '/' + AQUM_files, q_con)

# PM
pm10wet = iris.load_cube(AQUM_PATH + '/' + AQUM_files, pm10wet_con)
pm10dry = iris.load_cube(AQUM_PATH + '/' + AQUM_files, pm10dry_con)
pm2p5wet = iris.load_cube(AQUM_PATH + '/' + AQUM_files, pm2p5wet_con)
pm2p5dry = iris.load_cube(AQUM_PATH + '/' + AQUM_files, pm2p5dry_con)

# Mode MMRs
ait_ins_bc = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_ins_bc_con)
ait_ins_om = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_ins_om_con)
ait_sol_so4 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_sol_so4_con)
ait_sol_bc = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_sol_bc_con)
ait_sol_om = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_sol_om_con)
ait_sol_no3 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_sol_no3_con)
ait_sol_nh4 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_sol_nh4_con)
acc_sol_so4 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_so4_con)
acc_sol_bc = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_bc_con)
acc_sol_om = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_om_con)
acc_sol_ss = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_ss_con)
acc_sol_no3 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_no3_con)
acc_sol_nh4 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_nh4_con)
coa_sol_so4 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_so4_con)
coa_sol_bc = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_bc_con)
coa_sol_om = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_om_con)
coa_sol_ss = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_ss_con)
coa_sol_no3 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_no3_con)
coa_sol_nh4 = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_nh4_con)

# Mode Numbers
ait_sol_number = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_sol_number_con)
acc_sol_number = iris.load_cube(AQUM_PATH + '/' + AQUM_files, acc_sol_number_con)
coa_sol_number = iris.load_cube(AQUM_PATH + '/' + AQUM_files, coa_sol_number_con)
ait_ins_number = iris.load_cube(AQUM_PATH + '/' + AQUM_files, ait_ins_number_con)

add_forecast_day(theta)


# %% [markdown]
# ## Process cubes

# %%
# Calculate rh
rhw = RH_wrt_water_from_cubes(theta=theta, p=p, q=q)
print(rhw)

# Combine Mode MMRs into one cube
modes_ait = pm2p5dry.copy()
modes_acc = pm2p5dry.copy()
modes_coa = pm2p5dry.copy()
modes_all = pm2p5dry.copy()

modes_ait.data = modes_ait.data * 0.0
modes_acc.data = modes_acc.data * 0.0
modes_coa.data = modes_coa.data * 0.0
modes_all.data = modes_all.data * 0.0

rho_air = 1.25  # assumes pho_air is 1.25 kg/m3
kgkg_to_ugm3 = rho_air * 1e9     # convert mmr to concentration

modes_ait.data = \
        (ait_ins_bc.data + ait_ins_om.data + \
        ait_sol_so4.data + ait_sol_bc.data + \
        ait_sol_om.data + ait_sol_no3.data + \
        ait_sol_nh4.data) * kgkg_to_ugm3

modes_acc.data = \
        (acc_sol_so4.data + acc_sol_bc.data + \
        acc_sol_om.data + acc_sol_ss.data + \
        acc_sol_no3.data + acc_sol_nh4.data) * kgkg_to_ugm3

modes_coa.data = \
        (coa_sol_so4.data + coa_sol_bc.data + \
        coa_sol_om.data + coa_sol_ss.data + \
        coa_sol_no3.data + coa_sol_nh4.data) * kgkg_to_ugm3

modes_all.data = \
        (ait_ins_bc.data + ait_ins_om.data + \
        ait_sol_so4.data + ait_sol_bc.data + \
        ait_sol_om.data + ait_sol_no3.data + \
        ait_sol_nh4.data + \
        acc_sol_so4.data + acc_sol_bc.data + \
        acc_sol_om.data + acc_sol_ss.data + \
        acc_sol_no3.data + acc_sol_nh4.data + \
        coa_sol_so4.data + coa_sol_bc.data + \
        coa_sol_om.data + coa_sol_ss.data + \
        coa_sol_no3.data + coa_sol_nh4.data) * kgkg_to_ugm3

#add bounds and remove domain edges
add_bounds(rhw)
add_bounds(modes_ait)
add_bounds(modes_acc)
add_bounds(modes_coa)
add_bounds(modes_all)
add_bounds(pm10wet)
add_bounds(pm10dry)
add_bounds(pm2p5wet)
add_bounds(pm2p5dry)
add_bounds(ait_sol_number)
add_bounds(acc_sol_number)
add_bounds(coa_sol_number)
add_bounds(ait_ins_number)

domain_dict = {
    "MINLON":-6,
    "MAXLON":2.28,
    "MINLAT":49.8,
    "MAXLAT":54.2,
}

extract_rhw = extract_cube_area(rhw,
                                MINLON=domain_dict["MINLON"],
                                MAXLON=domain_dict["MAXLON"],
                                MINLAT=domain_dict["MINLAT"],
                                MAXLAT=domain_dict["MAXLAT"],
                               )
extract_pm2p5dry = extract_cube_area(pm2p5dry,
                                     MINLON=domain_dict["MINLON"],
                                     MAXLON=domain_dict["MAXLON"],
                                     MINLAT=domain_dict["MINLAT"],
                                     MAXLAT=domain_dict["MAXLAT"],
                                    )
extract_pm2p5wet = extract_cube_area(pm2p5wet,
                                     MINLON=domain_dict["MINLON"],
                                     MAXLON=domain_dict["MAXLON"],
                                     MINLAT=domain_dict["MINLAT"],
                                     MAXLAT=domain_dict["MAXLAT"],
                                    )
extract_pm10dry = extract_cube_area(pm10dry,
                                     MINLON=domain_dict["MINLON"],
                                     MAXLON=domain_dict["MAXLON"],
                                     MINLAT=domain_dict["MINLAT"],
                                     MAXLAT=domain_dict["MAXLAT"],
                                    )
extract_pm10wet = extract_cube_area(pm10wet,
                                    MINLON=domain_dict["MINLON"],
                                    MAXLON=domain_dict["MAXLON"],
                                    MINLAT=domain_dict["MINLAT"],
                                    MAXLAT=domain_dict["MAXLAT"],
                                   )
extract_modes_ait = extract_cube_area(modes_ait,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_modes_acc = extract_cube_area(modes_acc,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_modes_coa = extract_cube_area(modes_coa,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_modes_all = extract_cube_area(modes_all,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_ait_sol_number = extract_cube_area(ait_sol_number,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_acc_sol_number = extract_cube_area(acc_sol_number,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_coa_sol_number = extract_cube_area(coa_sol_number,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )
extract_ait_ins_number = extract_cube_area(ait_ins_number,
                                      MINLON=domain_dict["MINLON"],
                                      MAXLON=domain_dict["MAXLON"],
                                      MINLAT=domain_dict["MINLAT"],
                                      MAXLAT=domain_dict["MAXLAT"],
                                     )

print(extract_pm2p5dry)


# %% [markdown]
# ## Initiate NN model

# %%
# Define the model
class PM_Net(nn.Module):
    def __init__(self):
        super(PM_Net, self).__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialise model
model = PM_Net()

# Loading the model
model.load_state_dict(torch.load("PM2p5_model.NN"))#, weights_only=True))
model.eval()



# %% [markdown]
# ## Loading Trained PM Model

# %%
NN_pm2p5dry = extract_pm2p5dry.copy()
NN_pm2p5wet = extract_pm2p5wet.copy()
NN_pm10dry = extract_pm10dry.copy()
NN_pm10wet = extract_pm10wet.copy()

dt, dx, dy = NN_pm2p5dry.data.shape
print(f"{NN_pm2p5dry.data.shape}")

for i in range(dt):
    for j in range(dx):
        for k in range(dy):
            # Define the features/predictors to be used:
            x1_1D = extract_modes_ait.data[i,j,k].reshape(-1, 1)
            x2_1D = extract_modes_acc.data[i,j,k].reshape(-1, 1)
            x3_1D = extract_modes_coa.data[i,j,k].reshape(-1, 1)
            x4_1D = extract_rhw.data[i,j,k].reshape(-1, 1)
            x5_1D = extract_ait_ins_number.data[i,j,k].reshape(-1, 1)
            x6_1D = extract_coa_sol_number.data[i,j,k].reshape(-1, 1)
            x7_1D = extract_acc_sol_number.data[i,j,k].reshape(-1, 1)
            x8_1D = extract_ait_sol_number.data[i,j,k].reshape(-1, 1)
            x_1D = np.hstack((x1_1D, x2_1D, x3_1D, x4_1D, x5_1D, x6_1D, x7_1D, x8_1D))
            x_array = torch.tensor(x_1D, dtype=torch.float32).unsqueeze(1)
            #print(f"x_array = {x_array}")
            
            with torch.no_grad():
                NN_pm2p5dry.data[i,j,k] = model(x_array).detach().numpy()
                NN_pm2p5wet.data[i,j,k] = model(x_array).detach().numpy()
                NN_pm10dry.data[i,j,k] = model(x_array).detach().numpy()
                NN_pm10wet.data[i,j,k] = model(x_array).detach().numpy()
                #print(f"{NN_pm2p5dry.data[i,j,k]} / {extract_pm2p5dry.data[i,j,k]}")


# %% [markdown]
# ## Plot AQUM PM diag vs NN estimates

# %%
fig = plt.figure(figsize=(12,4), dpi=100)

pad = 0.05
sh = 0.95
asp = 25

cmap = plt.get_cmap("Spectral")
cmap_r = cmap.reversed()
cmap = cmap_r

# PM Neural Network
#   Fig 1   #
plt.subplot(1, 2, 1)

cb_min=0
cb_max=25
levels = np.linspace(cb_min, cb_max, 21)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
tick_lvl = levels[0::5]
cb_label_str = 'ug/m3'

iplt.contourf(NN_pm2p5dry.collapsed(['time'],iris.analysis.MEAN),
              levels=levels,
              cmap=cmap, 
              norm=norm,
              extend='both')

plt.colorbar(orientation='vertical',
             ticks=tick_lvl,
             # format="%.1e",
             pad=pad,
             aspect=asp,
             shrink=sh).set_label(cb_label_str)

plt.gca().coastlines(resolution='10m')
plt.title("Neural Network pm2p5 dry", fontsize=10, linespacing=1.2)


# PM MODEL DIAG
#   Fig 2   #
plt.subplot(1, 2, 2)

cb_min=0
cb_max=25
levels = np.linspace(cb_min, cb_max, 21)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
tick_lvl = levels[0::5]
cb_label_str = 'ug/m3'

iplt.contourf(extract_pm2p5dry.collapsed(['time'],iris.analysis.MEAN),
              levels=levels,
              cmap=cmap, 
              norm=norm,
              extend='both')

plt.colorbar(orientation='vertical',
             ticks=tick_lvl,
             # format="%.1e",
             pad=pad,
             aspect=asp,
             shrink=sh).set_label(cb_label_str)

plt.gca().coastlines(resolution='10m')
plt.title("AQUM pm2p5 dry", fontsize=10, linespacing=1.2)



# %%
