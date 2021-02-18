import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel, PairwiseKernel)
from sklearn.metrics import mean_squared_error
import scipy.stats
from scipy.misc import derivative
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

############################################################################################################################

from kalman_filter import Kalman_filter
from pf import PF

############################################################################################################################

# target visual POD wrt camera distance  
def p_D(d):
    d_s = 2
    beta_s = 5
    beta_l = 5
    d_l = 3.5
    d_opt = 3
    p_D = (  ( 1/(1+np.exp(beta_l*(d-d_l))) ) ) # ( 1/(1+np.exp(-beta_s*(d-d_s))) ) * 
    return p_D

# log-distance path-loss model 
def PLM(r0,n,d):
    return r0 - 10*n*np.log10(d)

# inverse log-distance path-loss model 
def PLM_inverse(r0,n,r):
    return 10**( (r0-r)/(10*n) ) 

# function to predict the POD, given the GP model and a RSSI value 
def f(x,model):
    return model.predict(np.array([x]).reshape(-1,1))[0][0]

# 
def A(d):
    A0= 1 # normalized area at closest distance
    k = 0.1 # area attenuation gain
    return A0 - k*d


############################################################################################################################

# CONTROL CAMERA DIRECTION AND ZOOM VIA RSSI VALUES

                    # *** SETUP PARAMETERS ***
# target-camera distance
d_min = 0.5 # min target-camera distance
d_max = 5.5 # max target-camera distance

# synthetic RSSI generation according to the PLM
r0 = -35 # RSSI offset at d0 = 1 [m]
n = 2 # RSSI attenuation gain
sigma = 3 # RSSI noise std. dev.

N_train = int(9*1.0E+2) # train dataset cardinality
N_test = int(5*1.0E+2) # test duration

# RSSI tracking via PF
N_s = 100 # number of particles
mu_omega = 0.0 # mean of the process model
sigma_omega = 0.1 # std. dev. of the process model 
pf_train = PF(N_s,\
            init_interval=[PLM(r0,n,d_max),PLM(r0,n,d_min)],\
            draw_particles_flag=False)

# Kalman filter to smooth POD measurements
x0 = 0.5 # initial guess
kf = Kalman_filter(1.0,0.0,1.0,0.1,0.1,x0,0.1) # A, B, C, Q, R, P

# devices specifications
T_RF = 0.1 # [s] Rx sampling time
fps = 10 # camera fps

                    # *** GP TRAINING ***
# train dataset generation
distance_train = np.empty((N_train,1))
A_true = np.empty((N_train,1)) 
A_meas = np.empty((N_train,1))
rssi_raw = np.empty((N_train,1))
rssi_smooth = np.empty((N_train,1))
d=d_max
for t in range(N_train):
    # true distance
    d = max(min(d+np.random.normal(-0.01,0.2),d_max),d_min)
    distance_train[t] = d
    # true POD (no RSSI noise, groundtruth knowledge on the POD function) --> groundtruth
    A_true[t] = A(d)

    # 
    A_meas[t] = A_true[t] +  np.random.normal(0,1.0E-5)
    
    # RSSI predict
    pf_train.predict(mu_omega,sigma_omega)
    # RSSI sample
    rssi_raw[t] = PLM(r0,n,d) + np.random.normal(0,sigma)  
    # RSSI update
    pf_train.update(rssi_raw[t],sigma)
    # resampling
    pf_train.SIS_resampling()
    # compute MMSE estimate
    pf_train.estimation(est_type='MMSE')
    rssi_smooth[t] = pf_train.estimate
    
# plot 
fig = plt.figure(figsize=(9,6))
plt.subplot(4,1,1)
plt.plot(np.linspace(d_min,d_max,200), [A(d) for d in np.linspace(d_min,d_max,200)], \
    label=r'$p_D(d)$', linewidth=2)
plt.legend()
plt.xlabel(r"$d\;[m]$")
plt.subplot(4,1,2)
plt.plot(distance_train, label=r'$d_t$')
plt.xlabel(r"$t;[ms]$")
plt.ylabel(r"distance [m]")
plt.subplot(4,1,3)
plt.plot(rssi_raw,label=r"$\tilde{r}_t$") # raw RSSI
plt.plot(rssi_smooth,label=r"$\hat{r}_t^{PF}$") # smoothed RSSI
plt.legend()
plt.ylabel(r"RSSI $[dBm]$")
plt.xlabel(r"$t\;[ms]$")
plt.subplot(4,1,4)
plt.plot(A_true,label=r'$p_{D,t}$') 
plt.plot(A_meas,label=r'$\tilde{p}_{D,t}$') 
plt.xlabel(r"$t\;[ms]$")
plt.legend()
plt.tight_layout()
plt.show()

# define GP model
model_gp = GaussianProcessRegressor(Matern()+WhiteKernel(noise_level_bounds=(0.0,0.2)))
# fit the GP model
model_gp.fit(A_meas,rssi_smooth)
idx = np.argsort(A_meas,axis=0)
rssi_est_gp,std_gp = model_gp.predict(A_meas[idx,0],return_std=True)

# plot training results
# pD_ideal = [p_D(PLM_inverse(r0,n,r)) for r in rssi_train[idx,0]]
fig = plt.figure(figsize=(9,6))
# plt.plot(rssi_train[idx,0], pD_ideal, label=r'$p_D(r)$',linewidth=2) # true RSSI-POD function
plt.plot(A_meas[idx,0], rssi_est_gp,c='g',label='$\widehat{p}_D(r)$',linewidth=2) # estimated RSSI-POD function
plt.plot(A_meas[idx,0],rssi_smooth[idx,0],label=r'$\tilde{p}_D$',\
    alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
plt.fill_between(np.squeeze(A_meas[idx,0]),\
        np.squeeze(rssi_est_gp) - std_gp,\
        np.squeeze(rssi_est_gp) + std_gp,\
        alpha=0.5, facecolor='g') 
plt.legend(fontsize=30)
# plt.xticks(np.arange(int(min(rssi_train)), int(max(rssi_train)), step=int((int(max(rssi_train))-int(min(rssi_train)))/5)),fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()