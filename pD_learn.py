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
    d_s = 2.5
    beta_s = 1
    beta_l = 4
    d_l = 4.5
    d_opt = 3
    p_D =  (( 1/(1+np.exp(beta_l*(d-d_l))) ) ) * ( 1/(1+np.exp(-beta_s*(d-d_s))) ) 
    return p_D
def area(d):
    A0 = 1
    k = 0.2
    return np.clip(A0 - k*d,0.0,1.0)

# log-distance path-loss model 
def PLM(r0,n,d):
    return r0 - 10*n*np.log10(d)

# inverse log-distance path-loss model 
def PLM_inverse(r0,n,r):
    return 10**( (r0-r)/(10*n) ) 

# function to predict the POD, given the GP model and a RSSI value 
def f(x,model):
    return model.predict(np.array([x]).reshape(-1,1))[0][0]
############################################################################################################################

def train():
                        # *** SETUP PARAMETERS ***
    # target-camera distance
    d_min = 0.5 # min target-camera distance
    d_max = 7.0 # max target-camera distance

    # synthetic RSSI generation according to the PLM
    r0 = -35 # RSSI offset at d0 = 1 [m]
    n = 2 # RSSI attenuation gain
    sigma = 3 # RSSI noise std. dev.

    N_train = int(1.0E+3) # train dataset cardinality

    # RSSI tracking via PF
    N_s = 100 # number of particles
    mu_omega = 0.0 # mean of the process model
    sigma_omega = 0.1 # std. dev. of the process model 
    pf_train = PF(N_s,\
                init_interval=[PLM(r0,n,d_max),PLM(r0,n,d_min)],\
                draw_particles_flag=False)

    # Kalman filter to smooth POD measurements
    x0 = 0.5 # initial guess
    kf_pD = Kalman_filter(1.0,0.0,1.0,0.1,0.1,x0,0.1) # A, B, C, Q, R, P
    kf_A = Kalman_filter(1.0,0.0,1.0,0.1,0.1,x0,0.1)

    # devices specifications
    T_RF = 0.1 # [s] Rx sampling time
    fps = 30 # camera fps

                        # *** GP TRAINING ***
    # train dataset generation
    distance_train = np.empty((N_train,1))
    pD_true = np.empty((N_train,1)) 
    pD_meas = np.empty((N_train,1))
    A_true = np.empty((N_train,1)) 
    A_meas = np.empty((N_train,1))
    rssi_raw = np.empty((N_train,1))
    rssi_smooth = np.empty((N_train,1))
    d=d_max
    for t in range(N_train):
        # true distance 
        d = np.clip(d+np.random.normal(-0.01,0.2),d_min,d_max)
        distance_train[t] = d
        # true POD (no RSSI noise, groundtruth knowledge on the POD function) --> groundtruth
        pD_true[t] = p_D(d)
        A_true[t] = area(d)

        # visual detection event modeled as Bernoulli r.va. with parameter p_D(d), sampled fps times before a new RSSI comes
        # (hence, we have a Binomial experiment with parameters p_D(d) and fps)
        kf_pD.predict()
        kf_pD.update( np.random.binomial(fps, pD_true[t]) / fps ) # number of detections / number of frames (proportion of successes)
        pD_meas[t] = kf_pD.x

        kf_A.predict()
        kf_A.update( A_true[t]+np.random.normal(0,1.0E-2) ) # number of detections / number of frames (proportion of successes)
        A_meas[t] = kf_A.x

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
    plt.plot(np.linspace(d_min,d_max,200), [p_D(d) for d in np.linspace(d_min,d_max,200)], \
        label=r'$p_D(d)$', linewidth=2)
    plt.plot(np.linspace(d_min,d_max,200), [p_D(d)*area(d) for d in np.linspace(d_min,d_max,200)], \
        label=r'$p_D(d)A(d)$', linewidth=2)
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
    plt.plot(pD_true,label=r'$p_{D,t}$') 
    plt.plot(pD_meas,label=r'$\tilde{p}_{D,t}$') 
    plt.xlabel(r"$t\;[ms]$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ## PD

    # define GP model
    rssi_train = rssi_smooth
    model_gp = GaussianProcessRegressor(Matern()+WhiteKernel(noise_level_bounds=(0.0,0.2)))
    # fit the GP model
    model_gp.fit(rssi_train,pD_meas)
    idx = np.argsort(rssi_train,axis=0)
    pD_est_gp,std_gp = model_gp.predict(rssi_train[idx,0],return_std=True)
    pD_est_gp = np.clip(pD_est_gp,0.0,1.0)

    # define NIGP model
    f_der = np.zeros(N_train)
    for i in range(N_train):
        f_der[i] = derivative(f,rssi_train[i],args=(model_gp,))
    model_nigp = GaussianProcessRegressor(kernel=\
        Matern(length_scale=np.exp(model_gp.kernel_.theta)[0],length_scale_bounds="fixed") + \
        WhiteKernel(noise_level=np.exp(model_gp.kernel_.theta)[1],noise_level_bounds=(0.0,0.2))+\
        WhiteKernel(),alpha=f_der**2)
    # fit the NIGP model
    model_nigp.fit(rssi_train,pD_meas)
    pD_est_nigp,std_nigp = model_nigp.predict(rssi_train[idx,0],return_std=True)
    pD_est_nigp = np.clip(pD_est_nigp,0.0,1.0)

    # plot training results
    pD_ideal = [p_D(PLM_inverse(r0,n,r)) for r in rssi_train[idx,0]]
    fig = plt.figure(figsize=(9,6))
    # plt.subplot(2,1,1)
    # plt.plot(rssi_train[idx,0], pD_ideal, label=r'$p_D(r)$',linewidth=2) # POD if RSSI were noiseless
    # plt.plot(rssi_train[idx,0], pD_est_gp,c='g',label='$\widehat{p}_D(r)$',linewidth=2)
    # plt.plot(rssi_train[idx,0],pD_meas[idx,0],label=r'$\tilde{p}_D$',\
    #     alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
    # plt.fill_between(np.squeeze(rssi_train[idx,0]),\
    #         np.squeeze(pD_est_gp) - std_gp,\
    #         np.squeeze(pD_est_gp) + std_gp,\
    #         alpha=0.5, facecolor='g') 
    # plt.legend(fontsize=30)
    # plt.xlabel(r'$r\;[dBm]$',fontsize=35)
    # # plt.fill_between(np.squeeze(rssi_train[idx,0]),\
    # #         np.squeeze(pD_est_gp) - (std_gp + np.squeeze(f_der[idx]**2)),\
    # #         np.squeeze(pD_est_gp) + (std_gp+ np.squeeze(f_der[idx]**2)),\
    # #         alpha=0.2, facecolor='g', label='CI + noise')
    # print('LML: {:5.3f}'.format(model_gp.log_marginal_likelihood(model_gp.kernel_.theta)) + ', '\
    #     'R^2'+': {:4.3f}'.format(model_gp.score(rssi_train[idx,0],pD_ideal)) + ', '\
    #     'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_gp)))
    #     )
    # plt.subplot(2,1,2)
    plt.plot(rssi_train[idx,0],pD_meas[idx,0],label=r'$\tilde{p}_D$',\
        alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
    plt.plot(rssi_train[idx,0], pD_ideal, label=r'$p_D(r)$',linewidth=2) # true RSSI-POD function
    plt.plot(rssi_train[idx,0], pD_est_nigp,c='g',label='$\widehat{p}_D(r)$',linewidth=2) # estimated RSSI-POD function
    plt.fill_between(np.squeeze(rssi_train[idx,0]),\
            np.squeeze(pD_est_nigp) - std_nigp,\
            np.squeeze(pD_est_nigp) + std_nigp,\
            alpha=0.5, facecolor='g') 
    plt.legend(fontsize=30,framealpha=0.5,ncol=2,columnspacing=0.1)
    plt.xlabel(r'$r\;[dBm]$',fontsize=35)
    print('LML: {:5.3f}'.format(model_nigp.log_marginal_likelihood(model_nigp.kernel_.theta)) + ', '\
        'R^2'+': {:4.3f}'.format(model_nigp.score(rssi_train[idx,0],pD_ideal)) + ', '\
        'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_nigp)))
            )
    plt.xticks(np.arange(int(min(rssi_train)), int(max(rssi_train)), step=int((int(max(rssi_train))-int(min(rssi_train)))/5)),fontsize=35)
    plt.yticks(fontsize=35)
    plt.ylim([-0.1,1])
    ax = plt.gca()
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('2')
    ax.grid(ls = ':', lw = 0.5)
    plt.tight_layout()
    plt.show()

    ## area

    # define GP model
    # model_gp_A = GaussianProcessRegressor(Matern()+WhiteKernel(noise_level_bounds=(0.0,0.2)))
    # # fit the GP model
    # model_gp_A.fit(rssi_train,A_meas)
    # A_est_gp,std_gp_A = model_gp_A.predict(rssi_train[idx,0],return_std=True)
    # A_est_gp = np.clip(A_est_gp,0.0,1.0)

    # # plot training results
    # A_ideal = [area(PLM_inverse(r0,n,r)) for r in rssi_train[idx,0]]
    # fig = plt.figure(figsize=(9,6))
    # print('LML: {:5.3f}'.format(model_gp.log_marginal_likelihood(model_gp.kernel_.theta)) + ', '\
    #     'R^2'+': {:4.3f}'.format(model_gp.score(rssi_train[idx,0],pD_ideal)) + ', '\
    #     'RMSE'+': {:4.3f}'.format(np.sqrt(mean_squared_error(pD_ideal,pD_est_gp)))
    #     )
    # plt.plot(rssi_train[idx,0], A_ideal, label=r'$p_D(r)$',linewidth=2) # true RSSI-POD function
    # plt.plot(rssi_train[idx,0], A_est_gp, c='g',label='$\widehat{p}_D(r)$',linewidth=2) # estimated RSSI-POD function
    # plt.plot(rssi_train[idx,0],A_meas[idx,0],label=r'$\tilde{p}_D$',\
    #     alpha=0.5,linestyle=':', marker='o',markersize=3.0, c='k') # measured POD on noisy RSSI
    # plt.fill_between(np.squeeze(rssi_train[idx,0]),\
    #         np.squeeze(A_est_gp) - std_gp_A,\
    #         np.squeeze(A_est_gp) + std_gp_A,\
    #         alpha=0.5, facecolor='g') 
    # plt.legend(fontsize=30)
    # plt.xlabel(r'$r\;[dBm]$',fontsize=35)
    # plt.xticks(np.arange(int(min(rssi_train)), int(max(rssi_train)), step=int((int(max(rssi_train))-int(min(rssi_train)))/5)),fontsize=35)
    # plt.yticks(fontsize=35)
    # ax = plt.gca()
    # ax.patch.set_edgecolor('black')  
    # ax.patch.set_linewidth('2')
    # ax.grid(ls = ':', lw = 0.5)
    # plt.tight_layout()
    # plt.show()

    return model_nigp#, model_gp_A

# train()