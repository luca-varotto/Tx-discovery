from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel, PairwiseKernel)
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import math
import scipy.stats
import pickle

############################################################################################################################

import BayOpt_modified,pD_learn
from pf import PF

############################################################################################################################

############################################################################################################################

MC_tests =1
want2plot = 1 if MC_tests ==1 else 0
N_det = 20 # number of deterministic movements (pure exploration)
N_tot = N_det + 100 #if MC_tests >1 else N_det + 300 # total number of movements
change_time = int(N_tot+1) #if MC_tests >1 else int(N_tot/3)
exp_types = 1
half_fov = np.pi/8

performance = np.empty((MC_tests,N_tot,exp_types))

# train pD predictor
# model_nigp, model_gp_A = pD_learn.train()
model_nigp = pD_learn.train()

for MC_idx in range(MC_tests): 

    print("MC test %d"%(MC_idx))

    # define the coordinates of the targets (each row, [x,y] coordinates of one target)
    angles = []
    while True:
        tx = np.array([np.random.uniform(-5,5),np.random.uniform(-5,5)]) 
        if np.linalg.norm(tx) < 4:
            break
    trgt = tx

    # define which target is the Tx
    tx_idx = 0

    angles.append(np.arctan2(tx[1],tx[0]))
    while True:
        tx2 = np.random.uniform(-5,5,(1,2))
        angle = np.arctan2( tx2[0,1],tx2[0,0] )
        if abs( np.linalg.norm(tx) - np.linalg.norm(tx2) ) > 1 and not angle in angles:
            trgt = np.vstack((trgt, tx2))
            angles.append(angle)
            break
    n_targets = 20
    for k in range(n_targets-2): 
        while True:
            new_target = np.random.uniform(-5,5,(1,2))
            angle = np.arctan2( new_target[0,1],new_target[0,0] )
            if abs( np.linalg.norm(tx) - np.linalg.norm(new_target) ) > 1 and not angle in angles:# \
                # and abs(np.linalg.norm(tx2) -np.linalg.norm(new_target) ) > 1.5:
                trgt = np.vstack((trgt, new_target))
                angles.append(angle)
                break

    # devices parameters
    fps = 30
    T_RF = 0.1
    # synthetic RSSI generation according to the PLM
    r0 = -35 # RSSI offset at d0 = 1 [m]
    n = 2 # RSSI attenuation gain
    sigma = 3 # RSSI noise std. dev.

    for exp_type in range(exp_types):
        # initial camera orientation
        x = 0.0
        X = np.asarray([x])
        y = np.asarray([0.0])
        y_tot = [y]
        QoC = []
        QoS = []

        ucb_flag = 1 # if ??? else 0
        model_GP = model_nigp # if exp_type==0 else model_gp_A

        if want2plot:
            fig = plt.figure(figsize=(9,6))
            plt.ion()
            plt.show()

        # RSSI tracking via PF
        N_s = 100 # number of particles
        mu_omega = 0.0 # mean of the process model
        sigma_omega = 0.1 # std. dev. of the process model 
        pf = PF(N_s,\
                    init_interval=[pD_learn.PLM(r0,n,5.5),pD_learn.PLM(r0,n,0.5)],\
                    draw_particles_flag=False)

        N_s_pod = 500
        pf_pod = PF(N_s_pod,\
                init_interval=[-np.pi,np.pi],\
                draw_particles_flag=False)

        # reshape into rows and cols
        X = X.reshape(len(X), 1)
        y = y.reshape(len(y), 1)
        # define the model
        model = GaussianProcessRegressor()# Matern() + WhiteKernel()
        # fit the model
        model.fit(X, y)
        l_pD = []
        l_A = []
        reset_times = [0]
        for t in tqdm(range(N_tot)):

            if t == change_time:
                tx_idx=1
                       
            # detect anomaly likelihood and reset BO process
            # if t>40 and t-reset_times[-1]>40 and \
            #     np.mean(y[-40:-25])>0.4*max(y) and np.mean(y[-25:-1]) < 0.4*max(y):
            #     X = X[-1].reshape(-1,1)
            #     y = y[-1].reshape(-1,1)
            #     model = GaussianProcessRegressor(Matern() + WhiteKernel())
            #     model.fit(X, y)
            #     reset_times.append(t)
            #     print("reset at t=%d" %(t))

            # RSSI sample
            rssi_raw = pD_learn.PLM(r0,n,np.linalg.norm(trgt[tx_idx])) + np.random.normal(0,sigma)  
            # RSSI update
            pf.update(rssi_raw,sigma)
            # resampling
            pf.SIS_resampling()
            # compute MMSE estimate
            pf.estimation(est_type='MMSE')

            pf.predict(mu_omega,sigma_omega)
            pD_predict, std_pd = model_GP.predict(pf.particles.reshape(-1,1),return_std=True)
            E_pD = np.average(pD_predict, weights=pf.weights, axis=0)
            std_pd = np.average(std_pd, weights=pf.weights, axis=0)
            # A_predict, std_A =  model_gp_A.predict(pf.particles.reshape(-1,1),return_std=True)
            # E_A = np.average(A_predict, weights=pf.weights, axis=0)
            # std_A = np.average(std_A, weights=pf.weights, axis=0)

            x_proposed, opt = BayOpt_modified.opt_acquisition(X, y, model, pf_pod.estimate,ucb_flag)
            if t-reset_times[-1] < N_det: # deterministic exploration
                x += np.pi/6
            else: # select the next point to sample
                x = x_proposed #if abs(x-x_proposed) < np.pi/6 else x+np.sign(x_proposed)*np.pi/6
            if x > np.pi:
                x = x-2*np.pi
            if x < -np.pi:
                x = x+2*np.pi

            # compute camera FoV
            fov_length = 10
            fov_poly = Polygon(np.array([[0,0],\
                        [fov_length*np.cos(x+half_fov),fov_length*np.sin(x+half_fov)],\
                        [fov_length*np.cos(x-half_fov),fov_length*np.sin(x-half_fov)] ]))
            opt = np.asscalar(opt) if ucb_flag == 1 else pf_pod.estimate
            fov_opt = Polygon(np.array([[0,0],\
                        [fov_length*np.cos(opt+half_fov),fov_length*np.sin(opt+half_fov)],\
                        [fov_length*np.cos(opt-half_fov),fov_length*np.sin(opt-half_fov)] ]))

            # check if current orientation is optimal
            alpha_star = np.arctan2(trgt[tx_idx][1],trgt[tx_idx][0])
            # alpha_star = 2*np.pi + alpha_star if alpha_star < 0 else alpha_star
            x_2pi = 2*np.pi + x if x < 0 else x
            # opt = x
            Delta = min( abs(alpha_star-opt) , 2*np.pi-abs(alpha_star-opt) ) # if abs(alpha_star-opt) <= np.pi else 2*np.pi-abs(alpha_star-opt)
            performance[MC_idx,t,exp_type] = 1 if fov_opt.contains(Point(trgt[tx_idx])) else 0 #fov_poly.contains(Point(trgt[tx_idx])) else 0

            # find points inside FoV
            for i in range(np.shape(trgt)[0]):
                if fov_poly.contains(Point(trgt[tx_idx])):
                    pD_meas = np.random.binomial(fps, pD_learn.p_D(np.linalg.norm(trgt[tx_idx]))) / fps
                    # A_meas = pD_learn.area(np.linalg.norm(trgt[tx_idx]))+ np.random.normal(0,1.0E-2)
                    # print(pD_learn.p_D(np.linalg.norm(trgt[tx_idx]))*pD_learn.area(np.linalg.norm(trgt[tx_idx])),\
                    #         E_pD*E_A, \
                    #         pD_meas*A_meas)
                    # print(pD_learn.p_D(np.linalg.norm(trgt[tx_idx])),\
                    #         E_pD, \
                    #         pD_meas)
                    break
                else:
                    if fov_poly.contains(Point(trgt[i])):
                        pD_meas = np.random.binomial(fps, pD_learn.p_D(np.linalg.norm(trgt[i]))) / fps
                        # A_meas = pD_learn.area(np.linalg.norm(trgt[i])) + np.random.normal(0,1.0E-2)
                        break
                    else: 
                        pD_meas = 2
                        # A_meas = 2
            
            # predict
            pf_pod.predict(0.0,0.05)
            pf_pod.particles = np.clip(pf_pod.particles,-np.pi,np.pi)
            # update
            l_pD.append(scipy.stats.norm.pdf(pD_meas,E_pD,std_pd))
            # l_A.append(scipy.stats.norm.pdf(A_meas,E_A,0.1))
            l = np.ones(N_s_pod)
            l[(pf_pod.particles[:,0]>x-0.1) & (pf_pod.particles[:,0]<x+0.1)] = l_pD[-1]
            pf_pod.weights *= l
            pf_pod.weights += 1.e-300      # avoid round-off to zero
            pf_pod.weights /= sum(pf_pod.weights) # normalize
            # resampling
            pf_pod.SIS_resampling()
            # compute MMSE estimate
            pf_pod.estimation(est_type='MAP')

            Delta_t = min( abs(alpha_star-x) , 2*np.pi-abs(alpha_star-x) )# abs(alpha_star-x) if abs(alpha_star-x) <= np.pi else 2*np.pi-abs(alpha_star-x)
            QoC_actual,QoS_actual= BayOpt_modified.objective(x,E_pD,pD_meas,\
                Delta_t,pD_learn.PLM(r0,n,np.linalg.norm(tx)))
            # QoS_actual = l_pD[-1]
            QoC.append(QoC_actual)
            QoS.append(QoS_actual)
            if exp_type == 0:
                reward = QoC_actual*QoS_actual
            elif exp_type == 1:
                reward = QoC_actual
            else:
                reward = QoS_actual
            actual = np.asscalar( reward )

            # add the data to the dataset
            try:
                X = np.vstack((X, [[x]]))
            except ValueError:
                X = np.vstack((X, [[np.asscalar(x)]]))
            y = np.vstack((y, [[actual]]))
            y_tot.append(actual)

            if want2plot:
                sub1= plt.subplot(2,1,1)
                fov = plt.Polygon(np.array([[0,0],\
                                    [fov_length*np.cos(x+half_fov),fov_length*np.sin(x+half_fov)],\
                                    [fov_length*np.cos(x-half_fov),fov_length*np.sin(x-half_fov)] ]), color='r', alpha=0.1)
                plt.scatter(0,0,color='r',label='camera') # platform
                plt.plot([0,fov_length*np.cos(x+half_fov)],[0,fov_length*np.sin(x+half_fov)],'--r')
                plt.plot([0,fov_length*np.cos(x-half_fov)],[0,fov_length*np.sin(x-half_fov)],'--r')
                plt.gca().add_patch(fov)
                for i in range(np.shape(trgt)[0]): # targets
                    c = 'b' if i != tx_idx else 'k'
                    plt.scatter(trgt[i][0],trgt[i][1],color=c)
                    plt.xlim([-6,6])
                    plt.ylim([-6,6])
                    plt.xlabel(r"$\;[m]$",fontsize=25)
                    plt.ylabel(r"$\;[m]$",fontsize=25)
                ax = plt.gca()
                ax.patch.set_edgecolor('black')  
                ax.patch.set_linewidth('2')
                ax.grid(ls = ':', lw = 0.5)

                # sub2=plt.subplot(3,1,2)
                # idx = np.argsort(pf_pod.particles,axis=0)
                # plt.scatter(pf_pod.particles[idx]*180/np.pi,np.zeros(len(pf_pod.particles)),s=pf_pod.weights*2500)
                # plt.scatter(pf_pod.estimate*180/np.pi,0,c='y',s=max(pf_pod.weights)*2500,label=r'$\hat{\alpha}_t$')
                # plt.xlim([-180,180])
                # plt.xlabel(r'$\alpha\;[^\circ]$',fontsize=25)
                # ax = plt.gca()
                # ax.patch.set_edgecolor('black')  
                # ax.patch.set_linewidth('2')
                # ax.grid(ls = ':', lw = 0.5)

                sub3 = plt.subplot(2,1,2)
                plt.plot(QoC,label=r'QoC',linewidth=0.5)
                plt.plot(QoS,label=r'$E[p_D]-\tilde{p}_D$',linewidth=0.5)
                plt.plot(y_tot[1:],label=r'$y$',linewidth=1.5)
                if t >= change_time:
                    plt.vlines(change_time,0,1,linestyles='--',color='c',label='change target')
                    # plt.legend(fontsize=30)
                for j in reset_times:
                    if j!=0:
                        plt.vlines(j,0,1,linestyles='--',color='k',label='new search')
                plt.xlim([1,N_tot])
                # plt.ylabel(r'$E[p_D]$',fontsize=25)    
                plt.xlabel(r'$t\;[s]$',fontsize=25)
                plt.legend(fontsize=20)
                ax = plt.gca()
                ax.set_xticklabels(range(1,(int(N_tot*T_RF))))
                ax.patch.set_edgecolor('black')  
                ax.patch.set_linewidth('2')
                ax.grid(ls = ':', lw = 0.5)
                plt.tight_layout()
                # plt.savefig('./img/%d.png'%(t))
                if plt.waitforbuttonpress(0.1):
                    break 
                plt.pause(0.001)
                sub1.cla()
                # sub2.cla()
                sub3.cla()

            # update the model
            model.fit(X, y)
plt.ioff()

isInsideFoV = np.where(performance<=half_fov, 1, 0)

plt.figure(figsize=(9,6))
plt.subplot(2,1,1)
BayOpt_modified.plot(X, y, model)
plt.subplot(2,1,2)
labels = ['QoC+QoS' ,'QoC only','QoS only','PF']
colors = ['b','r','g','y']
for exp_type in range(exp_types):

    print(np.mean(performance[:,-1,exp_type],axis=0),
            np.std(performance[:,-1,exp_type],axis=0))
    print(np.mean(isInsideFoV[:,-1,exp_type],axis=0))

    label = labels[exp_type]
    col = colors[exp_type]
    plt.plot(np.mean(performance[:,:,exp_type],axis=0),label=label, color=col)
    plt.fill_between(range(N_tot),\
        np.mean(performance[:,:,exp_type],axis=0) - np.std(performance[:,:,exp_type],axis=0),\
        np.mean(performance[:,:,exp_type],axis=0) + np.std(performance[:,:,exp_type],axis=0),\
        alpha=0.1, facecolor=col)
    plt.plot(range(N_tot),half_fov*np.ones(N_tot),linestyle='--',color='k')
plt.legend()
plt.show()

# savings
with open('./data/X.pkl', 'wb') as f:
        pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)
with open('./data/y.pkl', 'wb') as f:
        pickle.dump(y_tot, f, pickle.HIGHEST_PROTOCOL)
with open('./data/performance.pkl', 'wb') as f:
        pickle.dump(performance, f, pickle.HIGHEST_PROTOCOL)

