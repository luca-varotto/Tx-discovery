import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True

with open('./data/real_ucb1_10MC_45deg/performance.pkl', 'rb') as f: # ucbVSpf_45deg
    performance_BO = pickle.load(f)
    performance_BO = performance_BO[:,:,0]
with open('./data/ucbVSpf_45deg/performance.pkl', 'rb') as f:
    performance_pf = pickle.load(f)
    performance_pf = performance_pf[:,:,-1]

plt.figure(figsize=(9,6))
exp_types = 1
N_tot = np.shape(performance_BO)[1]
T_RF = 0.1
half_fov = np.pi/8
for exp_type in range(exp_types):
    label = 'BO' if exp_type ==0 else 'PF'
    col = 'b' if exp_type ==0 else 'r'
    perf2plot = performance_BO if exp_type==0 else performance_pf
    plt.plot(np.linspace(1,int(T_RF*N_tot),N_tot),np.mean(perf2plot,axis=0),label=label, color=col)
    plt.fill_between(np.linspace(1,int(T_RF*N_tot),N_tot),\
        np.mean(perf2plot,axis=0) - np.std(perf2plot,axis=0),\
        np.mean(perf2plot,axis=0) + np.std(perf2plot,axis=0),\
        alpha=0.1, facecolor=col)
    plt.plot(np.linspace(1,int(T_RF*N_tot),N_tot),half_fov*np.ones(N_tot),linestyle='--',color='k')
plt.ylabel(r'$|\gamma-\gamma_t^*|\;[rad]$',fontsize=35)  # p(|\gamma - \alpha_t|<\theta)    
plt.xlabel(r'$t\;[s]$',fontsize=35)
plt.ylim([-0.1,None])
plt.xlim([0.99,N_tot*T_RF])
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
# plt.legend(fontsize=30,framealpha=0.5)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.tight_layout()
plt.show()

