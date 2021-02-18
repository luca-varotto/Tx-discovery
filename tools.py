                    # ***** PLOT DETECTION PROBABILITY FUNCTION *****

import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# detection probability as function of zoom and camera-target distance
def p_D(x,y):
    d_critic = 3
    f_critic = 0.5
    gamma = 0.1
    eps1 =0.9
    eps2 = 0.1
    z0 = ( ( 1 + np.exp( gamma*( - eps1) )) * (( 1 + np.exp( gamma*( - eps2) ))) )**(-1)
    return ( ( ( 1 + np.exp( gamma*( x/y - eps1) )) * ( ( 1 + np.exp( -gamma*( x/y - eps2) )) ) )**(-1) / z0 )

# plot a 2D p_D
d = np.arange(0.5,5.5,0.1)
f = np.arange(0.0,1.0,1.0E-1)
X,Y = meshgrid(d, f) # grid of point
Z = p_D(X, Y) # evaluation of the function on the grid

# im = imshow(Z,cmap=cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# cset = contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right
# # latex fashion title
# title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
# show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                cmap=cm.RdBu,linewidth=0, antialiased=False)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
title(r'$\left[ 1 + e^{\gamma(\varepsilon_t - \epsilon)}\right]^{-1}\left( 1 + e^{-\gamma \epsilon} \right)$', fontsize=18)
ax.set_xlabel(r'$d\left( \mathbf{c}_t, \mathbf{p}_t  \right)$ [m]',fontsize=18, labelpad=20)
ax.set_ylabel(r'$f_t$ [m]',fontsize=18, labelpad=20)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
ax.zaxis.set_tick_params(labelsize=18)
plt.show()

