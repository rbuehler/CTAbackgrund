#!/usr/bin/env python
#---------------------------------------------------
# Script to test the Spline2Dnp class. Steps:
#  1) Create data model in coarse binning and fine binning
#  2) Build spline from coarse binned model
#  3) Simulate data from fine model
#  4) Fit spline to data
# in  sky coordinates simulated with ctools
#
# Date: 2017-04-19
# Author: Rolf Buehler (rolf.buehler@desy.de)
# 
#---------------------------------------------------

import Spline2Dnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


lglikelg     = True # Set spline model to log(counts)? 
fitspline    = True

def model(xx,yy):
    "helper function defining model for spline building"
    return (np.sin(xx)-1.5)**2 + (np.cos(yy)-2)**2+40.
    #return xx+10

# (1) Setup data model

xmax = 4.0
ymax = 5.0
xcoarse = np.linspace( 1.0 , xmax, int(xmax) )
ycoarse = np.linspace( 0.0 , ymax, int(ymax) )
xxcoarse, yycoarse = np.meshgrid(xcoarse, ycoarse)
modelcoarse = model(xxcoarse, yycoarse)

xfine = np.linspace( 0.0 , xmax, int(xmax)*20 )
yfine = np.linspace( 0.0 , ymax, int(ymax)*20 )
xxfine, yyfine = np.meshgrid(xfine, yfine)
modelfine = model(xxfine, yyfine)

# (2) Build spline from coarse model
spl = Spline2Dnp.Spline2D(xcoarse,ycoarse,lglikelg)

# Add boundary knots
bknot_x = np.array([0.])
bknot_y =  ycoarse
bknot_xx, bknot_yy = np.meshgrid(bknot_x, bknot_y)
spl.addbknots("xmin",bknot_xx,bknot_yy,model(bknot_xx,bknot_yy))

if fitspline:
    spl.build(1.2*modelcoarse)
else:
    spl.build(modelcoarse)
    

spl.printknots()
print spl

# (3) Get random realisation of data
datafine = np.random.poisson(modelfine)
print "Data shape:",datafine.shape

# (4) Fit spline to data
if fitspline:
    res     = spl.fit(xxfine,yyfine,datafine,method='L-BFGS-B',tol=1e-3,options={'eps': 1e-3,'disp':True})
    #'Nelder-Mead', 'Powel','CG','BFGS','Newton-CG','L-BFGS-B','TNC','COBYLA','SLSQP','dogleg','trust-ncg'
    print res
    
splfine = spl.eval(xxfine, yyfine)

#### ------------- plot ---------------
fig = plt.figure(3,figsize=(10, 10))
cmin= min(modelfine.flatten())
cmax= max(modelfine.flatten())

cm = plt.cm.get_cmap("nipy_spectral")

plt.subplot(221,title="Model data")
plt.pcolormesh(xfine,yfine,modelfine,cmap="nipy_spectral",vmin=cmin, vmax=cmax)
plt.colorbar()
plt.scatter(xxcoarse, yycoarse, s=40,c=modelcoarse,cmap=cm,vmin=cmin, vmax=cmax)
plt.scatter(bknot_xx, bknot_yy, s=40,c=model(bknot_xx,bknot_yy),cmap=cm,vmin=cmin, vmax=cmax)

plt.subplot(222,title="Simulated data")
plt.pcolormesh(xfine,yfine,datafine,cmap="nipy_spectral",vmin=cmin, vmax=cmax)
plt.colorbar()
plt.scatter(xxcoarse, yycoarse, s=40,c=modelcoarse,cmap=cm,vmin=cmin, vmax=cmax)
plt.scatter(bknot_xx, bknot_yy, s=40,c=model(bknot_xx,bknot_yy),cmap=cm,vmin=cmin, vmax=cmax)

plt.subplot(223,title="Spline fit to data")
print splfine,splfine.shape
plt.pcolormesh(xfine,yfine,splfine,cmap="nipy_spectral") #,vmin=cmin, vmax=cmax
plt.colorbar()
plt.scatter(xxcoarse, yycoarse, s=40,c=modelcoarse,cmap=cm,vmin=cmin, vmax=cmax)
plt.scatter(bknot_xx, bknot_yy, s=40,c=model(bknot_xx,bknot_yy),cmap=cm,vmin=cmin, vmax=cmax)

plt.subplot(224,title="Residual (model-spline)/model")
resfine = (splfine-modelfine)/modelfine
absmax = np.abs(resfine).max()
plt.pcolormesh(xfine,yfine,resfine,cmap="seismic",vmin=-absmax, vmax=absmax)
plt.colorbar()
#plt.scatter(xxcoarse, yycoarse, s=40,facecolors='none')
plt.scatter(bknot_xx, bknot_yy, s=40, c="k")

plt.tight_layout()
plt.show()
#spl.draw()
