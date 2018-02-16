#!/usr/bin/env python
#---------------------------------------------------
# Class to fit 2D splines to data based on numpy.
#
# Date: 2017-04-19
# Author: Rolf Buehler (rolf.buehler@desy.de)
# 
#---------------------------------------------------


import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.interpolate as interp

class Spline2D():
    """
    Two dimensional B-Spline or Bezier Polynomial (CloughTocher)
    """
    def __init__(self, knots_x, knots_y,logspline = False,cloughtocher=True):
        """Setup regular 2d grid and boundary conditions
        @param   [float]    knots_x        Knots axis in x direction
        @param   [float]    knots_y        Knots axis in y direction
        @param   bool       logspline      Internally calculate with lg(z)? (input output is always in z)
        @param   bool       cloughtocher   Use scipy CloughTocher? If not use stardard Spline2D from scipy interp2d
        """
        
        #Setup grid
        self.knots_x       = knots_x.flatten()                          # Spline knots in x/y direction
        self.knots_y       = knots_y.flatten()
        knots_xx, knots_yy = np.meshgrid(knots_x, knots_y)              # Meshgrid of Spline knots in x/y direction
        self.knots_xx      = knots_xx
        self.knots_yy      = knots_yy
        self.knot_points   = np.dstack([knots_xx.flatten(), knots_yy.flatten()])[0]
        self.knots_shape   = self.knots_xx.shape                        # Knot shape (len(y),len(x))
        self.knot_vals     = np.zeros(shape = self.knots_shape)         # Knot values (initialize empty)
        self.built         = False                                      # Flag if spline has been build
        self.logspline     = logspline                                  # Flag if spine evaluation should happen in log z internally
        self.bknots        = {}                                         # Boundary knots for spline
                                                                        # convention: {"name":{"xx":[], "yy":[],"vals":[]}
        self.cloughtocher  = cloughtocher                               # Use cloughtocher interpolator, if not interp2d is used
                                                                        # which requires a regular grid
    
    def addbknots(self,name,xx,yy,vals):
        """Adds boundary knots taken into account in spline construction
        but not in spline fitting. 
        @param   string       name   Name of background knot array (e.g. "LeftBoundary")
        @param   [float]      xx     Coordinte mesh x values
        @param   [float]      yy     Coordinte mesh y values
        @param   [float]      vals   Knot values on passed mesh
        """
        points = np.dstack([xx.flatten(), yy.flatten()])[0]
        self.bknots[name]={"xx":xx,"yy":yy,"points":points,"vals":[]}
        if self.logspline:
            if np.any(vals<=0.):
                print "ERROR setbknots: At least one knot member is ~<0, cannot apply log10"
                return False
            else:
                vals = np.log10(vals)
        self.bknots[name]["vals"]=vals

    
    def printknots(self):
        "Prints out knots and values of spline"
        print 30*"-"
        print "* Knots x / y :"
        print self.knots_x
        print self.knots_y
        print "* Knots meshgrid x / y :"
        print self.knots_xx
        print self.knots_yy
        print "* Knots vals"
        print self.knot_vals
        print "* Bounday knots"
        print self.bknots
        print 30*"-"
        
    def __str__(self):
        "Print out spline properties"
        out =  "\nSpline2D length of arrays:\n"
        out += "Knots mesh shape y: "+str(self.knots_shape[0])+", x: "+str(self.knots_shape[1])+"\n"
        out += "Build "+str(self.built)+"\n"
        out += "Log spline "+str(self.logspline)+"\n"
        return out
    
    def build(self,knot_vals):
        """Build Spline for the passed knot values
        @param   [float]    knot_vals     Knot values  on the mesh of self.knots_xx etc.
        """
        
        #Check if logsline
        if self.logspline:
            if np.any(knot_vals<=0.):
                print "ERROR build: At least one knot member is ~<0, cannot apply log10"
                return False
            else:
                knot_vals = np.log10(knot_vals)
        
        #Check/correct if passed shape does not match
        if not np.array_equal(self.knots_shape, knot_vals.shape):
            #print "WARNING: passed knot array does not have the right shape"
            #print self.knots_shape, knot_vals.shape
            knot_vals = knot_vals.reshape(self.knots_shape)

        if self.built and  np.array_equal(knot_vals, self.knot_vals):
            return False
        else:
            self.knot_vals = knot_vals
            
            #Add boundary knots"
            bknot_points = []
            bvals_vals   = []
            for name in self.bknots:
                #~ print "Adding boundary", name
                bknot_points.append(self.bknots[name]["points"])
                bvals_vals.append(self.bknots[name]["vals"])
            
            #~ print np.array(bknot_points)[0],len(np.array(bknot_points)[0])
            #~ print self.knot_points,len(self.knot_points)
            #~ print np.array(bvals_vals).flatten()
            #~ print self.knot_vals.flatten()
            if len(bknot_points)>0:
                spline_points  = np.concatenate([self.knot_points,np.array(bknot_points)[0]])
                spline_vals    = np.concatenate([self.knot_vals.flatten(),np.array(bvals_vals).flatten()])
            else:
                spline_points = self.knot_points
                spline_vals   = self.knot_vals.flatten()
            #~ print spline_points,len(spline_points)
            #~ print spline_vals,len(spline_vals)
            
            if self.cloughtocher:
                self.spline2d  = interp.CloughTocher2DInterpolator(spline_points,spline_vals,tol=0.1,maxiter=100,rescale=True) # 
            else:
                pointsplit = np.hsplit(spline_points,2)
                #print "Build:"
                #print np.squeeze(pointsplit[0]),len(np.squeeze(pointsplit[0]))
                #print np.squeeze(pointsplit[1]),len(np.squeeze(pointsplit[1]))
                #print np.squeeze(spline_vals),len(np.squeeze(spline_vals))
                self.spline2d  = interp.interp2d(np.squeeze(pointsplit[0]), np.squeeze(pointsplit[1]),\
                                                 np.squeeze(spline_vals), kind='cubic',\
                                                 copy=False, bounds_error= False)         # Interpolator from np
                                                 
            self.built     = True
            return True
    
    def draw(self):
        "Draws spline in 3D"
        
        #Draw knots
        fig = plt.figure(figsize=(7, 7))
        knot_vals = self.knot_vals
        if self.logspline:
            knot_vals = np.power(10,self.knot_vals)
        plt.pcolormesh(knot_vals,cmap="nipy_spectral",snap=True)
        plt.title("Spline Knots")
        plt.colorbar()
        plt.tight_layout()
        
        #Draw interpolations
        fig = plt.figure(figsize=(7, 7))
        xbins = np.linspace(self.knots_x.min(),self.knots_x.max(),100)
        ybins = np.linspace(self.knots_y.min(),self.knots_y.max(),100)
        xxbins, yybins = np.meshgrid(xbins, ybins)
        vals = self.eval(xxbins, yybins)
        
        plt.pcolormesh(vals,cmap="nipy_spectral")
        plt.title("Spline Interpolation")
        plt.colorbar()
        plt.tight_layout()
        
        plt.show()
        
    def eval(self,xxvals, yyvals):
        """Evaluate spline on passed points
        @param    [float]    xxvals    x values of mesh
        @param    [float]    yyvals    y values of mesh
        """
        
        if not self.built:
            print "ERROR: spline has not jet been build"
        
        #Evaluate spline
        points = np.dstack([xxvals.flatten(), yyvals.flatten()])[0]
        
        if self.cloughtocher:
            svals = self.spline2d(points)
            svals = svals.reshape(xxvals.shape)
        else:
            #Assumes data on a regular grid (! required for interp2)
            pointsplit = np.hsplit(points,2)
            xvals = np.sort(np.unique(np.squeeze(pointsplit[0])))
            yvals = np.sort(np.unique(np.squeeze(pointsplit[1])))
            svals = self.spline2d(xvals,yvals)
        
        if self.logspline:
            svals = np.power(10,svals)
        
        return svals
        
    def logLike(self,knot_vals,x,y,log,data):
        """Get logLikelihood assuming spline model values in counts
        @param    [float]    knot_vals     Spline values on knot mesh for logL
        @param    [float]    x             x values to create evaluation mesh
        @param    [float]    y             y values to create evaluation mesh
        @param    bool       log           Is the spline working in log space?
        @param    [float]    data          Data to compare to spline model for logL on x vs y mesh
        """
        if log:
            self.build(np.power(10,knot_vals))
        else:
            self.build(knot_vals)
        model = self.eval(x,y)
        
        logL  = -poisson.logpmf(data.flatten(), model.flatten()).sum()
        return logL
        
    #'Nelder-Mead', 'Powel','CG','BFGS','Newton-CG','L-BFGS-B', 'TNC','COBYLA','SLSQP','dogleg','trust-ncg'
    def fit(self,xx,yy,zz,method='L-BFGS-B',tol=1e-3,knotbounds = 0.5,\
            options={'eps': 1e-3,'disp':True}):                         
        """Fits 2D spline to the passed data in counts
        @param    [float]    xx          Data parameter variable 1 array
        @param    [float]    yy          Data parameter variable 2 array
        @param    [float]    zz          Data value at xx,yy
        @param    string     method      Fit methods (see scipy.optimize.minimze manual)
        @param    float      tol         Fit tolerance (see scipy.optimize.minimze manual)
        @param    float      knotbounds  Bound of fit paramters [(1-k)* start val, (1+k)* start val]
        @param    {dict}     options     Further fit options (see scipy.optimize.minimze manual)
        @return   {dict}     Fit results (see scipy.optimize.minimze manual)
        """
        print  "---  Start fitting -----"
                
        print "- Start values of the fit:"
        print self.knot_vals

        if self.logspline:
            bound_min = self.knot_vals.flatten() + np.log10(1-knotbounds)
            bound_max = self.knot_vals.flatten() + np.log10(1+knotbounds)
            bounds    = np.dstack([bound_min,bound_max])[0]
            res = minimize(self.logLike,self.knot_vals,args=(xx,yy,True,zz,),\
                           method=method, tol=tol, bounds=bounds,options=options)
        else:
            bound_min = (1.-knotbounds)*self.knot_vals.flatten()
            bound_max = (1.+knotbounds)*self.knot_vals.flatten()
            bounds    = np.dstack([bound_min,bound_max])[0]
            res = minimize(self.logLike,self.knot_vals,args=(xx,yy,False,zz,),\
                           method=method, tol=tol, bounds=bounds,options=options) 
        
        print "- Fit result:"
        print self.knot_vals
        return res
    
    
