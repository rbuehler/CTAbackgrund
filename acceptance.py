#!/usr/bin/env python

#---------------------------------------------------
# Script to simulate and visualize acceptance
# in  sky coordinatessimulated with ctools
#
# Date: 2017-04-19
# Author: Rolf Buehler (rolf.buehler@desy.de)
# 
#---------------------------------------------------

from astropy.io import fits
import ctools
import cscripts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from astropy.table import Table, hstack, vstack
import Spline2Dnp
from scipy.interpolate import interp2d,griddata


#----------------------------------------------------------------------
#                Set running options: START
#          Setup methods to be run in __main__ at the bottom
#----------------------------------------------------------------------

# Set global variables of simulation.
ra         = 83.63              # Pointing direction
dec        = 22.01
tsim       = 18000.0             # 30 min = 1800
caldb      = "prod2"
irf        = "South_0.5h"
labelsize  = 12

outdir   = "./"
modeldir = "./models/"

fitspline = False               # Fit the spline of just setup one according to model?
bknots    = False                  # Should background knots be inserted?
lgspline  = False                   # Should the spline internally work in log space?

# --------- energy binning for ctools --------
emin       = 0.1  # minimum energy in TeV
lgemin     = np.log10(emin)
emax       = 10   # maximum energy in TeV
lgemax     = np.log10(emax)
enumbins   = 64   # number of energy bins
ebinmax    = enumbins -1

# --------- energy binning for ctools --------
npix       = 150   # Number of pixels along X/Y axis, choose pair number
binsz      = 0.04  # Bin diameter in degrees
nobinmax   = npix/2.
omax       = nobinmax*binsz
oextent    = [-omax,omax,-omax,omax]

# --------- Define offset and energy binning for plots and spline evaluation --------
oedges    = np.linspace(0,omax,80)      
ocenter   = (oedges[:-1]+oedges[1:])/2.
lgeedges  = np.linspace(lgemin,lgemax,60)
lgecenter = (lgeedges[:-1]+lgeedges[1:])/2.
oo,lgee   = np.meshgrid(ocenter,lgecenter)


# --------- Define offset and energy binning for spline knots--------
nkobins   = 6
nklgebins = 7
do   = 0.0 #(oedges[1]-oedges[0])/2.    # Offset distance to edge of simulation
dlge = 0.0 # (lgeedges[1]-lgeedges[0])/2. # lgEnergy distance to edge of simulation

ko   = np.linspace(0+do,omax-do,nkobins)
klge = np.linspace(-1+dlge,lgemax-dlge,nklgebins)
kko, kklge = np.meshgrid(ko, klge)

#----------------------------------------------------------------------
#                Set running options: END
#----------------------------------------------------------------------

def ebin2lge(ebin):
    """Helper function to obtain energy lgcenter from bin. 
    @param    int     ebin   Energy bin Nr. Bin Nr. start at 0.
    @return   float   Energy in TeV
    """
    dlge    = (lgemax - lgemin )/enumbins
    lgener  = lgemin  + dlge / 2. + ebin*dlge
    return lgener
    
def xybin2xy(xybin):
    """Helper function to covert X or Y bin Nr. into offset in degrees. 
    @param    int     xybin   X or Y bin Nr.  Bin Nr. start at 0.
    @return   float   Offset in degrees
    """
    return xybin*binsz + binsz/2. - omax


#ko   = np.array([0.,0.3,1.5,3.])
#klge   = np.array([-1,-0.8,-0.5,0.,1.])

# -------------------------------------------------
# -----Simulate, show and tabulate acceptance -----
# -------------------------------------------------

    
def sim(src     = "cCrab" ):   #bkg, Crabbkg
    "Simulates acceptance ccube and model cube"
    print 30*"-"+"\n","Simulating",src+"\n"+30*"-"
    #Simulate observation
    sim = ctools.ctobssim()
    sim["inmodel"]   = modeldir+src+".xml"
    sim["outevents"] = outdir+"events_"+src+".fits"
    sim["caldb"]     = caldb
    sim["irf"]       = irf
    sim["ra"]        = ra
    sim["dec"]       = dec
    sim["rad"]       = 10.0
    sim["tmin"]      = 0.0
    sim["tmax"]      = tsim
    sim["emin"]      = emin
    sim["emax"]      = emax
    sim["edisp"]     = False
    sim.execute()
    
    #Bin data into a cube
    ctbin = ctools.ctbin()
    ctbin["inobs"]    = outdir+"events_"+src+".fits"
    ctbin["outcube"]  = outdir+"ccube_"+src+".fits"
    ctbin["ebinalg"]  = "LOG"
    ctbin["emin"]     = emin
    ctbin["emax"]     = emax
    ctbin["enumbins"] = enumbins
    ctbin["nxpix"]    = npix
    ctbin["nypix"]    = npix
    ctbin["binsz"]    = binsz
    ctbin["coordsys"] = "CEL"
    ctbin["xref"]     = ra
    ctbin["yref"]     = dec
    ctbin["proj"]     = "AIT"
    ctbin.execute()
    
    #Create model cube
    ctmodel = ctools.ctmodel()
    ctmodel["inobs"]   = outdir+"events_"+src+".fits"
    ctmodel["inmodel"] = modeldir+src+".xml"
    ctmodel["incube"]  = outdir+"ccube_"+src+".fits"
    ctmodel["caldb"]   = "prod2"
    ctmodel["caldb"]   = caldb
    ctmodel["irf"]     = irf
    ctmodel["outcube"] = outdir+"mcube_"+src+".fits"
    ctmodel["edisp"]   = False
    ctmodel.execute()

    
def show(ebin = 3):
    "Plots acceptance"
    print 30*"-"+"\n","Plotting acceptance\n"+30*"-"
    
    #Simulated fits images
    mcube       = outdir+"mcube_bkg.fits"
    ccube       = outdir+"ccube_bkg.fits"
    mcube_ccrab = outdir+"mcube_cCrab.fits"
    
    #Get data
    hdu_mcube  = fits.open(mcube)
    mcube_data = hdu_mcube[0].data
    mcube_img  = mcube_data[ebin]
    
    ebounds = hdu_mcube["EBOUNDS"].data
    lges     = []
    for e in ebounds:
        lgemean = (np.log10(e[0])+np.log10(e[0]))/2. -9  # Mean lge in TeV
        lges.append(lgemean)
    lges = np.array(lges)
    
    hdu_ccube = fits.open(ccube)
    ccube_data = hdu_ccube[0].data
    ccube_img = ccube_data[ebin]
    
    hdu_mcube_ccrab = fits.open(mcube_ccrab)
    mcube_data_ccrab = hdu_mcube_ccrab[0].data
    
    #Plot
    fig = plt.figure(figsize=(12,10),edgecolor="w")
    
    # Count and model example images
    plt.subplot(221)
    plt.imshow(mcube_img, cmap="nipy_spectral",interpolation='none',extent=oextent)
    plt.colorbar()
    plt.title('Acceptance')
    plt.xlabel("dRA [deg]",fontsize=labelsize)
    plt.ylabel("dDec [deg]",fontsize=labelsize)
    plt.subplot(222)
    plt.imshow(ccube_img, cmap="nipy_spectral",interpolation='none',extent=oextent)
    plt.colorbar()
    plt.title('Simulated data')
    plt.xlabel("dRA [deg]",fontsize=labelsize)
    plt.ylabel("dDec [deg]",fontsize=labelsize)
    
    #Counts as a function of energy
    plt.subplot(224)
    cbin = int(npix/2.)
    plt.plot(lges,np.log10(mcube_data[:,cbin,cbin]),label="Counts per bin in center")
    avcounts = np.sum(np.sum(ccube_data,axis=1),axis=1)
    e_lgavcounts = np.log10(avcounts+np.sqrt(avcounts))-np.log10(avcounts)
    plt.errorbar(lges,np.log10(avcounts),e_lgavcounts,label="Total counts in sim.",fmt="o")
    avmcounts = np.sum(np.sum(mcube_data,axis=1),axis=1)
    plt.plot(lges,np.log10(avmcounts),label="Total counts")
    avmcounts_ccrab = np.sum(np.sum(mcube_data_ccrab,axis=1),axis=1)
    print "Total expected counts from cCrab",avmcounts_ccrab.sum()
    plt.plot(lges,np.log10(avmcounts_ccrab),label="Total counts cCrab")
    for lge in klge:
        plt.axvline(x=lge, linewidth=0.5, color='k',linestyle="--")
    leg = plt.legend(loc="lower left")
    leg.draw_frame(False)
    plt.xlabel("lg(Energy/TeV)",fontsize=labelsize)
    plt.ylabel("lg(Counts)",fontsize=labelsize)
    
    #Coutns as a function of offset
    plt.subplot(223)
    offsets = xybin2xy(np.array(range(0,len(mcube_img[cbin]))))
    for enum in range(0,len(lges),2):
        plt.plot(offsets,np.log10(mcube_data[enum,cbin]),label = str(lges[enum]))
    for o in ko:
        plt.axvline(x=o, linewidth=0.5, color='k',linestyle="--")
    plt.xlabel("Offset [deg]",fontsize=labelsize)
    plt.ylabel("lg(Counts per bin)",fontsize=labelsize)
    plt.legend(title="lg(Energy/TeV)")
    
    plt.tight_layout()
    plt.show()
    
def img2tab(img,dmax=omax):
    """Stores pixel values of an image in a astropy table, adding spherical coordinates
    @param   ndarray   img    2D numpy array with acceptance information
    @param   int      dmax    Maximum Offset to include
    @return  tab   Astropy table with cube information
    """

    #Setup data arrays
    xx = range(0,len(img[0,:]))
    yy = range(0,len(img[:,0]))
    XX,YY = np.meshgrid(xx,yy)
    xbin = XX.flatten()
    ybin = YY.flatten()
    cts  = img[XX,YY].flatten()
    
    #Add distance to center (offset)
    xc     = xybin2xy(xbin)
    yc     = xybin2xy(ybin)
    offset = np.sqrt(xc*xc+yc*yc)
    
    #Mask events with obinance greater dmax
    mask   =  offset<dmax
    xbin   = xbin[mask]
    ybin   = ybin[mask]
    cts    = cts[mask]
    offset = offset[mask]
    
    #Create table
    tab = Table()
    tab["xbin"]   = xbin
    tab["ybin"]   = ybin
    tab["cts"]    = cts
    #tab["obin"]   = obin
    tab["offset"] = offset
    tab.sort("offset")
    return tab
    
def cube2tab(cube,dmax=omax):
    """Stores pixel values of a cube in a astropy table, adding spherical coordinates
    @param   ndarray  ccube   Counts cube array to be stored in table
    @param   int      dmax    Maximum Offset to include
    @return  tab   Astropy table with cube information
    """
    tabs = []
    #Get tables for different energy bins
    for ebin in range(0,enumbins):
        tab = img2tab(cube[ebin],dmax)
        tab["ebin"] = ebin
        tab["lge"]  = ebin2lge(ebin)
        tabs.append(tab)
    #Stack and return table
    taball = vstack(tabs)
    return taball


# ----------------------------------------
# ------------- 2D Spline ----------------
# ----------------------------------------

def histcube(ctab,oedges, lgeedges):
    "Make 2D histograms of cts vs offset and energy, normalize to Nr pixels"
    mcube_h2_cts, x, y = np.histogram2d(ctab["offset"], ctab["lge"], bins=(oedges, lgeedges),weights=ctab["cts"])
    mcube_h2_ent, x, y = np.histogram2d(ctab["offset"], ctab["lge"], bins=(oedges, lgeedges))
    mcube_h2           = (mcube_h2_ent>0)*mcube_h2_cts/(mcube_h2_ent+1e-9)
    return mcube_h2.T 
    
def plotpanel(oo, lgee, spline_h2,cmin,cmax,kko, kklge,kkvals, cmap="nipy_spectral"):
    "Helper function to plot one panel"
    #plt.pcolormesh(oo, lgee, spline_h2,clip_on=False)
    doo =  0#np.abs(oo[0][0]-oo[0][1])/2.
    dlgee =  0#np.abs(lgee[0][0]-lgee[1][0])/2.
    ext = [0-doo,omax+doo,lgemin-dlgee,lgemax+dlgee]
    plt.imshow(np.flipud(spline_h2),cmap=cmap,vmin=cmin, vmax=cmax,\
               interpolation='nearest',extent=ext, aspect='auto')
    plt.xlabel("offset",fontsize=labelsize)
    plt.ylabel("lg(energy)",fontsize=labelsize)
    plt.colorbar()
    cm = plt.cm.get_cmap(cmap)
    plt.scatter(kko, kklge, s=10,c=kkvals,cmap=cm,vmin=cmin, vmax=cmax)
    
def bisplinebkg():
    "Fit 2D spline to CTA acceptance. Based on Spline2D class"
    
    # --------- Get data --------
    
    #Get simulated acceptance ccube data
    ccube      = outdir+"ccube_bkg.fits"
    hdu_ccube  = fits.open(ccube)
    ccube_data = hdu_ccube[0].data
    ccube_tab  = cube2tab(ccube_data)
    
    #Get accetpance model from mcube
    mcube      = outdir+"mcube_bkg.fits"
    hdu_mcube  = fits.open(mcube)
    mcube_data = hdu_mcube[0].data
    mcube_min  = 1e-20
    mcube_data = (mcube_data>mcube_min)*mcube_data+(mcube_data<=mcube_min)*mcube_min      #Set minimum to 1e-10
    mcube_tab  = cube2tab(mcube_data)
    
    #  --------- Histogram maps for this binning --------
    ccube_h2 = histcube(ccube_tab,oedges, lgeedges)
    mcube_h2 = histcube(mcube_tab,oedges, lgeedges)
    print "Total Nr of rows in cube tables",len(mcube_tab)
    
    # --------- Setup spline --------
    kpoints = np.dstack([kko.flatten(), kklge.flatten()])[0]
    points   = np.dstack([oo.flatten(), lgee.flatten()])[0]
    gridvals = griddata(points, np.log10(mcube_h2).flatten(), kpoints, method='nearest')
    kkvals   = np.power(10,gridvals)
    
    print "* Spline knot starting values:\n",kkvals
    print "* Spline knot x axis:\n",ko
    print "* Spline knot y axis:\n",klge
    
    #Build spline
    spl = Spline2Dnp.Spline2D(ko,klge,logspline = lgspline)
    spl.build(kkvals)
    lglike0 = spl.logLike(spl.knot_vals,ccube_tab["offset"],ccube_tab["lge"],lgspline,ccube_tab["cts"])
    print "* Knots from model logL:", lglike0
    
    # Randomize starting values before fitting
    if fitspline:
        randomnrs = 1+(np.random.rand(*kkvals.shape)*0.4 - 0.2)
        spl.build(randomnrs*kkvals)
    
    lglikestart = spl.logLike(spl.knot_vals,ccube_tab["offset"],ccube_tab["lge"],lgspline,ccube_tab["cts"]) 
    print "* Fit start logL:", lglikestart,",dlogL0:",lglikestart-lglike0
    
    # Add background knots that don't vary in the fit
    if bknots:
        #            omin knots           +   lgemin knots
        bknot_oo   = klge.flatten()*0.-0.3
        bknot_lgee = klge.flatten()
        bkpoints   = np.dstack([bknot_oo,bknot_lgee])[0]
        bgridvals  = griddata(points, np.log10(mcube_h2).flatten(), bkpoints, method='nearest')
        bkkvals    = np.power(10,bgridvals)
        spl.addbknots("omin",bknot_oo,bknot_lgee,bkkvals)
    
    
    #print spl
    #spl.printknots()
    #spl.draw()
    
    #Fit spline
    if fitspline:
        res = spl.fit(ccube_tab["offset"],ccube_tab["lge"],ccube_tab["cts"],\
                      method='L-BFGS-B',tol=1e-3,options={'eps': 1e-3,'disp':True})
        print "Fit result:",res
        lglikefit = spl.logLike(spl.knot_vals,ccube_tab["offset"],ccube_tab["lge"],lgspline,ccube_tab["cts"])
        print "* Fit logL:",lglikefit,",dlogLstart:",lglikefit-lglikestart,",dlogL0:",lglikefit-lglike0
        
        
    # ---------  Plotresults  2D --------- 
    fig = plt.figure(figsize=(12, 10))
    cmin= min(np.log10(mcube_h2).flatten())
    cmax= max(np.log10(mcube_h2).flatten())
    
    plt.subplot(321,title="ccube")
    plotpanel(oo, lgee, np.log10(ccube_h2),cmin,cmax,kko, kklge,spl.knot_vals)
    
    plt.subplot(322,title="mcube")
    plotpanel(oo, lgee, np.log10(mcube_h2),cmin,cmax,kko, kklge,spl.knot_vals)
    
    plt.subplot(323,title="spline")
    spline_h2 = spl.eval(oo, lgee)
    plotpanel(oo, lgee, np.log10(spline_h2),cmin,cmax,kko, kklge,spl.knot_vals) #spline_h2.min(),spline_h2.max()
    
    plt.subplot(324,title="mcube interpolation")
    intepvals = griddata(points, np.log10(mcube_h2).flatten(), points, method='nearest').reshape(oo.shape)
    plotpanel(oo, lgee, intepvals,cmin,cmax,kko, kklge,spl.knot_vals)
    if bknots:
        plt.scatter(spl.bknots["omin"]["xx"], spl.bknots["omin"]["yy"], s=10,\
                    c=spl.bknots["omin"]["vals"],cmap="nipy_spectral",vmin=cmin, vmax=cmax)
    
    plt.subplot(325,title="spline to model residual")
    spline_res = (spline_h2-mcube_h2)/mcube_h2
    print "--------------vResidual info ---------------"
    print "Mean:",spline_res.mean(),",std:",np.std(spline_res)
    absmax = np.abs(spline_res).max()
    plotpanel(oo, lgee, spline_res,-absmax,absmax,kko, kklge,spl.knot_vals,cmap="seismic")
    
    plt.subplot(326,title="spline to model residual significance")
    spline_sig = (spline_h2-mcube_h2)/np.sqrt(mcube_h2)
    sigmax = np.abs(spline_sig).max()
    plotpanel(oo, lgee, spline_sig,-sigmax,sigmax,kko, kklge,spl.knot_vals,cmap="seismic")
    
    plt.tight_layout()
    
    # ---------  Plotresults  1D --------- 
    fig2 = plt.figure(figsize=(20, 10))
    cmap = plt.get_cmap("nipy_spectral")
    
    plt.subplot(121)
    cc   = [cmap(i) for i in np.linspace(0, 1, len(lgecenter))]
    for ebin in range(0,len(lgecenter),4):
        plt.plot(ocenter,np.log10(ccube_h2[ebin]),color=cc[ebin], linestyle='dashed')
        plt.plot(ocenter,np.log10(mcube_h2[ebin]),color=cc[ebin], linestyle='dotted')
        plt.plot(ocenter,np.log10(spline_h2[ebin]),color=cc[ebin],label=str(lgecenter[ebin]))
    plt.xlabel("Offset [deg]",fontsize=labelsize)
    plt.ylabel("lg(Counts per bin)",fontsize=labelsize)
    plt.legend(title="lg(Energy/TeV)")
    for o in ko:
        plt.axvline(x=o, linewidth=0.5, color='k',linestyle="--")
    
    plt.subplot(122)
    cco   = [cmap(i) for i in np.linspace(0, 1, len(ocenter))]
    for obin in range(0,len(ocenter),15):
        plt.plot(lgecenter,np.log10(ccube_h2[:,obin]),color=cco[obin], linestyle='dashed')
        plt.plot(lgecenter,np.log10(mcube_h2[:,obin]),color=cco[obin], linestyle='dotted')
        plt.plot(lgecenter,np.log10(spline_h2[:,obin]),color=cco[obin],label=str(ocenter[obin]))
    for lge in klge:
        plt.axvline(x=lge, linewidth=0.5, color='k',linestyle="--")
    plt.legend(title="Offset [deg]")
    plt.xlabel("lg(Energy/TeV)",fontsize=labelsize)
    plt.ylabel("lg(Counts)",fontsize=labelsize)
    plt.tight_layout()
    
    fig.savefig(outdir+"/AcceptanceFit_2Dplots.png")
    fig2.savefig(outdir+"/AcceptanceFit_1Dplots.png")
    
    #--------------------
    #3D plots
    #--------------------
    fig3 = plt.figure(figsize=(14,10))
    
    ax   = fig3.add_subplot(2, 2, 1, projection='3d')
    surf = plot3d(ax,oo, lgee, np.log10(spline_h2),'log10(Spline)',cmin,cmax)
    ax.scatter(kko, kklge,np.log10(kkvals),c="k")
    fig3.colorbar(surf, shrink=0.5, aspect=7)
    
    ax2   = fig3.add_subplot(2, 2, 2, projection='3d')
    surf2 = plot3d(ax2,oo, lgee,spline_res,'Residual to model',-absmax,absmax,"seismic")
    fig3.colorbar(surf2, shrink=0.5, aspect=7)
    ax2.scatter(kko, kklge,-absmax,c="k")
    ax2.scatter(kko, kklge,absmax,c="k")
    
    ax3   = fig3.add_subplot(2, 2, 3, projection='3d')
    surf3 = plot3d(ax3,oo, lgee,np.log10(ccube_h2),'log10(Data)',cmin,cmax)
    fig3.colorbar(surf3, shrink=0.5, aspect=7)
    ax3.scatter(kko, kklge,np.log10(kkvals),c="k")
    
    ax4   = fig3.add_subplot(2, 2, 4, projection='3d')
    surf4 = plot3d(ax4,oo, lgee,np.log10(mcube_h2),'log10(Model)',cmin,cmax)
    fig3.colorbar(surf4, shrink=0.5, aspect=7)
    ax4.scatter(kko, kklge,np.log10(kkvals),c="k")
    
    plt.tight_layout()
    
    plt.show()

def plot3d(ax,xx,yy,zz,zlabel,cmin,cmax,cmap="nipy_spectral"):
    surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,\
                    cmap=plt.cm.get_cmap(cmap), linewidth=1,\
                    antialiased=False,vmin=cmin, vmax=cmax)
    ax.set_xlabel('Offset [deg]')
    ax.set_ylabel('log(Energy)')
    ax.set_zlabel(zlabel)
    ax.view_init(30, 30)
    return surf
    
    
            
if __name__ == "__main__":
    
    #sim("cCrab")        # Simulate  point source with0.01 flux of Crab nebula
    #sim("bkg")          # Simulate CTA acceptance
    #show()              # Show the acceptance and shource counts
    bisplinebkg()        # Fit acceptance and show fit results

