import pdb
import itertools
import os
import glob
import datetime
import pickle

import matplotlib as M
import numpy as N
from netCDF4 import Dataset

from evac.stats.fi import FI
from evac.plot.heatmap import HeatMap

# PATHS etc
outdir = '/home/john.lawson/fracign/pyoutput'
compute = False
plot = True
ensdir = '/work1/wof/realtime/FCST'
# /work1/wof/realtime/FCST/20180601/2130/ENS_MEM_18
# wrfout_d01_2018-06-01_22:40:00
ncpus = 30

mrmsdirs = {2017:'/work1/skinnerp/MRMS_verif/mrms_cressman',
                2018:'/scratch/skinnerp/2018_newse_post/mrms'}
mrmscasedirs = {2017:glob.glob(os.path.join(mrmsdirs[2017],"2017*")),
                2018:glob.glob(os.path.join(mrmsdirs[2018],"2018*"))}
mrmscases = os.listdir(mrmsdirs[2017]) + os.listdir(mrmsdirs[2018])

fissdir = '/work/john.lawson/fracign/NEWSe_compare'

# 3 init times each, 1h, 2h, 3h forecasts
vrbl = "REFL_comp"
inittimes = (20,23,2)
fcstmins = (30,90,150)

newscasedirs = {2017:glob.glob(os.path.join(ensdir,"2017*")),
                2018:glob.glob(os.path.join(ensdir,"2018*"))}
newscases = os.listdir(ensdir)# newscasedirs[2017]) + os.listdir(newscasedirs[2018])

brokencases = ['20180518',]


# Find intersection of news-e and mrms cases
intersect_cases_raw = sorted(list(set(mrmscases) & set(newscases)))
# Get rid of any cases that don't have the three inittimes in there
# Why won't they all go away?! 20170517 still here!!
intersect_cases = []
for i in intersect_cases_raw:
    present = []
    for it in inittimes:
        check = "{:02d}00".format(it) in os.listdir(os.path.join(ensdir,i))
        present.append(check)
    if (False in present) or (i in brokencases):
        print("REMOVING",i)
        # intersect_cases.remove(i)
    else:
        intersect_cases.append(i)

ensnames = sorted(os.listdir('/work1/wof/realtime/FCST/20180601/2130/'))

# FUNCS
def get_wrfout_fname(t):
    fname = "wrfout_d01_{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:00".format(
                t.year, t.month, t.day, t.hour, t.minute)
    return fname

def get_mrms_fname(t):
    fmt = "%Y%m%d-%H%M00.nc"
    fname = datetime.datetime.strftime(t,fmt)
    return fname

def get_out_fname(casestr,initstr,fcstmin):
    fname = "FISS_{}_{}_{}min.pickle".format(
                casestr,initstr,fcstmin)
    return fname

# time-aware
windows = [1,3,5]
# scale-aware
neighs = [1,3,5,7,9]
# 10, 30, 50 dBZ threshs
threshs = [10,30,50]
# send five times, and we want the middle on
tidxs = [2,]


# COMPUTE 
if compute:
    for fcstmin in fcstmins:
        print("Forecast lead time: {} min".format(fcstmin))
        # for casedir in all_casedirs:
        for casestr in intersect_cases:
            # casestr = os.path.basename(casedir)
            casedir = os.path.join(ensdir,casestr)
            print("Case:",casestr)
            caseutc = datetime.datetime.strptime(casestr,"%Y%m%d")
            for inittime in inittimes:
                hrdelta = inittime if inittime > 12 else (inittime + 24)
                initutc = caseutc + datetime.timedelta(seconds=60*60*hrdelta)
                initstr = "{:02d}00".format(inittime)
                print("Initisation time:",initstr)

                xfs = None
                xa = None

                out_fname = get_out_fname(casestr,initstr,fcstmin)
                fracdump = os.path.join(fissdir,out_fname)

                if os.path.exists(fracdump):
                    print("Skipping the file that exists at:",fracdump)
                    continue

                for ne,mem in enumerate(ensnames):
                    print("Loading data for",mem)
                    memdir = os.path.join(casedir,initstr,mem)
                    # Load a block of 5 so time window can be computed
                    print("Looping through time window...")
                    for dn,diff in enumerate((-2,-1,0,1,2)):
                        validmin = fcstmin + (diff*5)
                        validtime = initutc + datetime.timedelta(seconds=60*fcstmin)

                        # Load forecast data
                        fname0 = get_wrfout_fname(validtime)
                        fpath0 = os.path.join(memdir,fname0)
                        nc0 = Dataset(fpath0)
                        data0 = N.max(nc0.variables["REFL_10CM"][0,:,:,:],axis=0)
                        data0[data0<0] = 0 
                        lats0 = nc0.variables["XLAT"][0,...]
                        lons0 = nc0.variables["XLONG"][0,...]
                        nlat0 = data0.shape[-2]
                        nlon0 = data0.shape[-1]
                        # Check min/max? -32? 0?

                        if xfs is None:
                            xfs = N.empty((18,5,nlat0,nlon0))
                        xfs[ne,dn,:,:] = data0

                        # Verification data (MRMS)
                        mrmsdir = mrmsdirs[int(caseutc.year)]
                        fname1 = get_mrms_fname(validtime)
                        fpath1 = os.path.join(mrmsdir,casestr,fname1)
                        nc1 = Dataset(fpath1)
                        data1 = nc1.variables["DZ_CRESSMAN"][:]
                        data1[data1 < 0] = 0
                        lats1 = nc1.variables["XLAT"][:]
                        lons1 = nc1.variables["XLON"][:]
                        nlat1 = data1.shape[-2]
                        nlon1 = data1.shape[-1]

                        # This is done in FI but just being careful.
                        assert nlon0 == nlon1
                        assert nlat1 == nlat0

                        if xa is None:
                            xa = N.empty((5,nlat1,nlon1))
                        xa[dn,:,:] = data1
                        # pdb.set_trace()
                # threshs = [10,]
                # neighs = [15,]
                # windows = [5]
                fracign = FI(xa=xa,xfs=xfs,thresholds=threshs,
                                neighborhoods=neighs,
                                temporal_windows=windows,
                                tidxs=tidxs,ncpus=ncpus)

                del lats0, lats1
                del xa,xfs
                del data0,data1
                nc0.close()
                nc1.close()
                # Save to disk
                # pdb.set_trace()
                with open(fracdump,'wb') as fd:
                    pickle.dump(fracign.results,fd)
                print("Saved to",fracdump)
                del fracign
                # pdb.set_trace()

scores = ("FISS","FI","UNC","REL","RES")
# windows, neighs, threshs
data_temp = N.empty([len(threshs),len(neighs),len(windows)])
DATA = {c:{fm:{th:{n:{w:{s: [] for s in scores} for w in windows} for n in neighs} 
            for th in threshs} for fm in fcstmins} for c in (0,1)}
if plot:
    for fcstmin in fcstmins:
        for casestr in intersect_cases:
            caseutc = datetime.datetime.strptime(casestr,"%Y%m%d")
            for inittime in inittimes:
                hrdelta = inittime if inittime > 12 else (inittime + 24)
                initutc = caseutc + datetime.timedelta(seconds=60*60*hrdelta)
                initstr = "{:02d}00".format(inittime)
                out_fname = get_out_fname(casestr,initstr,fcstmin)
                fracpick = os.path.join(fissdir,out_fname)
                with open(fracpick,'rb') as f:
                    data = pickle.load(f)
                
                for th,n,w,s in itertools.product(threshs,
                                    neighs,windows,scores):
                    if caseutc.year == 2017:
                        cidx = 0
                    elif caseutc.year == 2018:
                        cidx = 1
                    DATA[cidx][fcstmin][th][n][w][s].append(
                            data[th][n][w][2][s])
    # plot here
    plotstr = "bruteyears"
    neighlabels = [n*3 for n in neighs]
    for fcstmin, score in itertools.product(fcstmins,scores):
        #data_temp = [len(thresh),len(neighs),len(windows)]
        matrix0 = N.copy(data_temp)
        matrix1 = N.copy(data_temp)
        thidxs = range(len(threshs))
        nidxs = range(len(neighs))
        widxs = range(len(windows))
        for (nth,th),(nn,n),(nw,w) in itertools.product(enumerate(threshs),
                                enumerate(neighs),enumerate(windows)):
            matrix1[nth,nn,nw] = N.mean(DATA[1][fcstmin][th][n][w][score])
            matrix0[nth,nn,nw] = N.mean(DATA[0][fcstmin][th][n][w][score])
        for yearstr in ("2017","2018","diff"):
            for nw,window in enumerate(windows):
                scorestr = ''.join((score,yearstr))
                fname = "heatmap_{:03d}min_{}twindow_{}_{}.png".format(
                                    fcstmin,window,scorestr,plotstr)
                if yearstr == "2017":
                    mat = matrix0
                    div = False
                    if score == "FISS":
                        cm = M.cm.bwr
                    else:
                        cm = M.cm.Blues
                elif yearstr == '2018':
                    mat = matrix1
                    div = False
                    if score == "FISS":
                        cm = M.cm.bwr
                    else:
                        cm = M.cm.Reds
                elif yearstr == 'diff':
                    mat = matrix1-matrix0
                    div = True
                    cm = M.cm.bwr
                H = HeatMap(mat[:,:,nw],outdir=outdir,fname=fname,)
                H.plot(xlabels=neighlabels,ylabels=threshs,
                            annotate_values=True,cmap=cm,
                            xlabel="Neighbourhood diameter (km)",
                            ylabel="Threshold (dBZ)",diverging=div)

    # For all days
    # For 3x10 severe days in each 
    # For supercells
    # For MCSs


    # For each pickle file, load into different arrays for score and year
    # Then do averages over each year, and subtract to find improvement

    # Do matrix as in Duc paper
    # ?-axis is spatial neighbourhood
    # ?-axis is dBZ threshold
    # One plot for no time window, and another for 3-time window
    # Do stats to colour z-axis (and annotate value of improvement):
    # --> 2018-2017 FISS 
    # --> 2018-2017 FI
    # --> 2018-2017 REL
    # --> 2018-2017 RES

    # What about 2:1 weighing for RES v REL? Need to re-calc a FIVS (value score).
    # Heat
