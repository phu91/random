
import numpy as np
import sys, math
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib.font_manager import FontProperties
#import dask.dataframe as dd
from matplotlib import rc
from tqdm import tqdm

#### FUNCTION ####

def cal_2d(x,y,temp,R,pmf_max):
    H, xedges, yedges = np.histogram2d(x,y,density=True,range=[[0,x.max()],[0,y.max()]])
    stepx = xedges[1]-xedges[0]
    stepy = yedges[1]-yedges[0]
    xx, yy = np.mgrid[xedges.min():xedges.max():stepx,yedges.min():yedges.max():stepy]
    pos = np.dstack((xx, yy))
    pmax = 0
    for i in H:
        p = i.sum()
        if p >=pmax:
            pmax = p
    print("Found pmax = ",pmax)

    for i in range(len(H)):
        for j in range(len(H.T)):
            if H[i,j]!=0:
                H[i,j]=-R*temp*np.log(H[i,j]/pmax)
            else:
                H[i,j]=pmf_max
    return pos,H

### I/O ###
df1 = pd.read_csv("input.dat",header=0,delim_whitespace=True)
x='DISTANCE'
y1='ANGLE'

df = pd.DataFrame.from_dict(data=data)

X = df['x']
Y1 = df['y']

TEMP = 300
R = 0.002 # kcal/mol
PMF_max = 6 # kcal / mol
print("Finished reading input")
# print(df.head())

ofile1 = open("output.dat", "w")

ofile1.write(x+"  "+y1+"   pmf"+'\n')
ofile1.write(x+"  "+y2+"   pmf"+'\n')
ofile1.write(x+"  "+y3+"   pmf"+'\n')

out = '/home/phu/Desktop/self-assemble/12-mols/cubicBox/2-1DP/test2-SYSTEM-II/pi-pi/2D-PMF-test2-100ns.png'

pos1,H1 = cal_2d(X,Y1,TEMP,R,PMF_max)

fig , ax = plt.subplots(1,3)
plot1 = ax[0].contourf(pos1[:,:,0],pos1[:,:,1],H1,10,cmap="viridis_r",vmin=H1.min(),vmax=H1.max(),extend='max')

# ylabels = [y1,y2,y3]

# for i in range(3):
#     ax[i].set(xlabel=x.upper(),ylabel=ylabels[i].upper())
#     ax[i].xaxis.set_minor_locator(AutoMinorLocator())
#     ax[i].yaxis.set_minor_locator(AutoMinorLocator()) 
       
# cbar1 = fig.colorbar(plot1,ax=ax[0],shrink=0.9,spacing='proportional')
# cbar2 = fig.colorbar(plot2,ax=ax[1],shrink=0.9,spacing='proportional')
# cbar3 = fig.colorbar(plot3,ax=ax[2],shrink=0.9,spacing='proportional')

# cbar1.set_label('kcal / mol',rotation=270,labelpad=15)
# cbar2.set_label('kcal / mol',rotation=270,labelpad=15)
# cbar3.set_label('kcal / mol',rotation=270,labelpad=15)

# cbar1.minorticks_on()
# cbar2.minorticks_on()
# cbar3.minorticks_on()

# plt.tight_layout(pad=0.05)
# # plt.show()
# # 
# plt.rcParams['ps.useafm'] = True
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rcParams['pdf.fonttype'] = 42
# plt.gcf().set_size_inches(7.,3.5)
# 
### SAVE FIGURE ### 
# plt.savefig(out,dpi=300,orientation=None, papertype=None, format=None,transparent=True, bbox_inches=None, pad_inches=None,metadata=None)
plt.show()
