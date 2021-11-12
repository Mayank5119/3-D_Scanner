import pandas as pd
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% Processing variables
# ; %Upper limit -- raw scan value only scanning "air"
maxDistance=20
# ; %Lower limit -- raw scan value error: reporting negative reading
minDistance=0
# ; %Offset radius threshold around 0
midThreshUpper=0.5
# ; %Offset radius threshold around 0
midThreshLower=-midThreshUpper
# ; %Window size for average filter to clean up mesh
windowSize=3
# ; %Interpolation resolution, i.e. keep every interRes-th row
interpRes=1
# ;%9.2; %[cm] - Distance from scanner to center of turntable
centerDistance=10.3
zDelta=0.1

# ; %Load text file from SD Card
# writing to file
file1 = open('farmer.txt',"r+")
lines = file1.readlines()
for l in range(len(lines)):
    lines[l] = float(lines[l].rstrip())
rawData = lines
rawData = pd.DataFrame(lines)
rawData.columns = ["values"]
# print(rawData)
# print(lines)
# ; %Remove erroneous scans from raw data
for i in range(len(lines)):
    if lines[i]<0:
        lines[i] = 0
rawData[rawData.values<0] = 0
#################### some in b/w values are 9999 ###################################
# ; %Find indeces of '9999' delimiter in text file, indicating end of z-height
indeces = rawData.index[rawData["values"] == 9999].tolist()
# print(indeces)
# % Arrange into matrix, where each row corresponds to one z-height.
lines2d=[]
s = 0
for e in indeces:
    lines2d.append(lines[s:e])
    s = e+1
# print(lines2d)
r = pd.DataFrame(lines2d)
# print(rawData)
# ; %Offset scan so that distance is with respect to turntable center of rotation
r=centerDistance-r
# print(rawData)
# # ; %Remove scan values greater than maxDistance;
r[r>maxDistance]=None
# # ; %Remove scan values less than minDistance
r[r<minDistance]=None
# # print(rawData)
#
# # %Remove scan values around 0
# midThreshIdx=rawData.where((rawData.values>midThreshLower) & (rawData.values<midThreshUpper))
r[(r>midThreshLower) & (r<midThreshUpper)] = None
# print(r)

# % Create theta matrix with the same size as r -- each column in r corresponds to specific orientation
# %theta=0:360/size(r,2):360;
theta=[[round(x, 2) for x in np.arange(360.0, 0, -360/r.shape[1])] for y in range(r.shape[0])]
theta = pd.DataFrame(theta)

# # ; %Convert to radians
theta=theta*np.pi/180
# print(theta)

# % Create z-height array where each row corresponds to one z-height
z=[[round(y*zDelta, 2) for x in range(0, r.shape[1])] for y in range(r.shape[0])]
z = pd.DataFrame(z)
# print(z)
# rawData["z"] = rawData.index*zDelta
# print(rawData)

# %Replace NaN values in r, theta with avg values
r = (r.fillna(method='ffill') +
                     r.fillna(method='bfill'))/2
theta = (theta.fillna(method='ffill') +
                      theta.fillna(method='bfill'))/2
z = (z.fillna(method='ffill') +
                      z.fillna(method='bfill'))/2

def pol2cartx(rho, theta):
    x = rho*np.cos(theta)
    return x
def pol2carty(rho, theta):
    y = rho*np.sin(theta)
    return y

# # ; %Convert to cartesian coordinates
x = pol2cartx(r, theta)
y = pol2carty(r, theta)
# print(x)
# print(y)

# %Resample array based on desired mesh resolution
interpIdx=[x for x in range(1, x.shape[0], interpRes)]
# print(interpIdx)
xInterp=x.loc[interpIdx,:]
yInterp=y.loc[interpIdx,:]
zInterp=z.loc[interpIdx,:]

# %Smoothe data to eliminate more noise

xInterp = cv2.blur(xInterp.to_numpy(),(windowSize,windowSize))
yInterp = cv2.blur(yInterp.to_numpy(),(windowSize,windowSize))
zInterp = zInterp.to_numpy()
# print(xInterp)
# print(yInterp)
# pd.DataFrame

# %Force scan to wrap by duplicating first column values at end of arrays
xInterp=np.hstack((xInterp,np.reshape(xInterp[:, 0], (xInterp.shape[0], 1))))
yInterp=np.hstack((yInterp,np.reshape(yInterp[:, 0], (yInterp.shape[0], 1))))
zInterp=np.hstack((zInterp,np.reshape(zInterp[:, 0], (zInterp.shape[0], 1))))
# yInterp.loc[:, len(yInterp.columns)]=yInterp.loc[:, 0]
# zInterp.loc[:, len(zInterp.columns)]=zInterp.loc[:, 0]
# print(xInterp)
# print(yInterp)

# %Add top to close shape
xTop=np.mean(xInterp[-1,:])
yTop=np.mean(yInterp[-1,:])
xInterp=np.vstack((xInterp,[xTop for i in range(len(xInterp[1]))]))
yInterp=np.vstack((yInterp,[yTop for i in range(len(yInterp[1]))]))
zInterp = np.vstack( (zInterp , [(zInterp[-1,1]-zInterp[-2,1]+zInterp[-1,1]) for i in range(len(yInterp[1]))]))


print(xInterp.shape)
print(yInterp.shape)
print(zInterp.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xInterp, yInterp, zInterp, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(60, 35)
# plt.show()
xInterp[np.isnan(xInterp)] = 0
yInterp[np.isnan(yInterp)] = 0
zInterp[np.isnan(zInterp)] = 0

import surf2stl
surf2stl.surf2stl("farmer.stl", xInterp, yInterp, zInterp)