import numpy as np
#import math
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from read3 import read_corenn_file
from kappalin import initPar, kappalinear


# -------------------------
# 1. Load training data
# -------------------------
#data = np.loadtxt("training.dat")
data = np.array(read_corenn_file("data.txt"))
nprm=4

#print(data[:,0])
ifitke=2 # gpr of momentum(1) or energy(2)
faclin = 1.0 # separate approx linear kappa-dep(1.0) or not(0.0)

X = data[:,0:nprm]   # m, rnc, ab, ann1
#ye = np.zeros((nres, nprm))
ye= data[:,4]*1.0     # last column (target)
#yk=ye[:]
yk= np.sqrt(2.0*ye[:]*X[:,0]/(41.47*(1.0+X[:,0])))

yel = np.zeros(len(ye))

initPar()
rnn=1.65
kappa=0.0
Ener=0.0
for i in range(len(ye)):
  mc = X[i,0]
  rnc=X[i,1]
  ab1=X[i,2]
  ann1=X[i,3]
  kappa, Ener = kappalinear(mc, rnn, rnc, ab1, ann1)
#  print (mc,rnc,ab,kappa, Ener)
  yk[i]=yk[i]-kappa*faclin
#  yk[i]=yk[i]/kappa - 1.0
  yel[i] = Ener
  ye[i]=ye[i]-Ener*faclin
#  ye[i]=ye[i]/Ener - 1.0


X[:,0] = 1.0/X[:,0] #1/m
X[:,2] = X[:,1]*X[:,2] #rnc/ab
X[:,1] = rnn/X[:,1] #rnn/rnc
X[:,3] = rnn*X[:,3] #rnn/ann

with open("train.dat", "w") as f:
    for i in range(len(ye)):
        f.write(
                f"{X[i,0]:9.5f}"
                f"{X[i,1]:9.5f}"
                f"{X[i,2]:13.5e}"
                f"{X[i,3]:13.5e}"
                f"{yel[i]:13.5e}"
                f"{ye[i]:13.5e}"
                f"{data[i,4]:13.5e}\n")
f.close

# -------------------------
# 2. Define kernel
# -------------------------
kernel = (
    C(1.0, (1e-3, 1e3)) *
    RBF(length_scale=np.ones(nprm), length_scale_bounds=(1e-2, 1e2)) +
    WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
)

# -------------------------
# 3. Create & train model
# -------------------------
gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10
)


if (ifitke==1 or ifitke==11):
  gpr.fit(X, yk)
else:
  gpr.fit(X, ye)


print("Optimized kernel:")
print(gpr.kernel_)

# -------------------------
# 4. Predict new point
# -------------------------

mc= 10.0
rnn, rnc, ann = 1.65, 3.3, 1.0e99 #18.953
rnc = 1.25
if (rnc < 1.5):
  rnc = rnc*mc**(1.0/3.0)
ann1 = 0.0
ann1 = -1.0/18.953
rnn, ann1 = 1.8, -1.0/18.90 # physical

mc, rnc = 17.0, 5.15
#mc, rnc = 9.0, 3.29
#mc, rnc = 18.0, 2.88
#mc, rnc = 10.0, 2.58
#mc, rnc =  14.0, 2.94
#mc, rnc = 22.0, 6.09
#mc, rnc = 30.0, 3.16
mu2 = mc * 2.0 / (41.47*(mc + 1.0)) 
refac = (1.65/rnn)**2

nres=1000
X_test = np.zeros((nres, nprm))
X_test[0,2] = 1.0e-99 #  unitarity
X_test[1:nres-1, 2] = np.linspace(-0.47, 0.0, nres-2)  # vary rnc/ab
X_test[nres-1, 2] = 2.5
X_test[:, 0] = 1.0/mc
X_test[:, 1] = rnn/rnc
if  nprm==4:
  X_test[:, 3] = rnn*ann1



y_mean, y_std = gpr.predict(X_test, return_std=True)

yex = np.zeros((nres))
ykx = np.zeros((nres))
yel = np.zeros((nres))
ykl = np.zeros((nres))

for i in range(nres):
  ab1=X_test[i,2]/rnc
  kappa, Ener = kappalinear(mc, rnn, rnc, ab1, ann1)
  ykl[i]=kappa
  yel[i]=Ener

if ifitke==1:
  ykx=faclin*ykl[:] + y_mean[:] 
  yex=ykx[:]*np.abs(ykx[:])/mu2
else:
  yex=faclin*yel[:] + y_mean[:]*refac
#  yex=faclin*yel[:] + (y_mean[:]+y_std[:])*refac
  ykx= np.sqrt(mu2*np.abs(yex[:]))

i=nres-1 # last point
yex[i]=yel[i]
ykx[i]=ykl[i]


e3u = yex[0]
s0=0.998306 + 0.141194 * np.exp(-3.39135 / mc)


with open("output19b-rncb.dat", "w") as f:
    for i in range(nres):
        if (i==0 or ykx[i]<0.0 or yex[i]<0.0):
          continue
        if (ykx[i]<ykx[i-1] or yex[i]<yex[i-1]):
          continue

        ab1 = X_test[i,2] / rnc
        e2 = ab1 ** 2 / mu2
        ab = 1.0 / ab1

        ksi = -np.atan(ab * ykx[i]) / np.pi - (1.0 if ab < 0.0 else 0.0)
        e3 = yex[i]
        delta = s0 * np.log((e3 + e2) / e3u)

        f.write(
                f"{e2:13.5e}"
                f"{yex[i]:13.5e}"
                f"{ykx[i]:13.5e}"
                f"{ab:13.5e}"
                f"{yel[i]:13.5e}"
                f"{ykl[i]:13.5e}"
                f"{ksi:13.5e}"
                f"{delta:13.5e}\n"
            )

#f"{X_test[i,2]:9.4f}{y_mean[i]:9.4f}\n")
f.close

#print("Prediction:", y_pred)
#print("Uncertainty (1σ):", y_std)

# Plot
plt.plot(X_test[1:,2], y_mean[1:], label="GPR mean")
plt.plot(X_test[1:,2], yex[1:], label="E")
plt.plot(X_test[1:,2], yel[1:], label="EL")
plt.fill_between(
    X_test[1:,2].ravel(),
    yex[1:] - 2*y_std[1:],
    yex[1:] + 2*y_std[1:],
    alpha=0.3,
    label="±2σ"
)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

nres=7
X_test[0:7,0] = [18.0, 14.0, 10.0, 22.0, 30.0, 9.0, 17.0]
X_test[0:7,1] = X_test[0:7,0]**(1.0/3.0)* 1.25 #rnc
X_test[0:7,1] = [2.88, 2.94, 2.58, 6.09, 3.16, 3.29, 5.15] #rnc-b

#X_test[0:7,1] = ((X_test[0:7,0]+2.0)*0.11125 + 0.08)/0.7 #rnc pagal AK
#X_test[0:3,1] = [3.38, 2.62, 2.05]
X_test[0:7,2] = 2.0*X_test[0:7,0]/(41.47*(1.0+X_test[0:7,0])) # 2mu
X_test[0:5,2] = X_test[0:5,2]*np.array([0.58,1.218,0.504,2.73,2.312]) # 2muB2
X_test[0:5,2] = np.sqrt(X_test[0:5,2]) # 1/aB
X_test[5:7,2] = -1.0/np.array([45.87,102.37])

print (X_test[0:7,1])

for i in range(nres):
  ab1=X_test[i,2]/rnc
  kappa, Ener = kappalinear(X_test[i,0], rnn, X_test[i,1], X_test[i,2], ann1)
  ykl[i]=kappa
  yel[i]=Ener
  print (X_test[i,1], X_test[i,2], kappa, Ener)

X_test[0:7,0] = 1.0/X_test[0:7,0] # 1/m
X_test[0:7,2] = X_test[0:7,1]*X_test[0:7,2] #rnc/aB
X_test[0:7,1] = rnn/X_test[0:7,1] #rnn/rnc
X_test[0:7,3] = rnn*ann1

y_mean[0:7], y_std[0:7] = gpr.predict(X_test[0:7,:nprm], return_std=True)

yex[0:7]=faclin*yel[0:7] + y_mean[0:7]*refac
print (yel[0:7])
print (yex[0:7])

