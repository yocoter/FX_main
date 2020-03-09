import os
import sys
import argparse
import urllib.request

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pylab import *

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments


def Normalization(x):
    x_min = np.min(x)
    xd = x - x_min
    xd_max = np.max(xd)
    XD = xd / xd_max
    return XD, xd_max, x_min

def Normalization__(x, xd_max, x_min):
    xd = x - x_min
    XD = xd / xd_max
    return XD

def ReNormalization(XD, xd_max, x_min):
    xd = XD * xd_max
    x = xd + x_min
    return x

def DesignMatrix(x, M=200, sigma=1):
    mu = np.linspace(0, 1, M-1)
    
    basis = []
    b = np.ones([len(x)])
    basis.append(b)

    for i in range(M-1):
        b = 1 / (2*np.pi*sigma**2)**0.5 * np.exp(-0.5 / sigma**2 * (x - mu[i])**2)
        basis.append(b)
    
    basis = np.array(basis)
    return basis.T


# --* Result dir *--
filedir = "Result"
if not os.path.exists(filedir):
    os.mkdir(filedir)


# --* My account *--
accountID = ""
access_token = ""
api = API(access_token=access_token)


# --* Get data *--
params = {
    "count": 3123,
    "granularity": "H2"
}

# H1:3123 -> 6M
# H2:3123 -> 1y

# 米日: "USD_JPY"
# 欧豪: "EUR_AUD"

name = "USD_JPY"
name_ = name.replace("_", "")

r = instruments.InstrumentsCandles(instrument=name, params=params)
api.request(r)

data = []
for raw in r.response["candles"]:
    data.append([raw["time"], raw["volume"], raw["mid"]["o"],
                 raw["mid"]["h"], raw["mid"]["l"], raw["mid"]["c"]])

df = pd.DataFrame(data)
df.columns = ["time", "volume", "open", "high", "low", "close"]
df = df.set_index("time")

df.index = pd.to_datetime(df.index)
print(df)


data = np.array(df, dtype="float")
volume = data[:,0]
p_open = data[:,1]
p_high = data[:,2]
p_low = data[:,3]
p_close = data[:,4]

target = p_close
n_max = len(target)

fac_s = 20
p_shorts = []
for i in range(n_max-fac_s):
    p_short = np.mean(target[i:i+fac_s])
    p_shorts.append(p_short)

fac_l = 100
p_longs = []
for i in range(n_max-fac_l):
    p_long = np.mean(target[i:i+fac_l])
    p_longs.append(p_long)

p_shorts = np.array(p_shorts)
p_longs = np.array(p_longs)

time = np.linspace(0, 1, n_max)
time_s = time[fac_s:n_max]
time_l = time[fac_l:n_max]

target, td_max, t_min = Normalization(target)
p_shorts = Normalization__(p_shorts, td_max, t_min)
p_longs = Normalization__(p_longs, td_max, t_min)


# --* AIC *--
sigma = 1e-2
alpha = 5e-3

m_0 = 100
m_max = 200
m = np.arange(m_0, m_max+1, 1)

AIC_min = 0
best_M = m_0
AICs = []
for i in range(len(m)):
    phi = DesignMatrix(time, M=m[i], sigma=sigma)
    pseudo_inverse_matrix = np.linalg.inv(alpha*np.identity(m[i]) + phi.T.dot(phi)).dot(phi.T)
    w = pseudo_inverse_matrix.dot(target)
    p_reg = phi.dot(w)
    p_reg = p_reg.reshape([len(p_reg)])

    SSE = np.sum(np.square(target - p_reg))
    AIC = len(target)*(np.log(2*np.pi*SSE/len(target)) + 1) + 2*(m[i] + 2)
    AICs.append(AIC)
    
    if AIC_min > AIC:
        AIC_min = AIC
        best_M = m[i]

AICs = np.array(AICs)
print("Best: M=%d, AIC=%f" %(best_M, AIC_min))

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(m, AICs, label="close", linestyle="-")
plt.scatter(best_M, AIC_min, label="close", marker="o", color="r")
plt.xlabel("M", fontsize=15)
plt.ylabel("AIC", fontsize=15)
plt.grid()
plt.savefig(filedir + os.sep + "AIC.png")


# --* Best Model *--
phi = DesignMatrix(time, M=best_M, sigma=sigma)
pseudo_inverse_matrix = np.linalg.inv(alpha*np.identity(best_M) + phi.T.dot(phi)).dot(phi.T)
w = pseudo_inverse_matrix.dot(target)
p_reg = phi.dot(w)
p_reg = p_reg.reshape([len(p_reg)])

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(time, target, label="close", linestyle="-")
plt.plot(time, p_reg, label="close", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "Best_regression.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(time, p_reg, label="close", linestyle="-")
plt.plot(time_s, p_shorts, label="short", linestyle="-")
plt.plot(time_l, p_longs, label="long", linestyle="-")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "train_data.png")


p_true = p_true[fac_l:n_max]
p_reg = p_reg[fac_l:n_max]
p_shorts = p_shorts[fac_l-fac_s:n_max]

time = time[fac_l:n_max]
time_s = time_s[fac_l-fac_s:n_max]

print(len(p_reg), len(p_shorts), len(p_longs))
print(len(time), len(time_s), len(time_l))
print(np.max(time), np.max(time_s), np.max(time_l))


ch = 40
pit = ch - 1
dim = 3
train_ratio = 0.9
n = len(p_reg)-pit
n_train = int(n * train_ratio)

in_reg = []
for i in range(len(p_reg)-pit):
    t = p_reg[i:i+ch]
    in_reg.append(t)

in_shorts = []
for i in range(len(p_shorts)-pit):
    s = p_shorts[i:i+ch]
    in_shorts.append(s)

in_longs = []
for i in range(len(p_longs)-pit):
    l = p_longs[i:i+ch]
    in_longs.append(l)

reg_ = np.array(in_reg)
shorts_ = np.array(in_shorts)
longs_ = np.array(in_longs)

data = np.zeros([len(p_longs)-pit, ch, dim])
data[:,:,0] = reg_
data[:,:,1] = shorts_
data[:,:,2] = longs_

T_train = time[pit:n_train+pit]
T_test = time[n_train+pit:len(time)]

X = data[:,0:pit,:]
Y = data[:,pit,:]
X_train = X[0:n_train,:,:]
Y_train = Y[0:n_train,:]
X_test = X[n_train:len(p_reg)-pit,:,:]
Y_test = Y[n_train:len(p_reg)-pit,:]

print(np.shape(X), np.shape(Y))
print(np.shape(X_train), np.shape(Y_train))
print(np.shape(X_test), np.shape(Y_test))


# --* RNN main *--
len_sequence = pit
n_in = 3
n_hidden = 300
n_out = 3

batch_size = 100
epochs = 300
val_split = 0.2
patience = 70

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, len_sequence, n_in), return_sequences=False))
model.add(Dense(n_out))
model.add(Activation("linear"))
model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")

cb_es = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
cb_cp = ModelCheckpoint(filepath="weights.hdf5", monitor="val_loss", mode="auto", save_best_only="True")

hist = model.fit(X_train, Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_split=val_split,
                 callbacks=[cb_es, cb_cp])

loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
Epoch = np.arange(1, len(loss)+1, 1)

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(Epoch, loss, label="loss")
plt.plot(Epoch, val_loss, label="val loss")
plt.yscale("log")
plt.xlabel("eposh", fontsize=15)
plt.ylabel("loss", fontsize=15)
plt.savefig(filedir + os.sep + "loss.png")

model.load_weights("weights.hdf5")
Y_pred = model.predict(X_test)

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(T_train, Y_train[:,0], label="close", linestyle="-")
plt.plot(T_test, Y_test[:,0], label="close", linestyle="-")
plt.plot(T_test, Y_pred[:,0], label="close", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "test_reg.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(T_train, Y_train[:,1], label="short", linestyle="-")
plt.plot(T_test, Y_test[:,1], label="short", linestyle="-")
plt.plot(T_test, Y_pred[:,1], label="short", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "test_short.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(T_train, Y_train[:,2], label="long", linestyle="-")
plt.plot(T_test, Y_test[:,2], label="long", linestyle="-")
plt.plot(T_test, Y_pred[:,2], label="long", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "test_long.png")


# --* Predict unknown data *--
x_test = X_test[0,:,:]
x_test = x_test.reshape([1,pit,dim])

y_preds = []
for i in range(n-n_train):
    y_pred = model.predict(x_test)
    y_preds.append(y_pred)
    y_pred = y_pred.reshape([1,1,dim])
    x_test = np.append(x_test, y_pred, axis=1)
    x_test = np.delete(x_test, obj=0, axis=1)
y_preds = np.array(y_preds)

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(T_train, Y_train[:,0], label="close", linestyle="-")
plt.plot(T_test, Y_test[:,0], label="close", linestyle="-")
plt.plot(T_test, y_preds[:,:,0], label="close", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "pred_reg.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(T_train, Y_train[:,1], label="short", linestyle="-")
plt.plot(T_test, Y_test[:,1], label="short", linestyle="-")
plt.plot(T_test, y_preds[:,:,1], label="short", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "pred_short.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(T_train, Y_train[:,2], label="long", linestyle="-")
plt.plot(T_test, Y_test[:,2], label="long", linestyle="-")
plt.plot(T_test, y_preds[:,:,2], label="long", linestyle="--")
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.04))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel("days", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid()
plt.savefig(filedir + os.sep + "pred_long.png")

d = 12
h = 2
times = np.arange(h, d*h+h, h)

Y_test = ReNormalization(Y_test, td_max, t_min)
Y_pred = ReNormalization(Y_pred, td_max, t_min)
y_preds = ReNormalization(y_preds, td_max, t_min)

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(times, Y_test[0:d,0], marker="^", label="True")
plt.plot(times, Y_pred[:,0][0:d], marker="^", label="Test")
plt.plot(times, y_preds[0:d,:,0], marker="^", label="Pred")
plt.xlim(0, d*h+h)
ax.xaxis.set_major_locator(MultipleLocator(4))
plt.xlabel("hours", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.grid()
plt.savefig(filedir + os.sep + "pred_reg_ST.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(times, Y_test[0:d,1], marker="^", label="True")
plt.plot(times, Y_pred[:,1][0:d], marker="^", label="Test")
plt.plot(times, y_preds[0:d,:,1], marker="^", label="Pred")
plt.xlim(0, d*h+h)
ax.xaxis.set_major_locator(MultipleLocator(4))
plt.xlabel("hours", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.grid()
plt.savefig(filedir + os.sep + "pred_short_ST.png")

fig = plt.figure(figsize=(10,6), dpi=100)
ax = fig.add_subplot(111)
plt.plot(times, Y_test[0:d,2], marker="^", label="True")
plt.plot(times, Y_pred[:,2][0:d], marker="^", label="Test")
plt.plot(times, y_preds[0:d,:,2], marker="^", label="Pred")
plt.xlim(0, d*h+h)
ax.xaxis.set_major_locator(MultipleLocator(4))
plt.xlabel("hours", fontsize=15)
plt.ylabel(name_, fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.grid()
plt.savefig(filedir + os.sep + "pred_long_ST.png")
