# Code for double well simulation in Chaudhari, Pratik, and Stefano Soatto. 
# "Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks." 
# International Conference on Learning Representations. 2018.
#
# Adapted from https://github.com/pratikac/grad-info

import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D


T = 55.
dt = 0.05
iterations = int(T//dt)

b = torch.exp(torch.tensor(0.))
eps = 7.5 * 10**(-4)

ls = [0,1.5,2.0]                                                # strength of nonequilibrium (solenoidal) current

def phi(x, y):
    return (x**2-1)**2/4. + y**2/2.

def f_u(x, y, l):
    return   l*torch.exp(phi(x,y) - (x**2 + y**2)**2/4.)*(-y)

def f_v(x, y, l):
    return -y + l*torch.exp(phi(x,y) - (x**2 + y**2)**2/4.)*(x)

x = torch.zeros(iterations, 2, 2)
x[0,:,0] = torch.tensor([-1.5, 0.])

xi = torch.normal(0, 2*eps*b, size=(iterations, 1))



for i in range(len(ls)):
    l = ls[i]

    plt.figure(i, figsize=(7,7))
    plt.clf()

    d = 1.5

    # Contour Plot
    y2,y1 = torch.meshgrid(torch.linspace(-d,d,20), torch.linspace(-d,d,20))

    # Vector Field
    phi0 = phi(y1,y2)
    u = f_u(y1,y2,l)
    v = f_v(y1,y2,l)

    s = torch.sqrt(u**2 + v**2) + 1e-6
    u /= s
    v /= s

    plt.streamplot(y1.numpy(),y2.numpy(),u.numpy(),v.numpy(), density=0.5, color='k', linewidth=7*s.numpy() / s.numpy().max())
    plt.grid()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xticks([-1,0,1])
    plt.yticks([-1,0,1])
    # plt.xticks([])
    # plt.yticks([])
    plt.plot([-1,1],[0,0], 'ro', ms=20)
    plt.axes().set_aspect('equal')
    plt.contourf(y1, y2, phi0, levels=torch.linspace(phi0.min(), phi0.max(), 6), cmap=cm.Blues, alpha=0.25)
    plt.savefig('./fig/double_well%d.pdf'%(i+1), bbox_inches='tight')

fig = plt.figure(i+20, figsize=(7,7))
ax = fig.gca(projection='3d')
ax.plot_surface(y1.numpy(), y2.numpy(), phi0.numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False)


intervals_n = len(ls)
interval = int(iterations//intervals_n)-1

for j in range(intervals_n):
    l = ls[j]
    for i in range(j*interval,(j+1)*interval):
        x[i,0,1] = f_u(x[i,0,0], x[i,1,0], l)
        x[i,1,1] = f_v(x[i,0,0], x[i,1,0], l) + xi[i]/torch.sqrt(torch.tensor(dt))

        x[i+1, :, 0] = x[i, :, 0] + dt * x[i, :, 1]

color=iter(cm.rainbow(torch.linspace(0,1,intervals_n+1)))
c=next(color)
plt.figure(10, figsize=(7,7))
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.plot([-1,1],[0,0], 'ro', ms=20)
plt.plot(x[0, 0, 0], x[0, 1, 0], 'o')
for j in range(intervals_n):
    c = next(color)
    plt.plot(x[j*interval:(j+1)*interval, 0, 0], x[j*interval:(j+1)*interval, 1, 0], c=c)


color=iter(cm.rainbow(torch.linspace(0,1,intervals_n+1)))
c=next(color)

fig = plt.figure(11, figsize=(8,8))
ax = fig.gca(projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim(-.1, 0.5)
ax.plot([-1,1],[0,0], phi(torch.tensor([-1,1]),torch.tensor([0,0])), 'ko', ms=4)
for j in range(intervals_n):
    c = next(color)
    ax.plot(x[j*interval:(j+1)*interval, 0, 0], x[j*interval:(j+1)*interval, 1, 0], phi(x[j*interval:(j+1)*interval, 0, 0], x[j*interval:(j+1)*interval, 1, 0]))

plt.show()