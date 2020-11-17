import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CTRNN(object):
    def __init__(self, N=1, M=2, dt=0.01, T=100, w=torch.empty(0), tau=torch.empty(0), theta=torch.empty(0), I=torch.empty(0)):
        self.dt = dt
        self.T = T
        self.iterations = int(T//dt)
        self.N = N
        self.M = M                                                              # embedding orders, e.g., 1 for state, 2 for state and its derivative, etc.

        self.x = 30*torch.rand(self.iterations, self.N, self.M)-15                    # states
        if w.nelement() == 0:
            self.w = torch.rand(self.iterations, self.N, self.N)                    # weights
        else:
            self.w = torch.zeros(self.iterations, self.N, self.N)                    # weights
            self.w[0,:,:] = w

            # symmetric/antisymmetric decomposition
            self.S = torch.zeros(self.iterations, self.N, self.N)                    # weights, symmetric part
            self.A = torch.zeros(self.iterations, self.N, self.N)                    # weights, antisymmetric part
            self.S[0,:,:] = .5 * (self.w[0,:,:] + self.w[0,:,:].T)
            self.A[0,:,:] = .5 * (self.w[0,:,:] - self.w[0,:,:].T)

        if tau.nelement() == 0:
            self.tau = torch.ones(self.iterations, self.N)                          # time constants
        else:
            self.tau = torch.zeros(self.iterations, self.N)                          # time constants
            self.tau[0,:] = tau
        if theta.nelement() == 0:
            self.theta = torch.zeros(self.iterations, self.N)                       # biases
        else:
            self.theta = torch.zeros(self.iterations, self.N)                       # biases
            self.theta[0,:] = theta
        if I.nelement() == 0:
            self.I = torch.zeros(self.iterations, self.N)                           # external inputs
        else:
            self.I = torch.zeros(self.iterations, self.N)                           # external inputs
            self.I[0,:] = I

    def dx(self, i):
        sigma = torch.sigmoid(self.x[i,:,0] + self.theta[i,:])
        self.x[i,:,1] = 1/self.tau[i,:] * (- self.x[i,:,0] + self.S[i,:,:] @ sigma + self.A[i,:,:] @ sigma + self.I[i,:])

    def step(self, i):
        self.dx(i)
        dw = 0
        dtau = 0
        dtheta = 0
        dI = 0
        dS = 0
        dA = 0


        self.x[i+1,:,0] = self.x[i,:,0] + self.dt * self.x[i,:,1]
        self.w[i+1,:,:] = self.w[i,:,:] + self.dt * dw
        self.tau[i+1,:] = self.tau[i,:] + self.dt * dtau
        self.theta[i+1,:] = self.theta[i,:] + self.dt * dtheta
        self.I[i+1,:] = self.I[i,:] + self.dt * dI
        self.S[i+1,:] = self.S[i,:] + self.dt * dS
        self.A[i+1,:] = self.A[i,:] + self.dt * dA

class StochasticCTRNN(CTRNN):
    def __init__(self, N=1, M=2, dt=0.01, T=100, w=torch.empty(0), tau=torch.empty(0), theta=torch.empty(0), I=torch.empty(0), C=torch.empty(0)):
        super().__init__(N, M, dt, T, w, tau, theta, I)

        if C.nelement() == 0:
            self.C = torch.zeros(self.N, self.N)                       # biases
            self.D = torch.zeros(self.N, self.N)                       # diffusion operator
        else:
            self.C = torch.zeros(self.N, self.N)                       # biases
            self.C = C                                                      # covariance matrix
            self.D = torch.cholesky(self.C)                                 # diffusion operator

        self.dW = torch.randn(self.iterations, self.N)          # Wiener process
        
    def dx(self, i):
        sigma = torch.sigmoid(self.x[i,:,0] + self.theta[i,:])
        irrotational = - self.x[i,:,0] + self.S[i,:,:] @ sigma
        solenoidal = 0#self.A[i,:,:] @ sigma
        self.x[i,:,1] = 1/self.tau[i,:] * (irrotational + solenoidal + self.I[i,:] +  self.D @ self.dW[i,:]/torch.sqrt(torch.tensor(self.dt)))


trajectories_n = 10

dt = .01
T = 50.

# variables = 2
# w = torch.tensor([[4.5, -2.],[2., 4.5]]).T
# theta = torch.tensor([-2.75, -1.75])
# C = torch.tensor([[.05, 0.],[0., .05]])

# NNeuronsCTRNNs = []

# for i in range(trajectories_n):
#     print('Trajectory #: ', i+1)
#     NNeuronsCTRNNs.append(StochasticCTRNN(dt=dt, T=T, N=variables, w=w, theta=theta))
#     for j in range(NNeuronsCTRNNs[i].iterations-1):
#         NNeuronsCTRNNs[i].step(j)

# plt.figure()
# for i in range(trajectories_n):
#     plt.plot(NNeuronsCTRNNs[i].x[:-1,0,0], NNeuronsCTRNNs[i].x[:-1,1,0])





variables = 3
w = torch.tensor([[5.422, -0.24, 0.535],[-0.018, 4.59, -2.25],[2.75, 1.21, 3.885]]).T
theta = torch.tensor([-4.108, -2.787, -1.114])
C = torch.tensor([[.05, 0., 0.],[0., .05, 0.], [0., 0., 0.05]])
tau = torch.tensor([1., 1., 1.])

NNeuronsCTRNNs = []

for i in range(trajectories_n):
    print('Trajectory #: ', i+1)
    NNeuronsCTRNNs.append(StochasticCTRNN(dt=dt, T=T, N=variables, w=w, theta=theta, tau=tau))
    for j in range(NNeuronsCTRNNs[i].iterations-1):
        NNeuronsCTRNNs[i].step(j)

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(trajectories_n):
    ax.plot(NNeuronsCTRNNs[i].x[:-1,0,0], NNeuronsCTRNNs[i].x[:-1,1,0], NNeuronsCTRNNs[i].x[:-1,2,0])




plt.show()