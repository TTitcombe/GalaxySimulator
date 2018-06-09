'''A Galaxy Collapse Simulator. N stars in Galaxy,
with multiple `Rogue stars`'''

from __future__ import division
import numpy
import time

N = 500 #number of stars in cluster

G = 6.67408e-11 #gravitational constant
massSun = 1.989e+30 # Sun's mass in kg
t_dyn = 3 * 10**14 #dynamical time
ro = 0.4 #stars per cubic parsec
V = N / ro #volume in parsecs
R = (V/((4/3)*numpy.pi))**(1/3)
R_m = R * 3.09 * 10**16 #radius of globular cluster in metres
eps = 0.1 * R_m

max_time = 3.375e+14 # 27x10^14 / 8...roughly 1/33 age of universe
iterations = 500 #4000 / 8
dt = max_time / iterations # dt roughly 1/1000 of Tdyn
times = numpy.linspace(0, max_time, iterations)

def updateA(x,m,N):
    '''Calculates the acceleration between objects of separation r and mass m'''
    a = numpy.zeros((len(m),3))

    for i in range(len(m)):
        for j in range(len(m)):
            if (i != j):
                r_sq = ((x[i,:]-x[j,:])**2).sum()
                a[i,:] += -G*(x[i,:]-x[j,:]) * m[j] / ((r_sq + eps**2)**1.5)

    return a

def updateX(x, v, dt):
    x = x + v * dt
    return x

def updateV(v, a, dt):
    v = v + a * dt
    return v

def c_o_m(r,m):
    '''returns the centre of mass of a system with position vectors p and masses m.'''
    COM = numpy.zeros(3)
    M = m.sum() #total mass
    for a in range(3):
        COM[a] = (r[:,a]*m).sum()/M

    return COM

def v_c_o_m(v,m):
    '''returns the velocity of the centre of mass'''
    v_COM = numpy.zeros(3)
    M = m.sum()
    for i in range(3):
        v_COM[i] = (v[:,i]*m).sum()/M

    return v_COM

def Potential(x,m):
    Phi = 0
    for i in range(len(m)):
        for j in range(len(m)):
            R = numpy.sqrt(((x[i,:]-x[j,:])**2).sum() + eps**2)
            if (i != j):
                Phi += - G * m[i] * m[j] / R

    return Phi

def Kinetic(v,m):
    KE = 0.
    M = m.sum()
    virialKE = 0.
    for i in range(3):
        KE += 0.5*(m * (v[:,i]*v[:,i]) ).sum()
        virialKE += (v[:,i]*v[:,i]).sum() / N
    virialKE = virialKE * 0.5 * M
    return KE, virialKE


def run():
    numpy.random.seed(256)

    #Initialise random starting positions and velocities
    X1, X2 = numpy.random.random(N), numpy.random.random(N)
    X3, X4 = numpy.random.random(N), numpy.random.random(N)
    X5, X6, X7 = numpy.random.random(N), numpy.random.random(N), numpy.random.random(N)

    r_scaled = (X1**(-2/3) - 1)**(-1/2)
    R_particle = r_scaled * R_m
    z_pos = (1 - 2* X2) * R_particle
    x_pos = numpy.sqrt(R_particle**2 - z_pos**2) * numpy.cos(2 * numpy.pi * X3)
    y_pos = numpy.sqrt(R_particle**2 - z_pos**2) * numpy.sin(2 * numpy.pi * X3)

    ve = (1 + r_scaled**2)**(-1/4) * numpy.sqrt(2)
    g5 = X5**2 * ((1 - X5**2)**(7/2))
    for i in range(N):
        while 0.1 * X4[i] > g5[i]:
            X4[i], X5[i] = numpy.random.random(), numpy.random.random()
            g5[i] = X5[i] ** 2 * ((1 - X5[i]**2)**(7/2))
    q = X5
    v_particle = q * ve * 100
    v_z = (1 - 2*X6)*v_particle
    v_x = numpy.sqrt(v_particle**2 - v_z**2) * numpy.cos(2 * numpy.pi * X7)
    v_y = numpy.sqrt(v_particle**2 - v_z**2) * numpy.sin(2 * numpy.pi * X7)


    #Place Rogue stars
    numpy.random.seed(5)
    m = numpy.random.uniform(0.8, 4, N) * massSun
    m[-1] = 5 * massSun #Rogue Star
    m[-2] = massSun
    m[-3] = 2.3 * massSun
    m[-4] = 4 * massSun

    x = numpy.zeros((iterations, N, 3))
    x[0,:,0] = x_pos
    x[0,:,1] = y_pos
    x[0,:,2] = z_pos
    x[0,-1,:] = numpy.array([-10000 * max_time * 2.7, 1e+17, 1e+17])
    x[0,-4,:] = numpy.array([-20000 * max_time, 0, 0])
    x[0,-3,:] = numpy.array([-50000 * max_time * 5, 2e+17,0])
    x[0,-2,:] = numpy.array([-10000 * max_time * 1.5, 0, -1e+17])

    v = numpy.zeros((iterations, N, 3))
    v[0,:,0] = v_x
    v[0,:,1] = v_y
    v[0,:,2] = v_z
    v[0,-1,:] = numpy.array([10000, 0.0,0.0])
    v[0,-4,:] = numpy.array([20000, 0.0, 0.0])
    v[0,-3,:] = numpy.array([50000, 0.0, 0.0])
    v[0,-2,:] = numpy.array([10000, 0.0, 0.0])

    a = numpy.zeros((iterations, N, 3))

    P = numpy.zeros(iterations)
    K = numpy.zeros(iterations)
    virialK = numpy.zeros(iterations)
    P[0] = Potential(x[0,:,:], m)
    K[0], virialK[0] = Kinetic(v[0,:,:], m)

    CoM = numpy.zeros((iterations,3))
    CoM[0,:] = c_o_m(x[0,:,:], m)

    for i in range(iterations-1):

        a[i,:,:] = updateA(x[i,:,:], m, i)
        v[i+1,:,:] = updateV(v[i,:,:], a[i,:,:], dt)
        tempPos = updateX(x[i,:,:], v[i+1,:,:], dt/2.0)

        P[i+1] = Potential(tempPos, m)
        K[i+1], virialK[i+1] = Kinetic(v[i+1,:,:], m)

        x[i+1,:,:] = updateX(x[i,:,:], v[i+1,:,:], dt)
        CoM[i+1,:] = c_o_m(x[i+1,:,:], m)

    return P, K, virialK, x, v, a, m, CoM

start_time = time.time()
P, K, virialK, x, v, a, m, CoM = run(True)
end_time = time.time()

total = P + K
energyChange = ((total[-1]-total[0])/total[0])*100

print("Final Percentage Energy Change: {}".format(energyChange))
print("Time taken: {}".format(end_time - start_time))
