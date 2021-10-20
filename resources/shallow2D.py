#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn
import sys
import imageio
from matplotlib import cm
NA = np.newaxis



# -------- SET UP SIMULATION PARAMETERS --------
Lx,Ly  = 500,400                # 500m x 400m
nx, ny = 125,100                 # Simulation points along each axes
n     = nx*ny                    # Total simulation points
dx,dy = Lx/nx, Ly/ny             # dx = dy = 1m
xs = np.linspace(0,Lx,nx)[NA,:]
ys = np.linspace(0,Ly,ny)[:,NA]

g  = 9.81                        # m/s^2
h0 = 10                          # 10m water

CFL_target = 0.1
dt = CFL_target*min(dx,dy)/np.sqrt(2*g*(h0+2))   # Choose largest time step that satisfies the CFL-condition to CFL_target<1

plot_interval   = int(np.ceil( 0.5/dt))  # Update plot every 0.5 simulation second
plot_resolution = int(np.ceil(  1/dx))   # Plot points at 1m intervals

print(f"dt: {dt}")

# Shapiro filter parameter
epsilon = 0.005
dtype=np.float64

# Middle, Left-shifted, and Right-shifted slices. This is short-hand to avoid sticking [1:-1],[0:-2], and [2:] everywhere.
M = slice(1,-1)
L = slice(0,-2)
R = slice(2,None)

# -------- ALLOCATE MEMORY --------
u   = np.zeros((ny,nx),        dtype=dtype)
v   = np.zeros((ny,nx),        dtype=dtype)
eta = np.zeros((ny,nx),        dtype=dtype)
u_x = np.zeros((ny-2,nx-2),  dtype=dtype)
v_y = np.zeros((ny-2,nx-2),  dtype=dtype)
eta_star   = np.zeros((ny,nx), dtype=dtype)
eta_smooth = np.zeros((ny,nx), dtype=dtype)
h          = np.zeros((ny,nx), dtype=dtype)



# -------- VARIOUS INITIAL CONDITIONS --------
#Gaussian dam break
def smooth_dam_break(w=(10,10),x0=(Lx/2,Ly/2),height=1):
    R2 = ((xs-x0[0])/w[0])**2 + ((ys-x0[1])/w[1])**2
    eta[:,:] += height*np.exp(-R2)

def wave_paddle(w=(10,10),center=(Lx/2,Ly/2),t=0,period=100,amplitude=1):
    paddle_area = (np.abs(xs-center[0])<=w[0]/2) & (np.abs(ys-center[1])<=w[1]/2)
    eta[paddle_area] = amplitude*np.sin(t*2*np.pi/period)
    

# -------- THE ACTUAL SHALLOW WATER SIMULATION--------
# Staggered grid first-order central finite difference
def grid0_Dx(f,dx): return (1/dx)*(f[R,M] - f[M,M])
def grid1_Dx(f,dx): return (1/dx)*(f[M,M] - f[L, M])
def grid0_Dy(f,dx): return (1/dy)*(f[M,R] - f[M,M])
def grid1_Dy(f,dx): return (1/dy)*(f[M,M] - f[M, L])


# u_t   = -g eta_x
# v_t   = -g eta_y
# eta_t = - d/dx (u*h) -d/dy (v*h) = -u*h_x - v*h_y - u_x*h -v_x*h
def simulation_step_naive():
    # OMB Eq. (5.1) i-ii:
    # u_t = -g eta_x
    eta_x = grid0_Dx(eta,dx); # eta_x = D(eta,x)
    eta_y = grid0_Dy(eta,dy); # eta_y = D(eta,y)
     
    u[M,M] += -g*eta_x*dt      # Forward Euler time integration
    v[M,M] += -g*eta_y*dt

    # Velocity boundary-conditions
    u[:, :1] = 0
    u[:,-2:] = 0
    v[:1, :] = 0
    v[-2:,:] = 0

    
    # OMB Eq. (5.1) iii
    u_x      = grid1_Dx(u,dx)   # u_x = D(u,x)
    v_y      = grid1_Dy(v,dy)   # v_y = D(v,y)
      
    eta_t    = -u[M,M]*eta_x - v[M,M]*eta_y - (u_x+v_y)*(h0+eta[M,M]); # EOM: D(eta,t) == -u*D(eta,x) - v*D(eta,y) - (D(u,x)+D(v,y))*(h0+eta)
               
   
    eta[M,M]    += eta_t*dt     # Integrer EOM med forward Euler: eta = eta + \int_{t_i}^{t_i+dt} D(eta,t)*dt

    # Height boundary-conditions    
    eta[:, 0] = eta[:, 1]
    eta[:,-1] = eta[:,-2]
    eta[0, :] = eta[1, :]
    eta[-1,:] = eta[-2,:]
    
    # First order Shapiro filter, OMB Eq. (4.21)
    eta[M,M] = (1-epsilon)*eta[M,M] + epsilon*(1/4)*(eta[R,M] + eta[L,M] + eta[M,R] + eta[M,L])

def simulation_step_controlvolume():
    # OMB Eq. (5.1) i-ii:
    # u_t = -g eta_x
    eta_x = grid0_Dx(eta,dx);
    eta_y = grid0_Dy(eta,dy);
     
    u[M,M] += -g*eta_x*dt      # Forward Euler time integration
    v[M,M] += -g*eta_y*dt

    # Velocity boundary-conditions
    u[:, :1] = 0
    u[:,-2:] = 0
    v[:1, :] = 0
    v[-2:,:] = 0

    
    # OMB Eq. (5.1) iii
    u_x      = grid1_Dx(u,dx)
    v_y      = grid1_Dy(v,dy)
      
    # Extract negative and positive parts of velocities u and v
    u_plus, u_minus = .5 * (u + np.abs(u)), .5 * (u - np.abs(u))
    v_plus, v_minus = .5 * (v + np.abs(v)), .5 * (v - np.abs(v))    

    # Time derivative for displacement height eta using the control-volume technique
    h[M,M]  = h0+eta[M,M]        
    eta_star[M,M] = eta[M,M] \
                     -(dt/dx)*( u_plus [M,M]*h[M,M]
                              + u_minus[M,M]*h[R,M]
                              - u_plus [L,M]*h[L,M]
                              - u_minus[L,M]*h[M,M])\
                     -(dt/dy)*( v_plus [M,M]*h[M,M]
                              + v_minus[M,M]*h[M,R]
                              - v_plus [M,L]*h[M,L]
                              - v_minus[M,L]*h[M,M])
    # Height boundary-conditions    
    eta_star[:, 0] = eta_star[:, 1]
    eta_star[:,-1] = eta_star[:,-2]
    eta_star[0, :] = eta_star[1, :]
    eta_star[-1,:] = eta_star[-2,:]
    
    # First order Shapiro filter, OMB Eq. (4.21)
    eta[M,M] = (1-epsilon)*eta_star[M,M] + epsilon*(1/4)*(eta_star[R,M] + eta_star[L,M] + eta_star[M,R] + eta_star[M,L])    
    eta[:, 0] = eta[:, 1]
    eta[:,-1] = eta[:,-2]
    eta[0, :] = eta[1, :]
    eta[-1,:] = eta[-2,:]
    
# -------- PLOT FUNCTION --------
def plot_update(frames):
    global t,t0

    for j in range(plot_interval):        # Run plot_interval simulation steps between each plot
        simulation_step_controlvolume()
#        if(t<120): wave_paddle(w=(40,10),period=21,amplitude=0.3,t=t,center=(Lx/2,Ly/2))
#        if(t<120): wave_paddle(w=(10,40),period=28,amplitude=0.3,t=t,center=(Lx/4,Ly/4))
#        if(t<120): wave_paddle(w=(5,5),  period=40,amplitude=0.3,t=t,center=(3*Lx/4,3*Ly/4))                
        t += dt
    
    if (t-t0>10): # Check water volume every 10 simulation seconds
        t0 = t
        print(f"at {round(t,2)} seconds, water volume in eta is {round(dx*dy*np.sum(eta[M,M]),8)}")

#    ax.clear()
#    ax.contour(xs[0],ys[:,0],eta,vmin=-1,vmax=1)
    im.set_array(eta[::plot_resolution,::plot_resolution])
    ax.set_title(f"time: {round(t)}s")    
#    line1.set_ydata(eta[ny//2,::1])




# -------- INITIALIZE AND RUN THE SIMULATION + PLOTS --------    
#square_dam_break(100,x0=L/2)
smooth_dam_break((10,10),x0=(Lx/3,Ly/3),height=1)
smooth_dam_break((10,10),x0=(2*Lx/3,Ly/2),height=1)
smooth_dam_break((20,20),x0=(Lx/2,Ly/3),height=1)
smooth_dam_break((10,10),x0=(Lx/2,2*Ly/3),height=1)

t,t0 = 0,0     # Global time
   
seaborn.set(style='ticks')
fig, ax = plt.subplots()
ax.set_aspect('equal')
#line1, = ax.plot(xs[0,::1],eta[ny//2,::1], '-')
im = ax.imshow(eta[::plot_resolution,::plot_resolution],vmin=-.35,vmax=.35,cmap='inferno')
#ax.set_ylim((-h0,2))
#ax.set_ylim((-1,1))
#ax.grid(True, which='both')
title  = ax.set_title("time: 0s")
ani = FuncAnimation(fig, plot_update, interval=5)

plt.show()
