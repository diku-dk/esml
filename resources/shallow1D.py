#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn
import sys

# 
# This file implements a 1D shallow water equation in a rectangular canal of depth h0.
#
# simulation_step_naive() implements the shallow-water equations directly
# simulation_step_controlvolume() uses the control-volume technique for greater numerical stability
#
# Try different initial condition set-ups using the square_dam_break(), smooth_dam_break(), and wave_paddle() functions
#
# Cheers, James Avery (avery@nbi.dk)
#
# [OMB]: Ocean Modeling for Beginners, Jochen Kaempf (2010)
# URL:   https://www.researchgate.net/publication/261174088_Ocean_Modelling_for_Beginners_-_Using_Open-Source_Software
# -------- SET UP SIMULATION PARAMETERS --------
Length = 1000                        # Length of canal in meters 
n      = 2000                        # Number of simulation points
g      = 9.81                        # m/s^2
h0     = 10                          # Initial water-depth in meters
dx     = Length / n                  # dx in meters

CFL_target = 0.2
dt = CFL_target*dx/np.sqrt(g*(h0+2))   # Choose largest time step that satisfies the CFL-condition to CFL_target<1

# -------- SET UP UNITS --------
xs = np.linspace(0,Length,n)
plot_interval   = int(np.ceil( 0.5/dt))  # Update plot every 0.5 simulation second
plot_resolution = int(np.ceil(  10/dx))  # Plot points at 1m intervals

# Shapiro filter parameter
epsilon = 0.002*np.sqrt(n) #0.0015
dtype=np.float64

# -------- ALLOCATE MEMORY --------
u          = np.zeros(n,dtype=dtype)
eta        = np.zeros(n,dtype=dtype)
h          = np.zeros(n,dtype=dtype)

# Middle, Left-shifted, and Right-shifted slices. This is short-hand to avoid sticking [1:-1],[0:-2], and [2:] everywhere.
M = slice(1,-1)
L = slice(0,-2)
R = slice(2,None)

# -------- VARIOUS INITIAL CONDITIONS --------
#Square dam break
def square_dam_break(dam_width,x0=Length/2,height=1):
    drop_start, drop_end = x0/dx-dam_width/(2*dx), x0/dx+dam_width/(2*dx)
    eta[int(round(drop_start)):int(round(drop_end))] = 1

#Gaussian dam break
def smooth_dam_break(width,x0=Length/2,height=1):
    xs = np.linspace(0,Length,n)
    eta[:] += height*np.exp(-(xs-x0)**2/(2*width**2))

def wave_paddle(width,x0=Length/2,t=0,period=100,amplitude=1):
    paddle_start, paddle_end = x0/dx-width/(2*dx), x0/dx+width/(2*dx)
    eta[int(round(paddle_start)):int(round(paddle_end))] = amplitude*np.sin(t*2*np.pi/period)

    

# -------- THE ACTUAL SHALLOW WATER SIMULATION--------
# Staggered grid first-order central finite difference
def Dx0(f,dx): return (1/dx)*(f[R]-f[M]) # d/dx on Grid 0
def Dx1(f,dx): return (1/dx)*(f[M]-f[L]) # d/dx on Grid 1


# -- STRAIGHT-FORWARD IMPLEMENTATION --
# u_t   = -g eta_x
# eta_t = - d/dx (u*h) = -u*h_x -u_x*h
def simulation_step_naive():
    # [OMB] Eq. (4.12)/(4.17):
    # u_t = -g eta_x
    eta_x    =  Dx0(eta,dx);
    u[M]    += -g*eta_x*dt      # Forward-Euler time integration

    # Velocity boundary-conditions
    u[:1]  =  0
    u[-2:] =  0
    
    # [OMB] Eq. (4.13)
    h        = h0+eta
    h_x      = Dx0(h,dx)
    u_x      = Dx1(u,dx)
    eta_t    = -u[M]*h_x - u_x*h[M];
    
    eta[M]  += eta_t*dt      # Forward-Euler time integration

    # Height boundary-conditions
    eta[0]  = eta[1]
    eta[-1] = eta[-2]
    
    # First order Shapiro filter, [OMB] Eq. (4.21)
    eta[M]  = (1-epsilon)*eta[M] + epsilon*(eta[R] + eta[L])/2


# -- CONTROL-VOLUME IMPLEMENTATION --    
def controlvolume(u,h):
    u_plus, u_minus = .5 * (u + np.abs(u)), .5 * (u - np.abs(u))          # Positive resp negative regions of u
    return (u_plus[M]-u_minus[L])*h[M] + u_minus[M]*h[R] - u_plus[L]*h[L] # [OMB], Eq. (4.19)

def simulation_step_controlvolume():
    # OMB Eq. (4.12)/(4.17):
    # Velocity: u_t = -g eta_x
    eta_x = Dx0(eta,dx);
    u[M] += -g*eta_x*dt      # Forward Euler time integration

    # Velocity boundary-conditions
    u[:1]  = 0
    u[-2:] = 0

    # OMB Eq. (4.13)/(4.19)
    # Control-volume technique    
    # Eq. (4.19)
    h[M]  = h0+eta[M]
    eta[M] = eta[M] -(dt/dx)*controlvolume(u,h)

    # Displacement-height boundary condition 
    eta[0]  = eta[1]
    eta[-1] = eta[-2]
     
    # First-order Shapiro filter, OMB Eq. (4.21)
    eta_smooth = 0.5 * (eta[L] + eta[R])
    eta[M] = (1 - epsilon) * eta[M] +  epsilon * eta_smooth

    
# -------- PLOT FUNCTION --------
def plot_update(frames):
    global t,t0

    for j in range(plot_interval):  # Run plot_interval simulation steps between each plot
# Replace the next line by simulation_step_controlvolume() to use the control-volume technique
        simulation_step_controlvolume()
# Uncomment the next line to place a wave-generating paddle, waving for 100 seconds
#        if(t<100): wave_paddle(40,period=43,amplitude=0.3,t=t,x0=Length/2)
        t += dt
    
    if (t-t0>10): # Check water volume every 10 simulation seconds
        t0 = t
        print(f"at {round(t,2)} seconds, water volume in eta is {round(dx*np.sum(eta[M]),8)}")

    # Update matplotlib-plot
    water_plot.set_ydata(eta[::plot_resolution])
    ax.set_title(f"time: {round(t)}s")





# -------- INITIALIZE AND RUN THE SIMULATION + PLOTS --------    
#square_dam_break(100,x0=Length/2)
smooth_dam_break(25,x0=1*Length/5,height=1)
smooth_dam_break(25,x0=2*Length/5,height=1)
smooth_dam_break(25,x0=3*Length/5,height=1)
smooth_dam_break(25,x0=4*Length/5,height=1)


#smooth_dam_break(25,x0=Length/2,height=1)
# CFL condition

print(f"epsilon={epsilon}\n"
      f"dt     ={dt}\n"
      f"dx     ={dx}\n"
      f"CFL    ={(dt/dx)*np.sqrt(g*(h0+eta.max()))}\n"
      f"time steps per plot update={plot_interval}\n"
      f"steps per plot point={plot_resolution}\n"
)


t,t0 = 0,0     # Global time

# Set up Matplotlib plot
seaborn.set(style='ticks')
fig, ax = plt.subplots()
water_plot, = ax.plot(xs[::plot_resolution],eta[::plot_resolution], '-')
#ax.set_ylim((-h0,2))
ax.set_ylim((-2,2))
ax.grid(True, which='both')
title  = ax.set_title("time: 0s")
ani = FuncAnimation(fig, plot_update, interval=5,frames=1000,repeat=False)

#ani.save("shallow1D.mp4",fps=20)
plt.show()


