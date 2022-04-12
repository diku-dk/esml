from ESML import *

g = Const('g')
h0 = Const('h0')

x,y = SpatialCoordinates('x', 'y')
t = TimeCoordinate('t')
eta = Field([x,y], ['eta'])
u,v = Field([x,y], ['u', 'v'])

h = eta + h0

eqns = {'5.1_u_t': D(u,t) == -g * D(eta,x),
        '5.1_v_t': D(v,t) == -g * D(eta,y),
        '5.1.eta_t': D(eta,t) == -D(u*h,x) - D(v*h,y)}

Lx = 1000
Ly = 500
d = RectDomain({x: (-Lx/2,Lx/2), y: (-Ly/2,Ly/2)})
initial = Initial({eta: exp(-x*x-y*y), u: 0, v: 0})
left = MinOf(x)
right = MaxOf(x)
top = MinOf(y)
bot = MaxOf(y)
boundary = [(left, u==0, Flux(eta)==0),
            (right, u==0, Flux(eta)==0),
            (top, v==0, Flux(eta)==0),
            (bot, v==0, Flux(eta)==0)]

system = ODESystem(domain=d,
                   initial=initial,
                   boundary=boundary,
                   time=t,
                   equations=eqns.values(),
                   consts={'g': 9.81, 'h0': 5})

print('# Description of system')
system.describe()

disc = Discretisation(system=system,
                      spatial_derivative = FiniteDifference("Central", order=4),
                      time_integrator = Integrator("Forward-Euler"))

print('\n# Futhark code')
disc.gen()
