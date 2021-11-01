from ESML import *

g = Const('g')

x,y = SpatialCoordinates('x', 'y')
t = TimeCoordinate('t')
eta = Field([x,y], ['eta'])
u,v = Field([x,y], ['u', 'v'])

EOM1 = D(u,t) == -g * D(eta,x)
EOM2 = D(v,t) == -g * D(eta,y)

Lx = 1000
Ly = 500
d = RectDomain({x: (-Lx/2,Lx/2), y: (-Ly/2,Ly/2)})
initial = Initial({eta: exp(-x*x-y*y), u: 0, v: 0})

system = ODESystem(domain=d,
                   initial=initial,
                   time=t,
                   equations=[EOM1, EOM2],
                   consts={'g': 9.81})

system.describe()
system.run()
