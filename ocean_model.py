
from ESML import *

Omega = Const('Omega') # Coriolis constant
A_M = Const('A_M') # Ocean viscosity
a = Const('a') # Mean sea radius
rho0 = Const('rho0') # Mean water density
mu = Const('mu') # Vertical mixing
g = Const('g') # Gravity constant

# Set up spatial and field variables
lam,phi,z = SpatialCoordinates('lam', 'phi', 'z') # Longitude, latitude and depth
t         = TimeCoordinate('t')
u,v,w     = Field([lam,phi,z], ['u', 'v', 'w']) # Water density velocities along lam, phi, and z
p, rho    = Field([lam,phi,z], ['p', 'rho']) # Pressure and water density

# Auxiliary definitions for defining equations
f = 2*Omega*Sin(phi)            # Coriolis parameter f
advection   = lambda alpha: 1/(a*Cos(phi)) * (D(u*u, lam) + D(Cos(phi)*v*alpha,phi)) + D(w*u,z)        # Eq. (2.3)
laplacian   = lambda alpha: 1/((a*Cos(phi))**2) * D(alpha,lam,2) + 1/(a*a*Cos(phi))*D( Cos(phi)*D(alpha,phi), phi) # Eq. (2.6)

friction_Hx = A_M*( laplacian(u) + u*(1-Tan(phi)**2)/a**2 - 2*Sin(phi)/((a*Cos(phi)**2) * D(v,lam))) # Eq. (2.4)
friction_Hy = A_M*( laplacian(u) + v*(1-Tan(phi)**2)/a**2 - 2*Sin(phi)/((a*Cos(phi)**2) * D(u,lam))) # Eq. (2.5)
friction_V  = lambda alpha: D(mu*D(alpha, z), z)                                                     # Eq. (2.7)

# Fundamental equations of motion: Momentum equations for the Ocean
EOM1 = D(u,t) == -advection(u) + (u*v)*Tan(phi)/a + f*v - 1/(rho0 * a * Cos(phi))*D(p,lam) + friction_Hx + friction_V(u) # Eq. (2.1)
EOM2 = D(v,t) == -advection(v) + (u*u)*Tan(phi)/a + f*u - 1/(rho0 * a * Cos(phi))*D(p,lam) + friction_Hy + friction_V(v) # Eq. (2.2)

velocities_eq  = [u,v,w] == Gradient(rho)
hydrostatic_eq = D(p,z) == -rho*g

print(EOM1)
print(EOM2)
print(str(velocities_eq))
print(str(hydrostatic_eq))
