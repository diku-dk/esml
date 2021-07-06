#include <ESML>
#include <Veros2/constants.hh>

using namespace ESML;
using namespace Veros2;

constexpr real_t
 a     = mean_sea_radius,
 rho0  = mean_water_density,
 mu    = vertical_mixing_mu,
 Omega = coriolis_constant,
 A_M   = ocean_viscosity,
  g    = gravity_constant;

// Set up spatial and field variables
ESML::Model M;
Variable lam, phi, z, t, u, v, w, p, rho;

M.setSpatialCoordinates({lam,phi,z});
M.setTimeCoordinate(t);
M.setFields({lam,phi,z}, {u,v,w});  // Water density velocities along lam, phi, and z
M.setFields({lam,phi,z}, {p,rho});  // Pressure and water density

// Auxiliary definitions for defining equations
f = 2*Omega*Sin(phi);            // Coriolis parameter f
auto advection = [&](Variable &alpha) { return 1/(a*Cos("phi")) * (D(u*u, lam) + D(Cos(phi)*v*alpha)) + D(w*u,z);  };                       // Eq. (2.3) 
auto laplacian = [&](Variable &alpha) { return 1/((a*Cos("phi"))^2) * D(alpha,lam,2) + 1/(a*a*Cos(phi))*D( Cos(phi)*D(alpha,phi), phi); }; // Eq. (2.6)

Expression friction_Hx = A_M*( laplacian(u) + u*(1-Tan(phi)^2)/a^2 - 2*Sin(phi)/((a*Cos(phi)^2) * D(v,lam))); // Eq. (2.4)
Expression friction_Hy = A_M*( laplacian(u) + v*(1-Tan(phi)^2)/a^2 - 2*Sin(phi)/((a*Cos(phi)^2) * D(u,lam))); // Eq. (2.5)
auto friction_V =[&](Variable &alpha) { return D(mu*D(alpha, z), z); };                                          // Eq. (2.7)

// Fundamental equations of motion: Momentum equations for the Ocean
Equation EOM1 = D(u,t) == -advection(u) + (u*v)*Tan(phi)/a + f*v - 1/(rho0 * a * Cos(phi))*D(p,lam) + friction_Hx + friction_V(u); // Eq. (2.1)
Equation EOM2 = D(v,t) == -advection(v) + (u*u)*Tan(phi)/a + f*u - 1/(rho0 * a * Cos(phi))*D(p,lam) + friction_Hy + friction_V(v); // Eq. (2.2)

Equation velocities_eq  = [u,v,w] == Gradient(rho);
Equation hydrostatic_eq = D(p,z) == -rho*g;

M.setEquations({EOM1,EOM2,velocities_eq,hydrostatic_eq})
