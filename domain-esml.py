from ESML import *
from ocean_model import lam,phi,z, EOM1, EOM2, velocities_eq, hydrostatic_eq

ocean_grid = GridData("world_map.h5", coordinates=[lam,phi,z])

ocean_grid.set_edge_bc(axis=0,        "periodic")
ocean_grid.set_edge_bc(axis=[1,2],    "no-flux")
ocean_grid.set_masked_bc("land_mask", "no-flux")

ocean_system = ODESystem(domain=ocean_grid, time=t,
                         equations=[EOM1,EOM2,velocities_eq, hydrostatic_eq],
                         time_integrator   = Integrators("Dormand-Prince"),  # NB: These are just named Butcher-tables, i.e. data; No need to program each separately
                         spatial_derivative= FiniteDifference("Central 5p"]) # NB: These are just named convolution coefficient tables, i.e. data.

biogeochemistry_system = ODESystem(domain=bio_grid, time=t,....);

earth_system = CoupledSystem([ocean_system, biogeochemistry_system], coupling_equations=[...])

earth_system.generate_module("earth_system",
                             static_analyses=[...],
                             runtime_analyses=[...],
                             target_hints=["GPGPU","largegrid",...]) # Generate fast code that can be integrated with legacy software
