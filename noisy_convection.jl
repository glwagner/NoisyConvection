using Oceananigans
using Oceananigans.Units
using Printf
using FFTW
using GLMakie
using Statistics

J̄ = 1e-6
N² = 1e-6

arch = CPU()
Nx = 96
Ny = 96
Nz = 64

Lx = 256
Ly = 256
Lz = 64

source = :constant
#source = :noise
#source = :delta

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

J = Field{Center, Center, Nothing}(grid)

if source === :constant
    set!(J, 1)

elseif source == :delta

    # Delta function source
    i = ceil(Int, Nx/2)
    j = ceil(Int, Ny/2)
    J[i, j, 1] = 1

elseif source == :noise
    # J ∘ filter
    set!(J, (x, y) -> randn())
    Ĵ = rfft(interior(J))

    f = Field{Center, Center, Nothing}(grid)
    δ = 32 # meters
    filter(x, y) = exp(-(x^2 + y^2) / 2δ^2)
    set!(f, filter)
    f̂ = rfft(interior(f))

    fJ = irfft(Ĵ .* f̂, Nx)
    fJ .-= minimum(fJ)

    set!(J, fJ)
end
    
# Normalize: ⟨J⟩ = J̄
J .*= J̄ / mean(J)

heatmap(interior(J, :, :, 1))
display(current_figure())

buoyancy_flux = FluxBoundaryCondition(J)
b_bcs = FieldBoundaryConditions(bottom=buoyancy_flux)

model = NonhydrostaticModel(; grid,
                            advection = WENO(order=5),
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=b_bcs))

bᵢ(x, y, z) = N² * z

w★ = (J̄ * Lz)^(1/3)
wᵢ(x, y, z) = 1e-3 * w★ * randn()

set!(model, b=bᵢ, u=wᵢ, w=wᵢ)

# h² ∼ 0.3 J * t / N²
# → t = h² N² / (0.3 J)
stop_h = Lz / 3
stop_time = stop_h^2 * N² / (0.3 * J̄)
@show stop_time / hours
simulation = Simulation(model; Δt=1, stop_time)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    msg = @sprintf("Iter: %s, time: %s, Δt: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt))

    u, v, w = sim.model.velocities
    msg *= @sprintf(", max|w|: %.2e m s⁻¹", maximum(abs, w))

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

run!(simulation)

b = model.tracers.b
u, v, w = model.velocities

fig = Figure()
axb = Axis(fig[1, 1])
axw = Axis(fig[1, 2])
heatmap!(axb, interior(b, :, 1, :))
heatmap!(axw, interior(w, :, 1, :))

display(fig)

