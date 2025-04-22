/*--------------------------------------------------------------------
  associated include
  --------------------------------------------------------------------*/
#include "helpers.H"

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX_BCUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

/*--------------------------------------------------------------------
  defines and static variables
  --------------------------------------------------------------------*/
// Helpers for indexing velocity components
static constexpr int U = 0;
static constexpr int V = 1;
static constexpr int W = 2;
/*--------------------------------------------------------------------
  private free function declarations
  --------------------------------------------------------------------*/
static amrex::Real ExpectedPressure(
    const amrex::Geometry& geom,
    const int i,
    const int j,
    const int k)
{
    const amrex::Real x = geom.CellCenter(i, U);
    const amrex::Real y = geom.CellCenter(j, V);
    const amrex::Real z = geom.CellCenter(k, W);
    constexpr amrex::Real coeff = 1.0/6.0;
    return coeff * (x * x + y * y + z * z);
}

class PressureBndryFunc
{
public:
    void operator()(
        amrex::Box const& bx,
        amrex::FArrayBox& data,
        const int dcomp,
        const int numcomp,
        amrex::Geometry const& geom,
        const amrex::Real time,
        const amrex::Vector<amrex::BCRec>& bcr,
        const int bcomp,
        const int scomp) const
    {
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        const auto arr = data.array();
        const amrex::Box& valid_box = geom.Domain();

        amrex::ParallelFor(bx, numcomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                // Only operate on external ghost cells
                if (!valid_box.contains(i,j,k)) {
                    arr(i,j,k,n+dcomp) = ExpectedPressure(geom, i, j, k);
                }
            });
    }
};

static void GaussSeidelIteration(
    amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    amrex::Real omega,
    int iteration);

static amrex::Real ComputeResidual(
    const amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom);

static void ComputeDivergence(
    amrex::MultiFab& divergence,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom);

/*--------------------------------------------------------------------
  public free function definitions
  --------------------------------------------------------------------*/
amrex::Geometry DefineGeometry(int nx, int ny, int nz, double dx)
{
    const amrex::RealBox real_box({0.0, 0.0, 0.0},
                                 {nx * dx, ny * dx, nz * dx});
    constexpr amrex::CoordSys::CoordType coord = amrex::CoordSys::CoordType::cartesian;
    const amrex::IntArray is_periodic{0, 0, 0};  // Non-periodic in all directions

    const amrex::Box domain(amrex::IntVect(0, 0, 0),
                           amrex::IntVect(nx - 1, ny - 1, nz - 1));
    return amrex::Geometry(domain, real_box, coord, is_periodic);
}

amrex::BoxArray DefineBoxArray(int nx, int ny, int nz)
{
    const amrex::Box domain(amrex::IntVect(0, 0, 0),
                           amrex::IntVect(nx - 1, ny - 1, nz - 1));
    return amrex::BoxArray(domain);
}

amrex::DistributionMapping DefineDM(const amrex::BoxArray& ba)
{
    return amrex::DistributionMapping(ba);
}

void DefineFABs(
    amrex::MultiFab& pressure,
    std::array<amrex::MultiFab, 3>& velocity,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm)
{
    constexpr int SingleComp = 1;
    constexpr int PressureGhosts = 1;
    constexpr int VelocityGhosts = 1;

    // Pressure is cell-centered
    pressure.define(ba, dm, SingleComp, PressureGhosts);
    pressure.setVal(0.0);  // Initialize all cells (including ghosts) to zero

    // Velocity components are face-centered
    const auto Uba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(U));
    const auto Vba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(V));
    const auto Wba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(W));

    velocity[U].define(Uba, dm, SingleComp, VelocityGhosts);
    velocity[V].define(Vba, dm, SingleComp, VelocityGhosts);
    velocity[W].define(Wba, dm, SingleComp, VelocityGhosts);

    // Initialize velocity components to zero
    for (int d = 0; d < 3; ++d) {
        velocity[d].setVal(0.0);  // Initialize all cells (including ghosts) to zero
    }
}

void InitializeVelocity(
    std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom)
{
    // Initialize with a simple velocity field that has known divergence
    // For testing, we'll use a field with constant divergence
    const amrex::Real div = 1.0;  // Constant divergence

    for (int d = 0; d < 3; ++d) {
        for (amrex::MFIter mfi(velocity[d]); mfi.isValid(); ++mfi)
        {
            const amrex::Box& box = mfi.validbox();
            const auto& vel_arr = velocity[d].array(mfi);

            const auto lo = amrex::lbound(box);
            const auto hi = amrex::ubound(box);

            for (int i = lo.x; i <= hi.x; ++i)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int k = lo.z; k <= hi.z; ++k)
                    {
                        // Find the x-, y-, or z-coordinate of given face
                        const int ijk = (d == U) ? i : (d == V) ? j : k;
                        const amrex::Real xyz = geom.LoEdge(ijk, d);
                        vel_arr(i, j, k) = div * xyz / 3.0;
                    }
                }
            }
        }
    }

    // Fill ghost cells
    for (int d = 0; d < 3; ++d)
    {
        velocity[d].FillBoundary(geom.periodicity());
    }
}

void SolvePressure(
    amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom,
    amrex::Real tolerance,
    int max_iterations,
    amrex::Real omega)
{
    // Allocate and compute divergence
    amrex::MultiFab divergence(pressure.boxArray(), pressure.DistributionMap(), 1, 0);
    ComputeDivergence(divergence, velocity, geom);

    // Initialize pressure to zero
    pressure.setVal(0.0);

    // Solve for pressure
    amrex::Real residual = 1.0;
    int iteration = 0;

    while (residual > tolerance && iteration < max_iterations)
    {
        GaussSeidelIteration(pressure, divergence, geom, omega, iteration);
        residual = ComputeResidual(pressure, divergence, geom);
        iteration++;

        if ((max_iterations < 100) || (iteration % 100 == 0))
        {
            amrex::Print() << "Iteration " << iteration << ", residual = " << residual << "\n";
        }
    }

    amrex::Print() << "Final iteration " << iteration << ", residual = " << residual << "\n";
}

void CheckResults(
    const amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom)
{
    // For our test case with constant divergence, the analytic solution is
    // p = (div/6) * (x^2 + y^2 + z^2) + C
    const amrex::Real div = 1.0;  // Constant divergence from initialization
    const amrex::Real expected_coeff = div / 6.0;

    amrex::Real max_error = 0.0;
    amrex::Real avg_error = 0.0;
    amrex::Real volume = 0.0;

    for (amrex::MFIter mfi(pressure); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& p_arr = pressure.array(mfi);

        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int k = lo.z; k <= hi.z; ++k)
                {
                    const amrex::Real x = geom.CellCenter(i, U);
                    const amrex::Real y = geom.CellCenter(j, V);
                    const amrex::Real z = geom.CellCenter(k, W);

                    const amrex::Real expected = expected_coeff * (x * x + y * y + z * z);
                    const amrex::Real error = std::abs(p_arr(i, j, k) - expected);

                    max_error = std::max(max_error, error);
                    avg_error += error;
                    volume += 1.0;
                }
            }
        }
    }

    // Sum across processors
    amrex::ParallelDescriptor::ReduceRealMax(max_error);
    amrex::ParallelDescriptor::ReduceRealSum(avg_error);
    amrex::ParallelDescriptor::ReduceRealSum(volume);

    avg_error /= volume;
    amrex::Print() << "Pressure solution verification:\n";
    amrex::Print() << "  Maximum error: " << max_error << "\n";
    amrex::Print() << "  Average error: " << avg_error << "\n";

    const amrex::Real error_tolerance = 1.0;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(avg_error <= error_tolerance,
        "Average pressure error " + std::to_string(avg_error) +
        " exceeds maximum allowed value of " + std::to_string(error_tolerance));
}


/*--------------------------------------------------------------------
  private free function definitions
  --------------------------------------------------------------------*/
static void ComputeDivergence(
    amrex::MultiFab& divergence,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dxinv = 1.0 / dx;

    for (amrex::MFIter mfi(divergence); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& div_arr = divergence.array(mfi);
        const auto& u_arr = velocity[U].array(mfi);
        const auto& v_arr = velocity[V].array(mfi);
        const auto& w_arr = velocity[W].array(mfi);

        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int k = lo.z; k <= hi.z; ++k)
                {
                    // Compute divergence using central differences
                    div_arr(i, j, k) = dxinv * (
                        u_arr(i + 1, j, k) - u_arr(i, j, k) +
                        v_arr(i, j + 1, k) - v_arr(i, j, k) +
                        w_arr(i, j, k + 1) - w_arr(i, j, k));
                }
            }
        }
    }
}

static void GaussSeidelIteration(
    amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    amrex::Real omega,
    int iteration)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dx2 = dx * dx;

    // Define boundary conditions
    amrex::Vector<amrex::BCRec> bc(1);
    
    // Set Dirichlet boundary conditions based on expected solution
    for (int n = 0; n < 1; ++n) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            bc[n].setLo(dir, amrex::BCType::ext_dir);  // External Dirichlet
            bc[n].setHi(dir, amrex::BCType::ext_dir);  // External Dirichlet
        }
    }

    // Create boundary condition functor
    PressureBndryFunc pbf;
    amrex::PhysBCFunct<PressureBndryFunc> physbc(geom, bc, pbf);

    // Fill ghost cells with boundary conditions
    const int start_comp = 0;  // Starting component
    const int num_comp = 1;    // Number of components
    const amrex::IntVect nghost(1);  // Ghost cell width
    const amrex::Real time = 0.0;    // Time
    const int bccomp = 0;      // Starting component for boundary conditions
    physbc.FillBoundary(pressure, start_comp, num_comp, nghost, time, bccomp);

    // Now do standard Gauss-Seidel iteration for all cells in the domain
    for (amrex::MFIter mfi(pressure); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& p_arr = pressure.array(mfi);
        const auto& div_arr = divergence.array(mfi);

        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int k = lo.z; k <= hi.z; ++k)
                {
                    // Gauss-Seidel update with under-relaxation
                    const amrex::Real p_new = (1.0 / 6.0) * (
                        p_arr(i + 1, j, k) + p_arr(i - 1, j, k) +
                        p_arr(i, j + 1, k) + p_arr(i, j - 1, k) +
                        p_arr(i, j, k + 1) + p_arr(i, j, k - 1) -
                        dx2 * div_arr(i, j, k));

                    p_arr(i, j, k) = (1.0 - omega) * p_arr(i, j, k) + omega * p_new;
                }
            }
        }

        // Print pressure at a cell near the center
        const auto& domain = geom.Domain();
        const int center_i = domain.length(0) / 2;
        const int center_j = domain.length(1) / 2;
        const int center_k = domain.length(2) / 2;

        if (lo.x <= center_i && hi.x >= center_i &&
            lo.y <= center_j && hi.y >= center_j &&
            lo.z <= center_k && hi.z >= center_k) {
            const amrex::Real expected = ExpectedPressure(geom, center_i, center_j, center_k);
            amrex::Print() << "Pressure at center cell (" 
                          << center_i << "," << center_j << "," << center_k << "): " 
                          << p_arr(center_i,center_j,center_k)
                          << " (expected: " << expected << ")\n";
        }
    }

    // Fill internal ghost cells between patches
    pressure.FillBoundary(geom.periodicity());
}

static amrex::Real ComputeResidual(
    const amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dx2 = dx * dx;

    amrex::Real residual = 0.0;
    amrex::Real volume = 0.0;

    for (amrex::MFIter mfi(pressure); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& p_arr = pressure.array(mfi);
        const auto& div_arr = divergence.array(mfi);

        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int k = lo.z; k <= hi.z; ++k)
                {
                    // Compute residual using central differences
                    const amrex::Real lap_p = (
                        p_arr(i + 1, j, k) + p_arr(i - 1, j, k) +
                        p_arr(i, j + 1, k) + p_arr(i, j - 1, k) +
                        p_arr(i, j, k + 1) + p_arr(i, j, k - 1) -
                        6.0 * p_arr(i, j, k)) / dx2;

                    const amrex::Real res = lap_p - div_arr(i, j, k);
                    residual += res * res;
                    volume += 1.0;
                }
            }
        }
    }

    // Sum across processors
    amrex::ParallelDescriptor::ReduceRealSum(residual);
    amrex::ParallelDescriptor::ReduceRealSum(volume);

    return std::sqrt(residual / volume);
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
