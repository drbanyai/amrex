/*
 *  Testbed implementing a single-level Gauss-Seidel pressure solver on a MAC grid.
 */

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX.H>

/*--------------------------------------------------------------------
  non-standard includes
  --------------------------------------------------------------------*/
#include "helpers.H"

/*--------------------------------------------------------------------
  forward declarations
  --------------------------------------------------------------------*/
int MyMain();

/*--------------------------------------------------------------------
  function definitions
  --------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    const int ret = MyMain();
    amrex::Finalize();
    return ret;
}

int MyMain()
{
    // Domain setup
    constexpr int nx = 32;
    constexpr int ny = 32;
    constexpr int nz = 32;
    constexpr double dx = 1.0;  // meters

    // Solver parameters
    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr int max_iterations = 1000;
    constexpr amrex::Real omega = 1.0;  // Under-relaxation parameter

    // Create geometry and mesh
    const amrex::Geometry geom = DefineGeometry(nx, ny, nz, dx);
    const amrex::BoxArray ba = DefineBoxArray(nx, ny, nz);
    const amrex::DistributionMapping dm = DefineDM(ba);

    // Allocate data structures
    amrex::MultiFab pressure;
    std::array<amrex::MultiFab, 3> velocity;
    DefineFABs(pressure, velocity, ba, dm);

    // Initialize velocity field
    InitializeVelocity(velocity, geom);

    // Allocate and compute divergence
    amrex::MultiFab divergence(ba, dm, 1, 0);
    ComputeDivergence(divergence, velocity, geom);

    // Initialize pressure to zero
    pressure.setVal(0.0);

    // Solve for pressure
    amrex::Real residual = 1.0;
    int iteration = 0;

    while (residual > tolerance && iteration < max_iterations)
    {
        GaussSeidelIteration(pressure, divergence, geom, omega);
        residual = ComputeResidual(pressure, divergence, geom);
        iteration++;

        if (iteration % 100 == 0)
        {
            amrex::Print() << "Iteration " << iteration << ", residual = " << residual << "\n";
        }
    }

    amrex::Print() << "Final iteration " << iteration << ", residual = " << residual << "\n";

    // Verify results
    CheckResults(pressure, velocity, geom);

    return 0;
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 