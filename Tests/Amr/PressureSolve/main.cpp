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
    constexpr int multiplier = 16;
    constexpr int nx = multiplier ;
    constexpr int ny = multiplier;
    constexpr int nz = multiplier;
    constexpr double dx = 1.0/multiplier;  // meters

    // Solver parameters
    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr int max_iterations = 10*multiplier;
    constexpr amrex::Real omega = 1.0;  // Under-relaxation parameter

    amrex::Print() << "Running test with domain size: " << nx << "x" << ny << "x" << nz << "\n";
    amrex::Print() << "Solver parameters:\n";
    amrex::Print() << "  Tolerance: " << tolerance << "\n";
    amrex::Print() << "  Max iterations: " << max_iterations << "\n";
    amrex::Print() << "  Relaxation parameter: " << omega << "\n";

    // Create geometry and mesh
    const amrex::Geometry geom = DefineGeometry(nx, ny, nz, dx);
    const amrex::BoxArray ba = DefineBoxArray(nx, ny, nz);
    const amrex::DistributionMapping dm = DefineDM(ba);

    // Allocate data structures
    amrex::MultiFab pressure;
    std::array<amrex::MultiFab, 3> velocity;
    DefineFABs(pressure, velocity, ba, dm);

    // Define the nonzero face indices (hardcoded for this test)
    const int i_face = nx/2;
    const int j_face = ny/2;
    const int k_face = nz/2;

    // Initialize velocity field
    InitializeVelocity(velocity, geom, i_face, j_face, k_face);

    // Solve for pressure
    SolvePressure(pressure, velocity, geom, tolerance, max_iterations, omega, i_face, j_face, k_face);

    // Verify results
    CheckResults(pressure, velocity, geom, i_face, j_face, k_face);

    return 0;
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
