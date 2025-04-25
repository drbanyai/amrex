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
    // Number of levels (AMR-ready, even if only single-level for now)
    constexpr int nlevels = 3;
    constexpr int base_n = 4;
    constexpr amrex::Real domain_length = 1.0; // meters
    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr amrex::Real omega = 1.0;
    constexpr int max_iterations = 500;

    // AMReX containers for mesh and field data
    amrex::Vector<amrex::Geometry> geom(nlevels);
    amrex::Vector<amrex::BoxArray> ba(nlevels);
    amrex::Vector<amrex::DistributionMapping> dm(nlevels);
    amrex::Vector<amrex::MultiFab> pressure(nlevels);
    amrex::Vector<std::array<amrex::MultiFab, 3>> velocity(nlevels);

    // Loop over levels: setup geometry, mesh, fields
    for (int lev = 0; lev < nlevels; ++lev) {
        const int n = base_n * (1 << lev); // 4, 8, 16
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n > 0, "Grid size must be positive");
        const amrex::Real dx = domain_length / n;

        // Geometry and mesh setup (AMReX idioms)
        geom[lev] = DefineGeometry(n, n, n, dx);
        ba[lev] = DefineBoxArray(n, n, n);
        dm[lev] = DefineDM(ba[lev]);
        DefineFABs(pressure[lev], velocity[lev], ba[lev], dm[lev]);

        // Centered face indices
        const int i_face = n/2;
        const int j_face = n/2;
        const int k_face = n/2;

        amrex::Print() << "\nLevel " << lev << ": domain size " << n << "x" << n << "x" << n << ", dx = " << dx << "\n";
        amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";

        // Velocity initialization, pressure solve, diagnostics
        InitializeVelocity(velocity[lev], geom[lev], i_face, j_face, k_face);
        SolvePressure(pressure[lev], velocity[lev], geom[lev], tolerance, max_iterations, omega, i_face, j_face, k_face);
        CheckResults(pressure[lev], velocity[lev], geom[lev], i_face, j_face, k_face);
    }

    // Sample and output results for all levels to a combined CSV file
    SamplePressureAlongLine(pressure, geom, "pressure_profile_levels.csv"); // Use base grid center as reference
    return 0;
}


/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
