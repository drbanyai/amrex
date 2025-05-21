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
void CompareMultiFabs(const amrex::MultiFab& expected_mf, const amrex::MultiFab& actual_mf, const std::string& name);

/*--------------------------------------------------------------------
  function definitions
  --------------------------------------------------------------------*/
void CompareMultiFabs(const amrex::MultiFab& expected_mf, const amrex::MultiFab& actual_mf, const std::string& name)
{
    amrex::Print() << "\nComparing " << name << ":\n";
    
    // Create a copy of expected_mf with the same distribution mapping as actual_mf
    amrex::MultiFab expected_remapped(actual_mf.boxArray(), actual_mf.DistributionMap(), 1, 0);
    expected_remapped.ParallelCopy(expected_mf);
    
    // Compute max absolute difference and L2 norm of difference
    amrex::Real max_diff = 0.0;
    amrex::Real l2_diff = 0.0;
    for (amrex::MFIter mfi(actual_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const auto& expected_fab = expected_remapped[mfi];
        const auto& actual_fab = actual_mf[mfi];
        
        for (int i = bx.loVect()[0]; i <= bx.hiVect()[0]; ++i) {
            for (int j = bx.loVect()[1]; j <= bx.hiVect()[1]; ++j) {
                for (int k = bx.loVect()[2]; k <= bx.hiVect()[2]; ++k) {
                    amrex::Real diff = std::abs(expected_fab(amrex::IntVect(i,j,k)) - 
                                              actual_fab(amrex::IntVect(i,j,k)));
                    max_diff = std::max(max_diff, diff);
                    l2_diff += diff * diff;
                }
            }
        }
    }
    
    // Reduce max_diff across all processes
    amrex::ParallelDescriptor::ReduceRealMax(max_diff);
    // Reduce l2_diff across all processes
    amrex::ParallelDescriptor::ReduceRealSum(l2_diff);
    l2_diff = std::sqrt(l2_diff);

    amrex::Print() << "  Maximum absolute difference: " << max_diff << "\n";
    amrex::Print() << "  L2 norm of difference: " << l2_diff << "\n";
}

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
    constexpr int nlevels = 1;
    constexpr int base_n = 8;
    constexpr amrex::Real domain_length = 1.0; // meters
    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr amrex::Real omega = 1.0;
    constexpr int max_iterations = 500;

    // First create and work with fine-level data
    const int fine_n = 8;
    const amrex::Real fine_dx = domain_length / fine_n;

    // Create fine-level data structures
    amrex::Geometry fine_geom = DefineGeometry(fine_n, fine_n, fine_n, fine_dx);
    amrex::BoxArray fine_ba = DefineBoxArray(fine_n, fine_n, fine_n);
    amrex::DistributionMapping fine_dm = DefineDM(fine_ba);
    amrex::MultiFab fine_pressure;
    std::array<amrex::MultiFab, 3> fine_velocity;
    DefineFABs(fine_pressure, fine_velocity, fine_ba, fine_dm);

    // Work with fine-level data
    InitializeVelocity(fine_velocity, fine_geom);
    SolvePressure(fine_pressure, fine_velocity, fine_geom, tolerance, max_iterations, omega);
    CheckResults(fine_pressure, fine_velocity, fine_geom);

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

        amrex::Print() << "\nLevel " << lev << ": domain size " << n << "x" << n << "x" << n << ", dx = " << dx << "\n";
        amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";

        // Velocity initialization, pressure solve, diagnostics
        InitializeVelocity(velocity[lev], geom[lev]);
        SolvePressure(pressure[lev], velocity[lev], geom[lev], tolerance, max_iterations, omega);
        CheckResults(pressure[lev], velocity[lev], geom[lev]);

        // Compare fine pressure with array pressure at finest level
        if (lev == nlevels-1) { // TODO: Compare all
            CompareMultiFabs(fine_pressure, pressure[lev], "fine pressure with pressure at level " + std::to_string(lev));
        }
    }

    // Sample and output results for all levels to a combined CSV file
    SamplePressureAlongLine(pressure, geom, "pressure.csv");

    return 0;
}


/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
