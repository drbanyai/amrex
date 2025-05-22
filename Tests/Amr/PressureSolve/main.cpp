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

struct LevelData
{
    LevelData(int n, amrex::Real domain_length)
    : geom(DefineGeometry(n, n, n, domain_length / n)),
      ba(DefineBoxArray(n, n, n)),
      dm(DefineDM(ba))
    {
    }
    const amrex::Geometry geom;
    const amrex::BoxArray ba;
    const amrex::DistributionMapping dm;
    amrex::MultiFab pressure;
    std::array<amrex::MultiFab, 3> velocity;
};

LevelData MakeDenseLevelData(int n, amrex::Real domain_length)
{
    LevelData level_data(n, domain_length);

    DefineFABs(level_data.pressure, level_data.velocity, level_data.ba, level_data.dm);
    InitializeVelocity(level_data.velocity, level_data.geom);
    return level_data;
}

std::vector<LevelData> MakeDenseCompositeLevels(int base_n, int nlevels, amrex::Real domain_length)
{
    std::vector<LevelData> composite_levels;
    for (int lev = 0; lev < nlevels; ++lev) {
        int n = base_n * (1 << lev);
        composite_levels.push_back(MakeDenseLevelData(n, domain_length));
    }
    return composite_levels;
}

int MyMain()
{
    // Number of levels (AMR-ready, even if only single-level for now)
    constexpr int nlevels = 2;
    constexpr int base_n = 4;
    constexpr amrex::Real domain_length = 1.0; // meters
    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr amrex::Real omega = 1.0;
    constexpr int max_iterations = 500;

    // First create and work with fine-level data
    const int fine_n = 8;
    LevelData fine_level = MakeDenseLevelData(fine_n, domain_length);
    std::vector<LevelData> composite_levels = MakeDenseCompositeLevels(base_n, nlevels, domain_length);

    // Create fine-level data structures
    amrex::Geometry fine_geom = fine_level.geom;
    amrex::BoxArray fine_ba = fine_level.ba;
    amrex::DistributionMapping fine_dm = fine_level.dm;
    amrex::MultiFab fine_pressure;
    std::array<amrex::MultiFab, 3> fine_velocity;
    DefineFABs(fine_pressure, fine_velocity, fine_ba, fine_dm);

    // Work with fine-level data
    InitializeVelocity(fine_velocity, fine_geom);
    amrex::Print() << "\nComplete fine solution, domain: " << fine_geom.Domain() << ", dx = " << fine_geom.CellSize()[0] << "\n";
    amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";
    SolvePressure(fine_pressure, fine_velocity, fine_geom, tolerance, max_iterations, omega);
    CheckResults(fine_pressure, fine_velocity, fine_geom);

    // AMReX containers for mesh and field data
    amrex::Vector<amrex::Geometry> geom(nlevels);
    amrex::Vector<amrex::BoxArray> ba(nlevels);
    amrex::Vector<amrex::DistributionMapping> dm(nlevels);
    amrex::Vector<amrex::MultiFab> pressure(nlevels);
    amrex::Vector<std::array<amrex::MultiFab, 3>> velocity(nlevels);

    // Loop over levels: setup geometry, mesh, fields
    // TODO: Need to generate proper AMR hierarchy
    for (int lev = 0; lev < nlevels; ++lev) {
        const int n = base_n * (1 << lev); // 4, 8, 16
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n > 0, "Grid size must be positive");
        const amrex::Real dx = domain_length / n;

        // Geometry and mesh setup (AMReX idioms)
        geom[lev] = DefineGeometry(n, n, n, dx);
        ba[lev] = DefineBoxArray(n, n, n);
        dm[lev] = DefineDM(ba[lev]);
        DefineFABs(pressure[lev], velocity[lev], ba[lev], dm[lev]);
        InitializeVelocity(velocity[lev], geom[lev]);
    }

    // Loop over levels: solve and check results
    // TODO: Need to implement full composite solve
    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::Print() << "\nLevel: " << lev << ", domain: " << geom[lev].Domain() << ", dx = " << geom[lev].CellSize()[0] << "\n";
        amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";
        SolvePressure(pressure[lev], velocity[lev], geom[lev], tolerance, max_iterations, omega);
        CheckResults(pressure[lev], velocity[lev], geom[lev]);
    }

    // Compare fine pressure with all levels
    CompareMultiFabs(fine_pressure, pressure, 
                    fine_geom, geom,
                    "fine pressure with AMR levels");

    // Sample and output results for all levels to a combined CSV file
    SamplePressureAlongLine(pressure, geom, "pressure.csv");

    return 0;
}


/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
