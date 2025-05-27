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
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>

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
    int i,
    int j,
    int k)
{
    // Analytic solution for a dipole in infinite domain:
    // p(r) = -1/(4*pi) * (1/|r - r1| - 1/|r - r2|)
    const amrex::Real x = geom.CellCenter(i, U);
    const amrex::Real y = geom.CellCenter(j, V);
    const amrex::Real z = geom.CellCenter(k, W);

    // The nonzero x-face is at (coarse_face[0], coarse_face[1], coarse_face[2]), between cells (coarse_face[0]-1, coarse_face[1], coarse_face[2]) and (coarse_face[0], coarse_face[1], coarse_face[2])
    const amrex::Real centerMinus = 3.5/8.0; // 15.5/32.0;  // Cell center of cell to the left of the face
    const amrex::Real centerPlus = 4.5/8.0; // 16.5/32.0;   // Cell center of cell to the right of the face

    const amrex::Real r1_x = centerMinus;   // reference_geom.CellCenter(coarse_face[0]-1, U); // cell to the left
    const amrex::Real r2_x = centerPlus;    // reference_geom.CellCenter(coarse_face[0], U);   // cell to the right
    const amrex::Real r1_y = centerPlus;    // reference_geom.CellCenter(coarse_face[1], V);
    const amrex::Real r2_y = centerPlus;    // reference_geom.CellCenter(coarse_face[1], V);
    const amrex::Real r1_z = centerPlus;    // reference_geom.CellCenter(coarse_face[2], W);
    const amrex::Real r2_z = centerPlus;    // reference_geom.CellCenter(coarse_face[2], W);

    // Use cell-averaged inverse distance for source cells, pointwise otherwise
    const amrex::Real h = 1.0/8.0;
    const amrex::Real avg_inv_r = 2.0 * 1.516386 / h; // <1/r> over the cube

    const amrex::Real dist1 = std::sqrt((x - r1_x)*(x - r1_x) + (y - r1_y)*(y - r1_y) + (z - r1_z)*(z - r1_z));
    const amrex::Real dist2 = std::sqrt((x - r2_x)*(x - r2_x) + (y - r2_y)*(y - r2_y) + (z - r2_z)*(z - r2_z));

    // Use cell-averaged inverse distance if the evaluation point coincides with the source, otherwise use pointwise inverse distance
    const amrex::Real tol = 1e-10;
    const amrex::Real inv_dist1 = (dist1 < tol) ? avg_inv_r : 1.0/dist1;
    const amrex::Real inv_dist2 = (dist2 < tol) ? avg_inv_r : 1.0/dist2;
    const amrex::Real p = -1.0/(4.0*M_PI) * (inv_dist1 - inv_dist2);
    return p;
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
static void SolvePressureIterations(
    amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    amrex::Real tolerance,
    int max_iterations,
    amrex::Real omega);

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

amrex::BoxArray DefineSparseBoxArray(int nx, int ny, int nz)
{
    const amrex::Box domain(amrex::IntVect(nx/4, ny/4, nz/4),
                           amrex::IntVect(3*nx/4 - 1, 3*ny/4 - 1, 3*nz/4 - 1));
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
    // Set all velocities to zero
    for (int d = 0; d < 3; ++d) {
        velocity[d].setVal(0.0);
    }

    if (geom.Domain().length(0) == 8) {
        const int i_face = 4;
        const int j_face = 4;
        const int k_face = 4;

        // Set a single nonzero x-face at the specified face indices
        for (amrex::MFIter mfi(velocity[U]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &box = mfi.validbox();
            const auto &u_arr = velocity[U].array(mfi);
            if (box.contains(i_face, j_face, k_face)) {
                const amrex::Real dx = geom.CellSize(0);
                u_arr(i_face, j_face, k_face) = 1.0 / (dx * dx);
            }
        }

        // Fill ghost cells
        for (int d = 0; d < 3; ++d) {
            velocity[d].FillBoundary(geom.periodicity());
        }
    }
}

void SamplePressureAlongLine(
    const std::vector<LevelData>& level_data,
    const std::string& filename)
{
    const int finest_lev = static_cast<int>(level_data.size()) - 1;
    const amrex::Geometry& fine_geom = level_data[finest_lev].geom;

    // Get centerline indices for y and z at finest level
    const amrex::Real center_y = 0.5 * (fine_geom.ProbLo(1) + fine_geom.ProbHi(1));
    const amrex::Real center_z = 0.5 * (fine_geom.ProbLo(2) + fine_geom.ProbHi(2));

    // Prepare: for each level, create a MultiFab on the finest grid
    amrex::Vector<amrex::MultiFab> fine_level_pressure(level_data.size());

    // Simple wrapper for AMReX InterpFromCoarseLevel for cell-centered, single-component interpolation
    auto InterpFromCoarseToFineSimple = [](amrex::MultiFab& fine, const amrex::MultiFab& coarse,
                                           const amrex::Geometry& coarse_geom, const amrex::Geometry& fine_geom)
    {
        BL_PROFILE("InterpFromCoarseToFineSimple");
        AMREX_ALWAYS_ASSERT(fine.nComp() == 1 && coarse.nComp() == 1);
        
        // Calculate refinement ratio
        amrex::IntVect ratio = fine_geom.Domain().size() / coarse_geom.Domain().size();
        AMREX_ALWAYS_ASSERT(ratio[0] == ratio[1] && ratio[1] == ratio[2]); // Ensure uniform refinement
        
        // Use cell-centered conservative interpolation
        amrex::CellConservativeLinear interp;
        
        // Set up boundary conditions (one component)
        amrex::Vector<amrex::BCRec> bcr(1);
        
        // Interpolate from coarse to fine level
        amrex::InterpFromCoarseLevel(fine,      // destination MultiFab
                                    amrex::IntVect{}, amrex::IntVect{},
                                    coarse,    // source MultiFab
                                    0,         // source component
                                    0,         // destination component
                                    1,         // number of components
                                    coarse_geom,  // coarse geometry
                                    fine_geom,    // fine geometry
                                    ratio,        // refinement ratio
                                    &interp,
                                    bcr, 0);     // interpolation operator
    };

    for (int lev = 0; lev <= finest_lev; ++lev) {
        // Define MultiFab with same structure as level_data[finest_lev].pressure
        fine_level_pressure[lev].define(
            level_data[finest_lev].pressure.boxArray(),
            level_data[finest_lev].pressure.DistributionMap(),
            level_data[finest_lev].pressure.nComp(),
            level_data[finest_lev].pressure.nGrow());
            
        if (lev == finest_lev) {
            // Copy data from level_data[finest_lev].pressure
            fine_level_pressure[lev].ParallelCopy(level_data[finest_lev].pressure);
        } else {
            // Interpolate from coarse level to finest grid
            InterpFromCoarseToFineSimple(
                fine_level_pressure[lev],
                level_data[lev].pressure,
                level_data[lev].geom,
                level_data[finest_lev].geom);
        }
    }

    // Prepare output
    std::ofstream outfile(filename);
    outfile << "x";
    for (int lev = 0; lev <= finest_lev; ++lev) {
        outfile << ",pressure_L" << lev;
    }
    outfile << ",expected\n";

    // Sample along x-axis at center_y, center_z
    const amrex::Box& domain = fine_geom.Domain();
    for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
        amrex::Real x = fine_geom.CellCenter(i, 0);
        outfile << x;
        
        // For each level, sample pressure at (i, center_j, center_k) on the finest grid
        int j = static_cast<int>((center_y - fine_geom.ProbLo(1)) / fine_geom.CellSize(1));
        int k = static_cast<int>((center_z - fine_geom.ProbLo(2)) / fine_geom.CellSize(2));
        for (int lev = 0; lev <= finest_lev; ++lev) {
            amrex::Real val = 0.0;
            for (amrex::MFIter mfi(fine_level_pressure[lev]); mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                if (box.contains(amrex::IntVect(i, j, k))) {
                    const auto& parr = fine_level_pressure[lev].array(mfi);
                    val = parr(i, j, k);
                    break;
                }
            }
            outfile << "," << val;
        }
        
        // Add analytic solution as last column
        amrex::Real expected = ExpectedPressure(fine_geom, i, j, k);
        outfile << "," << expected;
        outfile << "\n";
    }
    outfile.close();
    
    // Create gnuplot script
    std::ofstream script("plot_pressure.gp");
    script << "set terminal png size 800,600\n";
    script << "set output 'pressure_profile.png'\n";
    script << "set xlabel 'x (m)'\n";
    script << "set ylabel 'Pressure (Pa)'\n";
    script << "set xzeroaxis\n";
    script << "set datafile separator ','\n";
    script << "plot ";
    for (int lev = 0; lev <= finest_lev; ++lev) {
        if (lev > 0) script << ", ";
        script << "'" << filename << "' using 1:" << (lev+2) << " title 'Level " << lev << "' with linespoints";
    }
    script << ", '" << filename << "' using 1:" << (finest_lev+3) << " title 'Analytic' with linespoints\n";
    script << "\n";
    script << "set output 'pressure_error.png'\n";
    script << "set xlabel 'x (m)'\n";
    script << "set ylabel 'Pressure Error (Pa)'\n";
    script << "set xzeroaxis\n";
    script << "set datafile separator ','\n";
    // Error: finest level minus analytic
    script << "plot '" << filename << "' using 1:($" << (finest_lev+2) << "-$" << (finest_lev+3) << ") title 'Finest - Analytic' with linespoints\n";
    script.close();

    // Run gnuplot
    std::system("gnuplot plot_pressure.gp");
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

    // Solve for pressure using iterations
    SolvePressureIterations(pressure, divergence, geom, tolerance, max_iterations, omega);
}

void CheckResults(
    const amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom)
{
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
                    const amrex::Real expected = ExpectedPressure(geom, i, j, k);
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
    const amrex::Real error_tolerance = 1.0E-2;
    amrex::Print() << "Pressure solution verification:\n";
    amrex::Print() << "  Maximum error: " << max_error << "\n";
    amrex::Print() << "  Average error: " << avg_error << "\n";
    amrex::Print() << "  Error tolerance: " << error_tolerance << "\n";
    amrex::Print() << "  Average error relative to tolerance: " << (avg_error/error_tolerance) << "\n";
}

void CompareMultiFabs(
    const LevelData& expected_level,
    const std::vector<LevelData>& level_data)
{
    const auto& expected_mf = expected_level.pressure;
    const auto& expected_geom = expected_level.geom;

    for (int lev = 0; lev < level_data.size(); ++lev) {
        const auto& level_mf = level_data[lev].pressure;
        const auto& level_geom = level_data[lev].geom;

        // Check real boxes match
        if ( !amrex::AlmostEqual(expected_geom.ProbDomain(), level_geom.ProbDomain()) ) {
            amrex::Print() << "\nReal box mismatch at level " << lev << ":\n"
                          << "  Expected: " << expected_geom.ProbDomain() << "\n"
                          << "  Level:    " << level_geom.ProbDomain() << "\n";
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::AlmostEqual(expected_geom.ProbDomain(), level_geom.ProbDomain()),
            "Real boxes must match between expected and level geometries");

        amrex::Print() << "\nComparing pressure at level " << lev << ":\n";

        // Create a copy of expected_mf with the same distribution mapping as level_mf
        amrex::MultiFab expected_remapped(level_mf.boxArray(), level_mf.DistributionMap(), 1, 0);

        if (expected_geom.Domain() == level_geom.Domain()) {
            // TODO: Can this also use average_down()?
            expected_remapped.ParallelCopy(expected_mf);
        } else {
            // Calculate refinement ratio between expected and level geometries
            amrex::IntVect ratio = expected_geom.Domain().size() / level_geom.Domain().size();
            
            // Assert that expected_geom is a refined version of level_geom
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ratio[0] >= 1 && ratio[1] >= 1 && ratio[2] >= 1,
                "Expected geometry must be refined version of level geometry");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ratio[0] == ratio[1] && ratio[1] == ratio[2],
                "Refinement ratio must be uniform in all dimensions");

            // Average down (restrict) from fine to coarse grid
            amrex::average_down(expected_mf, expected_remapped, 0, 1, ratio);

            amrex::Print() << "\nRestricted expected data from fine grid (ratio=" 
                          << ratio[0] << ") to coarse grid for level " << lev << "\n";
        }

        // Compute max absolute difference and L2 norm of difference
        amrex::Real max_diff = 0.0;
        amrex::Real l2_diff = 0.0;
        amrex::Real volume = 0.0; // Add volume for proper L2 norm calculation

        for (amrex::MFIter mfi(level_mf); mfi.isValid(); ++mfi) {
            const amrex::Box &bx = mfi.validbox();
            const auto &expected_fab = expected_remapped[mfi];
            const auto &level_fab = level_mf[mfi];

            for (int i = bx.loVect()[0]; i <= bx.hiVect()[0]; ++i) {
                for (int j = bx.loVect()[1]; j <= bx.hiVect()[1]; ++j) {
                    for (int k = bx.loVect()[2]; k <= bx.hiVect()[2]; ++k) {
                        amrex::Real diff = std::abs(expected_fab(amrex::IntVect(i, j, k)) -
                                                    level_fab(amrex::IntVect(i, j, k)));
                        max_diff = std::max(max_diff, diff);
                        l2_diff += diff * diff;
                        volume += 1.0;
                    }
                }
            }
        }

        // Reduce across all processes
        amrex::ParallelDescriptor::ReduceRealMax(max_diff);
        amrex::ParallelDescriptor::ReduceRealSum(l2_diff);
        amrex::ParallelDescriptor::ReduceRealSum(volume);
        
        // Compute normalized L2 norm
        l2_diff = std::sqrt(l2_diff / volume);

        amrex::Print() << "  Maximum absolute difference: " << max_diff << "\n";
        amrex::Print() << "  L2 norm of difference: " << l2_diff << "\n";
        amrex::Print() << "  Cell volume: " << volume << "\n";

        // Check if differences exceed tolerance
        constexpr amrex::Real diff_tolerance = 1.0e-3;
        if (max_diff > diff_tolerance) {
            amrex::Print() << "\nWARNING: Maximum difference exceeds tolerance (" << diff_tolerance << ")!\n";
        }
        constexpr amrex::Real l2_tolerance = 1.0e-3;
        if (l2_diff > l2_tolerance) {
            amrex::Print() << "\nWARNING: L2 norm of difference exceeds tolerance (" << l2_tolerance << ")!\n";
        }
    }
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

    // Track maximum divergence for diagnostics
    amrex::Real max_div = 0.0;

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
                    
                    max_div = std::max(max_div, std::abs(div_arr(i, j, k)));
                }
            }
        }
    }

    // Print maximum divergence for diagnostics
    amrex::ParallelDescriptor::ReduceRealMax(max_div);
    amrex::Print() << "Maximum divergence: " << max_div << "\n";
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
    
    // Set Dirichlet boundary conditions based on analytic solution
    for (int n = 0; n < 1; ++n) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            bc[n].setLo(dir, amrex::BCType::ext_dir);  // External Dirichlet
            bc[n].setHi(dir, amrex::BCType::ext_dir);  // External Dirichlet
        }
    }

    // Create boundary condition functor
    PressureBndryFunc pbf{};
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

static void SolvePressureIterations(
    amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    amrex::Real tolerance,
    int max_iterations,
    amrex::Real omega)
{
    amrex::Real residual = 1.0;
    int iteration = 0;

    while (residual > tolerance && iteration < max_iterations)
    {
        GaussSeidelIteration(pressure, divergence, geom, omega, iteration);
        residual = ComputeResidual(pressure, divergence, geom);
        iteration++;

        if ((max_iterations < 100) || (iteration % 10 == 0))
        {
            amrex::Print() << "Iteration " << iteration << ", residual = " << residual << "\n";
        }
    }

    amrex::Print() << "Final iteration " << iteration << ", residual = " << residual << "\n";
}

LevelData::LevelData(int n, amrex::Real domain_length)
    : geom(DefineGeometry(n, n, n, domain_length / n))
{
}

LevelData MakeDenseLevelData(int n, amrex::Real domain_length)
{
    LevelData level_data(n, domain_length);
    const amrex::BoxArray ba = DefineBoxArray(n, n, n);
    const amrex::DistributionMapping dm = DefineDM(ba);
    DefineFABs(level_data.pressure, level_data.velocity, ba, dm);
    InitializeVelocity(level_data.velocity, level_data.geom);
    return level_data;
}

LevelData MakeSparseLevelData(int n, amrex::Real domain_length)
{
    LevelData level_data(n, domain_length);
    const amrex::BoxArray ba = DefineSparseBoxArray(n, n, n);
    const amrex::DistributionMapping dm = DefineDM(ba);
    DefineFABs(level_data.pressure, level_data.velocity, ba, dm);
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

std::vector<LevelData> MakeSparseCompositeLevels(int base_n, int nlevels, amrex::Real domain_length)
{
    std::vector<LevelData> composite_levels;
    for (int lev = 0; lev < nlevels; ++lev) {
        int n = base_n * (1 << lev);
        if (lev == 0) {
            composite_levels.push_back(MakeDenseLevelData(n, domain_length));
        } else {
            composite_levels.push_back(MakeSparseLevelData(n, domain_length));
        }
    }
    return composite_levels;
}

void FillPressureGhostCells(LevelData& fine_level, const LevelData& crse_level)
{
    // Set up boundary conditions for pressure
    amrex::Vector<amrex::BCRec> bcs(1);  // One component for pressure
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        bcs[0].setLo(idim, amrex::BCType::ext_dir);
        bcs[0].setHi(idim, amrex::BCType::ext_dir);
    }

    // Create boundary condition functors
    PressureBndryFunc cbc;  // No constructor parameters needed
    PressureBndryFunc fbc;  // No constructor parameters needed
    amrex::PhysBCFunct<PressureBndryFunc> cphysbc(crse_level.geom, bcs, cbc);
    amrex::PhysBCFunct<PressureBndryFunc> fphysbc(fine_level.geom, bcs, fbc);

    // Create or reuse FillPatcher if not already initialized
    if (!fine_level.fillpatcher) {
        const amrex::IntVect nghost(1);  // Number of ghost cells to fill
        const int ncomp = 1;  // One component for pressure
        fine_level.fillpatcher = std::make_unique<amrex::FillPatcher<amrex::MultiFab>>(
            fine_level.pressure.boxArray(),
            fine_level.pressure.DistributionMap(),
            fine_level.geom,
            crse_level.pressure.boxArray(),
            crse_level.pressure.DistributionMap(),
            crse_level.geom,
            nghost,
            ncomp,
            &amrex::pc_interp
        );
    }

    // Fill ghost cells using FillPatcher
    const amrex::IntVect nghost(1);
    const amrex::Real time = 0.0;  // Time is not used in this case
    amrex::Vector<amrex::MultiFab*> cmf = {const_cast<amrex::MultiFab*>(&crse_level.pressure)};
    amrex::Vector<amrex::Real> ct = {time};
    amrex::Vector<amrex::MultiFab*> fmf = {&fine_level.pressure};
    amrex::Vector<amrex::Real> ft = {time};

    fine_level.fillpatcher->fill(
        fine_level.pressure,  // Destination
        nghost,              // Number of ghost cells to fill
        time,                // Time
        cmf,                 // Coarse level data
        ct,                  // Coarse level times
        fmf,                 // Fine level data
        ft,                  // Fine level times
        0,                   // Source component
        0,                   // Destination component
        1,                   // Number of components
        cphysbc,            // Coarse level boundary conditions
        0,                   // Coarse level boundary condition component
        fphysbc,            // Fine level boundary conditions
        0,                   // Fine level boundary condition component
        bcs,                // Boundary conditions
        0                   // Boundary condition component
    );
}

void SolvePressureCorrection(
    amrex::MultiFab& crse_pressure,
    const amrex::MultiFab& fine_pressure,
    const amrex::Geometry& crse_geom,
    const amrex::Geometry& fine_geom,
    amrex::Real tolerance,
    int max_iterations,
    amrex::Real omega)
{
    // TODO: Calculate flux mismatch between coarse and fine levels
    // TODO: Calculate correction solve on coarse level
    // TODO: Add correction to coarse level
    // Calculate refinement ratio
    amrex::IntVect ratio = fine_geom.Domain().size() / crse_geom.Domain().size();
    AMREX_ALWAYS_ASSERT(ratio[0] == ratio[1] && ratio[1] == ratio[2]);  // Uniform refinement

    // Create a temporary MultiFab to store the averaged fine pressure
    amrex::MultiFab avg_fine_pressure(crse_pressure.boxArray(), crse_pressure.DistributionMap(), 1, 0);

    // Average down fine pressure to coarse grid
    amrex::average_down(fine_pressure, avg_fine_pressure, 0, 1, ratio);

    // Compute correction as difference between averaged fine and coarse pressure
    amrex::MultiFab correction(crse_pressure.boxArray(), crse_pressure.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(correction, avg_fine_pressure, 0, 0, 1, 0);  // Copy averaged fine pressure
    amrex::MultiFab::Subtract(correction, crse_pressure, 0, 0, 1, 0);  // Subtract coarse pressure

    // Compute divergence of correction field
    amrex::MultiFab correction_div(correction.boxArray(), correction.DistributionMap(), 1, 0);
    correction_div.setVal(0.0);  // Initialize to zero

    // Compute divergence using central differences
    const amrex::Real dx = crse_geom.CellSize(0);
    const amrex::Real dxinv = 1.0 / dx;

    for (amrex::MFIter mfi(correction_div); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& div_arr = correction_div.array(mfi);
        const auto& corr_arr = correction.array(mfi);

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
                        corr_arr(i + 1, j, k) - corr_arr(i - 1, j, k) +
                        corr_arr(i, j + 1, k) - corr_arr(i, j - 1, k) +
                        corr_arr(i, j, k + 1) - corr_arr(i, j, k - 1)) / (2.0 * AMREX_SPACEDIM);
                }
            }
        }
    }

    // Solve for correction using the same solver as pressure
    amrex::MultiFab correction_solution(correction.boxArray(), correction.DistributionMap(), 1, 0);
    correction_solution.setVal(0.0);  // Initialize to zero

    // Use the same iteration solver as pressure
    SolvePressureIterations(correction_solution, correction_div, crse_geom, tolerance, max_iterations, omega);

    // Add correction to coarse pressure
    amrex::MultiFab::Add(crse_pressure, correction_solution, 0, 0, 1, 0);

    // Print statistics about the correction
    amrex::Real max_correction = 0.0;
    amrex::Real avg_correction = 0.0;
    amrex::Real volume = 0.0;

    for (amrex::MFIter mfi(correction_solution); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& corr_arr = correction_solution.array(mfi);

        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);

        for (int i = lo.x; i <= hi.x; ++i)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int k = lo.z; k <= hi.z; ++k)
                {
                    const amrex::Real corr = std::abs(corr_arr(i, j, k));
                    max_correction = std::max(max_correction, corr);
                    avg_correction += corr;
                    volume += 1.0;
                }
            }
        }
    }

    // Reduce across processors
    amrex::ParallelDescriptor::ReduceRealMax(max_correction);
    amrex::ParallelDescriptor::ReduceRealSum(avg_correction);
    amrex::ParallelDescriptor::ReduceRealSum(volume);

    avg_correction /= volume;

    amrex::Print() << "\nPressure correction statistics:\n"
                   << "  Maximum correction: " << max_correction << "\n"
                   << "  Average correction: " << avg_correction << "\n"
                   << "  Number of cells: " << volume << "\n";
}
/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
