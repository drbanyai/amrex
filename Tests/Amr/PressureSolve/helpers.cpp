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
    const int k,
    int i_face, int j_face, int k_face)
{
    // Analytic solution for a dipole in infinite domain:
    // p(r) = -1/(4*pi) * (1/|r - r1| - 1/|r - r2|)
    const amrex::Real x = geom.CellCenter(i, U);
    const amrex::Real y = geom.CellCenter(j, V);
    const amrex::Real z = geom.CellCenter(k, W);

    // The nonzero x-face is at (i_face, j_face, k_face), between cells (i_face-1, j_face, k_face) and (i_face, j_face, k_face)
    const amrex::Real r1_x = geom.CellCenter(i_face-1, U); // cell to the left
    const amrex::Real r2_x = geom.CellCenter(i_face, U);   // cell to the right
    const amrex::Real r1_y = geom.CellCenter(j_face, V);
    const amrex::Real r2_y = geom.CellCenter(j_face, V);
    const amrex::Real r1_z = geom.CellCenter(k_face, W);
    const amrex::Real r2_z = geom.CellCenter(k_face, W);

    // Use cell-averaged inverse distance for source cells, pointwise otherwise
    const amrex::Real h = geom.CellSize(0); // Assume cubic cells
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
    PressureBndryFunc(int i_face_, int j_face_, int k_face_)
        : i_face(i_face_), j_face(j_face_), k_face(k_face_) {}
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
                    arr(i,j,k,n+dcomp) = ExpectedPressure(geom, i, j, k, i_face, j_face, k_face);
                }
            });
    }
private:
    int i_face, j_face, k_face;
};

static void GaussSeidelIteration(
    amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    amrex::Real omega,
    int iteration,
    int i_face, int j_face, int k_face);

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
    const amrex::Geometry& geom,
    int i_face, int j_face, int k_face)
{
    // Set all velocities to zero
    for (int d = 0; d < 3; ++d) {
        velocity[d].setVal(0.0);
    }

    // Set a single nonzero x-face at the specified face indices
    for (amrex::MFIter mfi(velocity[U]); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& u_arr = velocity[U].array(mfi);
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

void SamplePressureAlongLine(
    const amrex::Vector<amrex::MultiFab>& pressures,
    const amrex::Vector<amrex::Geometry>& geoms,
    const std::string& filename)
{
    const int finest_lev = static_cast<int>(pressures.size()) - 1;

    const amrex::Geometry& fine_geom = geoms[finest_lev];

    // Get centerline indices for y and z at finest level
    const amrex::Real center_y = 0.5 * (fine_geom.ProbLo(1) + fine_geom.ProbHi(1));
    const amrex::Real center_z = 0.5 * (fine_geom.ProbLo(2) + fine_geom.ProbHi(2));

    // Prepare: for each level, create a MultiFab on the finest grid
    amrex::Vector<amrex::MultiFab> fine_level_pressure(pressures.size());
    // Simple wrapper for AMReX FillPatchTwoLevels for cell-centered, single-component interpolation
    auto InterpFromCoarseToFineSimple = [](amrex::MultiFab& fine, const amrex::MultiFab& coarse,
                                           const amrex::Geometry& coarse_geom, const amrex::Geometry& fine_geom)
    {
        BL_PROFILE("InterpFromCoarseToFineSimple");
        AMREX_ALWAYS_ASSERT(fine.nComp() == 1 && coarse.nComp() == 1);
        amrex::IntVect ratio = fine_geom.Domain().size() / coarse_geom.Domain().size();
        // Set up BCRec (all interior Dirichlet for simplicity)
        amrex::Vector<amrex::BCRec> bcs(1, amrex::BCRec());
        // Set up coarse/fine state vectors
        amrex::Vector<amrex::MultiFab*> coarse_data{const_cast<amrex::MultiFab*>(&coarse)};
        amrex::Vector<amrex::MultiFab*> fine_data; // empty
        amrex::Vector<amrex::Real> time{0.0};
        amrex::Vector<amrex::Real> fine_time; // empty
        amrex::CellConservativeLinear interp;
        amrex::PhysBCFunctNoOp coarse_bc, fine_bc;
        amrex::FillPatchTwoLevels(
            fine, amrex::IntVect(0), amrex::Real(0.0),
            coarse_data, time,
            {&fine}, time,
            0, 0, 1,
            coarse_geom, fine_geom,
            coarse_bc, 0,
            fine_bc, 0,
            ratio, &interp, bcs, 0);
    };

    for (int lev = 0; lev <= finest_lev; ++lev) {
        // Define MultiFab with same structure as pressures[finest_lev]
        fine_level_pressure[lev].define(
            pressures[finest_lev].boxArray(),
            pressures[finest_lev].DistributionMap(),
            pressures[finest_lev].nComp(),
            pressures[finest_lev].nGrow());
        if (lev == finest_lev) {
            // Copy data from pressures[finest_lev]
            fine_level_pressure[lev].ParallelCopy(pressures[finest_lev]);
        } else {
            fine_level_pressure[lev].setVal(0.0);
            // Interpolate from coarse level to finest grid using AMReX FillPatchTwoLevels
            InterpFromCoarseToFineSimple(
                fine_level_pressure[lev],
                pressures[lev],
                geoms[lev],
                geoms[finest_lev]);
        }
    }

    // Prepare output
    std::ofstream outfile(filename);
    outfile << "x";
    for (int lev = 0; lev <= finest_lev; ++lev) {
        outfile << ",pressure_L" << lev;
    }
    outfile << ",expected\n";

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
        amrex::Real expected = ExpectedPressure(fine_geom, i, j, k, /*i_face=*/fine_geom.Domain().length(0)/2, /*j_face=*/fine_geom.Domain().length(1)/2, /*k_face=*/fine_geom.Domain().length(2)/2);
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
    amrex::Real omega,
    int i_face, int j_face, int k_face)
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
        GaussSeidelIteration(pressure, divergence, geom, omega, iteration, i_face, j_face, k_face);
        residual = ComputeResidual(pressure, divergence, geom);
        iteration++;

        if ((max_iterations < 100) || (iteration % 10 == 0))
        {
            amrex::Print() << "Iteration " << iteration << ", residual = " << residual << "\n";
        }
    }

    amrex::Print() << "Final iteration " << iteration << ", residual = " << residual << "\n";
}

void CheckResults(
    const amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom,
    int i_face, int j_face, int k_face)
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
                    const amrex::Real expected = ExpectedPressure(geom, i, j, k, i_face, j_face, k_face);
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

    const amrex::Real error_tolerance = 1.0E-2;
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
    int iteration,
    int i_face, int j_face, int k_face)
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
    PressureBndryFunc pbf(i_face, j_face, k_face);
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

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
