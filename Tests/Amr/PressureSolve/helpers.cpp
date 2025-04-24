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
static amrex::Real CalculateDipolePressure(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real z,
    const amrex::Real dipole_x,
    const amrex::Real dipole_y,
    const amrex::Real dipole_z,
    const amrex::Real dipole_strength,
    const int direction)
{
    const amrex::Real dx = x - dipole_x;
    const amrex::Real dy = y - dipole_y;
    const amrex::Real dz = z - dipole_z;
    const amrex::Real r2 = dx*dx + dy*dy + dz*dz;
    const amrex::Real r = std::sqrt(r2);
    
    amrex::Real p_x = 0.0;
    amrex::Real p_y = 0.0;
    amrex::Real p_z = 0.0;
    
    // Set dipole moment based on direction
    switch (direction) {
        case 0: p_x = dipole_strength; break;  // +x
        case 1: p_x = -dipole_strength; break; // -x
        case 2: p_y = dipole_strength; break;  // +y
        case 3: p_y = -dipole_strength; break; // -y
        case 4: p_z = dipole_strength; break;  // +z
        case 5: p_z = -dipole_strength; break; // -z
    }
    
    const amrex::Real r_dot_p = dx*p_x + dy*p_y + dz*p_z;
    return (1.0/(4.0*M_PI)) * r_dot_p / (r*r*r + 1e-6);
}

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

    // Evaluate the Green's function difference
    const amrex::Real eps = 1e-12; // avoid division by zero
    amrex::Real dist1 = std::sqrt((x - r1_x)*(x - r1_x) + (y - r1_y)*(y - r1_y) + (z - r1_z)*(z - r1_z) + eps);
    amrex::Real dist2 = std::sqrt((x - r2_x)*(x - r2_x) + (y - r2_y)*(y - r2_y) + (z - r2_z)*(z - r2_z) + eps);
    amrex::Real p = -1.0/(4.0*M_PI) * (1.0/dist1 - 1.0/dist2);
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
            u_arr(i_face, j_face, k_face) = 1.0;
        }
    }

    // Fill ghost cells
    for (int d = 0; d < 3; ++d) {
        velocity[d].FillBoundary(geom.periodicity());
    }
}

void SamplePressureAlongLine(
    const amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    const std::string& filename,
    int i_face, int j_face, int k_face)
{
    // Get domain center
    const amrex::Real center_x = 0.5 * (geom.ProbLo(U) + geom.ProbHi(U));
    const amrex::Real center_y = 0.5 * (geom.ProbLo(V) + geom.ProbHi(V));
    const amrex::Real center_z = 0.5 * (geom.ProbLo(W) + geom.ProbHi(W));
    
    // Find the cell indices closest to center
    const int center_i = static_cast<int>((center_x - geom.ProbLo(U)) / geom.CellSize(U));
    const int center_j = static_cast<int>((center_y - geom.ProbLo(V)) / geom.CellSize(V));
    const int center_k = static_cast<int>((center_z - geom.ProbLo(W)) / geom.CellSize(W));
    
    // Open file for writing
    std::ofstream outfile(filename);
    outfile << "# x p(x) p_expected(x) div(x)\n";
    
    // Sample along x-axis through center
    for (amrex::MFIter mfi(pressure); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        const auto& p_arr = pressure.array(mfi);
        const auto& div_arr = divergence.array(mfi);
        
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);
        
        // Only process if this box contains our line
        if (lo.y <= center_j && hi.y >= center_j &&
            lo.z <= center_k && hi.z >= center_k)
        {
            for (int i = lo.x; i <= hi.x; ++i)
            {
                const amrex::Real x = geom.CellCenter(i, U);
                const amrex::Real p = p_arr(i, center_j, center_k);
                const amrex::Real p_expected = ExpectedPressure(geom, i, center_j, center_k, i_face, j_face, k_face);
                const amrex::Real div = div_arr(i, center_j, center_k);
                outfile << x << ", " << p << ", " << p_expected << ", " << div << "\n";
            }
        }
    }
    outfile.close();
    
    // Create gnuplot script
    std::ofstream script("plot_pressure.gp");
    script << "set terminal png\n";
    script << "set output 'pressure_profile.png'\n";
    script << "set xlabel 'x'\n";
    script << "set ylabel 'pressure'\n";
    // script << "set logscale y\n";
    script << "plot '" << filename << "' using 1:2 title 'computed' with lines,\\\n";
    script << "     '" << filename << "' using 1:3 title 'expected' with lines,\\\n";
    script << "#    '" << filename << "' using 1:4 title 'divergence' with lines axis x1y2\n";
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
    
    // Sample pressure along center line and create plot
    SamplePressureAlongLine(pressure, divergence, geom, "pressure_profile.dat", i_face, j_face, k_face);
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
