/*--------------------------------------------------------------------
  associated include
  --------------------------------------------------------------------*/
#include "helpers.H"

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX_BCUtil.H>
#include <AMReX_MultiFabUtil.H>

/*--------------------------------------------------------------------
  defines and static variables
  --------------------------------------------------------------------*/
// Helpers for indexing velocity components
static constexpr int U = 0;
static constexpr int V = 1;
static constexpr int W = 2;

/*--------------------------------------------------------------------
  free function definitions
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

    for (amrex::MFIter mfi(velocity[U]); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
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
                    // Set velocities to create constant divergence
                    const amrex::Real x = geom.CellCenter(i, U);
                    const amrex::Real y = geom.CellCenter(j, V);
                    const amrex::Real z = geom.CellCenter(k, W);

                    u_arr(i, j, k) = div * x / 3.0;
                    v_arr(i, j, k) = div * y / 3.0;
                    w_arr(i, j, k) = div * z / 3.0;
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

void ComputeDivergence(
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

void GaussSeidelIteration(
    amrex::MultiFab& pressure,
    const amrex::MultiFab& divergence,
    const amrex::Geometry& geom,
    amrex::Real omega)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dx2 = dx * dx;

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

    // Fill physical boundary ghost cells with proper boundary conditions
    amrex::Vector<amrex::BCRec> bc_lo, bc_hi;
    bc_lo.resize(1);
    bc_hi.resize(1);
    for (int n = 0; n < 1; ++n) {
        bc_lo[n].setLo(0, amrex::BCType::foextrap);
        bc_hi[n].setHi(0, amrex::BCType::foextrap);
        bc_lo[n].setLo(1, amrex::BCType::foextrap);
        bc_hi[n].setHi(1, amrex::BCType::foextrap);
        bc_lo[n].setLo(2, amrex::BCType::foextrap);
        bc_hi[n].setHi(2, amrex::BCType::foextrap);
    }
    amrex::FillDomainBoundary(pressure, geom, bc_lo);
}

amrex::Real ComputeResidual(
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
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 