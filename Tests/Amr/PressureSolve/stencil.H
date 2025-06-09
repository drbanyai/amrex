#pragma once

#include <AMReX_Array.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

namespace PressureSolve {

// Enum to select stencil type at runtime
enum class StencilType {
  Standard5Point,  // Standard 5-point (2D) / 7-point (3D) stencil
  HOC9Point,       // 19-point (3D) HOC stencil (Spotz 1996)
};

// Base class for stencil operations
class StencilBase
{
public:
  virtual ~StencilBase() = default;

  // Compute Laplacian at a point
  virtual amrex::Real computeLaplacian(
    const amrex::Array4<const amrex::Real>& p,
    int i,
    int j,
    int k,
    amrex::Real dx2 ) const = 0;

  // Compute Gauss-Seidel update
  virtual amrex::Real computeGaussSeidelUpdate(
    const amrex::Array4<const amrex::Real>& p,
    const amrex::Array4<const amrex::Real>& div,
    int i,
    int j,
    int k,
    amrex::Real dx2 ) const = 0;

  // Get the number of ghost cells needed
  virtual int getNumGhostCells() const = 0;
};

// Standard 5-point (2D) / 7-point (3D) stencil
class StandardStencil : public StencilBase
{
public:
  amrex::Real computeLaplacian( const amrex::Array4<const amrex::Real>& p,
                                int i,
                                int j,
                                int k,
                                amrex::Real dx2 ) const override
  {
    return ( p( i + 1, j, k ) + p( i - 1, j, k ) + p( i, j + 1, k ) +
             p( i, j - 1, k ) + p( i, j, k + 1 ) + p( i, j, k - 1 ) -
             6.0 * p( i, j, k ) ) /
           dx2;
  }

  amrex::Real computeGaussSeidelUpdate(
    const amrex::Array4<const amrex::Real>& p,
    const amrex::Array4<const amrex::Real>& div,
    int i,
    int j,
    int k,
    amrex::Real dx2 ) const override
  {
    constexpr amrex::Real one_sixth = 1.0 / 6.0;
    return one_sixth *
           ( p( i + 1, j, k ) + p( i - 1, j, k ) + p( i, j + 1, k ) +
             p( i, j - 1, k ) + p( i, j, k + 1 ) + p( i, j, k - 1 ) -
             dx2 * div( i, j, k ) );
  }

  int getNumGhostCells() const override { return 1; }
};

// 19-point (3D) HOC stencil
class HOCStencil : public StencilBase
{
public:
  amrex::Real computeLaplacian( const amrex::Array4<const amrex::Real>& p,
                                int i,
                                int j,
                                int k,
                                amrex::Real dx2 ) const override
  {
    constexpr amrex::Real one_twentyfourth = 1.0 / 24.0;
    return one_twentyfourth *          //
           ( 2.0 * p( i + 1, j, k ) +  //
             2.0 * p( i - 1, j, k ) +  //
             2.0 * p( i, j + 1, k ) +  //
             2.0 * p( i, j - 1, k ) +  //
             2.0 * p( i, j, k + 1 ) +  //
             2.0 * p( i, j, k - 1 ) +  //
             p( i - 1, j - 1, k ) +    //
             p( i - 1, j + 1, k ) +    //
             p( i - 1, j, k - 1 ) +    //
             p( i - 1, j, k + 1 ) +    //
             p( i, j - 1, k - 1 ) +    //
             p( i, j - 1, k + 1 ) +    //
             p( i, j + 1, k - 1 ) +    //
             p( i, j + 1, k + 1 ) +    //
             p( i + 1, j - 1, k ) +    //
             p( i + 1, j + 1, k ) +    //
             p( i + 1, j, k - 1 ) +    //
             p( i + 1, j, k + 1 ) );
  }

  amrex::Real computeGaussSeidelUpdate(
    const amrex::Array4<const amrex::Real>& p,
    const amrex::Array4<const amrex::Real>& div,
    int i,
    int j,
    int k,
    amrex::Real dx2 ) const override
  {
    constexpr amrex::Real one_twentyfourth = 1.0 / 24.0;
    return one_twentyfourth *          //
           ( 2.0 * p( i + 1, j, k ) +  //
             2.0 * p( i - 1, j, k ) +  //
             2.0 * p( i, j + 1, k ) +  //
             2.0 * p( i, j - 1, k ) +  //
             2.0 * p( i, j, k + 1 ) +  //
             2.0 * p( i, j, k - 1 ) +  //
             p( i - 1, j - 1, k ) +    //
             p( i - 1, j + 1, k ) +    //
             p( i - 1, j, k - 1 ) +    //
             p( i - 1, j, k + 1 ) +    //
             p( i, j - 1, k - 1 ) +    //
             p( i, j - 1, k + 1 ) +    //
             p( i, j + 1, k - 1 ) +    //
             p( i, j + 1, k + 1 ) +    //
             p( i + 1, j - 1, k ) +    //
             p( i + 1, j + 1, k ) +    //
             p( i + 1, j, k - 1 ) +    //
             p( i + 1, j, k + 1 ) -    //
             6.0 * dx2 * div( i, j, k ) );
  }

  int getNumGhostCells() const override { return 2; }
};

// Factory function to create stencil
std::unique_ptr<StencilBase> createStencil( StencilType type );

}  // namespace PressureSolve
