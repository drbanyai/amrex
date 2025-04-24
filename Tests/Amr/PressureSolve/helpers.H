#pragma once

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <array>

/*--------------------------------------------------------------------
  public free function declarations
  --------------------------------------------------------------------*/

// Geometry and mesh setup
amrex::Geometry DefineGeometry(int nx, int ny, int nz, double dx);
amrex::BoxArray DefineBoxArray(int nx, int ny, int nz);
amrex::DistributionMapping DefineDM(const amrex::BoxArray& ba);

// Data structure setup
void DefineFABs(
    amrex::MultiFab& pressure,
    std::array<amrex::MultiFab, 3>& velocity,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm);

// Initialization
void InitializeVelocity(
    std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom,
    int i_face, int j_face, int k_face);

// Main pressure solver function
void SolvePressure(
    amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom,
    amrex::Real tolerance,
    int max_iterations,
    amrex::Real omega,
    int i_face, int j_face, int k_face);

// Verification
void CheckResults(
    const amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom,
    int i_face, int j_face, int k_face);

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
