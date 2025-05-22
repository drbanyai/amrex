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
    const amrex::Geometry& geom);

// Main pressure solver function
void SolvePressure(
    amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom,
    amrex::Real tolerance,
    int max_iterations,
    amrex::Real omega);

// Verification
// Multi-level sampling and CSV output
/**
 * @brief Sample pressure along the center line at the finest resolution, interpolating all levels to the fine grid, and write a CSV with columns for each level.
 * @param pressures Vector of pressure MultiFabs (coarse to fine)
 * @param geoms Vector of Geometries (coarse to fine)
 * @param filename Output CSV filename
 */
void SamplePressureAlongLine(
    const amrex::Vector<amrex::MultiFab>& pressures,
    const amrex::Vector<amrex::Geometry>& geoms,
    const std::string& filename);

void CheckResults(
    const amrex::MultiFab& pressure,
    const std::array<amrex::MultiFab, 3>& velocity,
    const amrex::Geometry& geom);

// Comparison utilities
/**
 * @brief Compare a reference MultiFab against a vector of MultiFabs at different AMR levels
 * @param expected_mf Reference MultiFab to compare against
 * @param level_mfs Vector of MultiFabs at different AMR levels
 * @param expected_geom Geometry for the reference MultiFab
 * @param level_geoms Vector of Geometries for each AMR level
 * @param name Name of the field being compared (for output)
 */
void CompareMultiFabs(
    const amrex::MultiFab& expected_mf,
    const amrex::Vector<amrex::MultiFab>& level_mfs,
    const amrex::Geometry& expected_geom,
    const amrex::Vector<amrex::Geometry>& level_geoms,
    const std::string& name);

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
