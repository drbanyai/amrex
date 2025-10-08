#ifndef AMREX_ML_CELL_ABECLAP_H_
#define AMREX_ML_CELL_ABECLAP_H_
#include <AMReX_Config.H>

#include <AMReX_MLCellABecLap_K.H>
#include <AMReX_MLCellLinOp.H>

namespace amrex {

template <typename MF>
class MLCellABecLapT // NOLINT(cppcoreguidelines-virtual-class-destructor)
    : public MLCellLinOpT<MF> {
public:
  using FAB = typename MF::fab_type;
  using RT = typename MF::value_type;

  using Location = typename MLLinOpT<MF>::Location;

  MLCellABecLapT() = default;
  ~MLCellABecLapT() override = default;

  MLCellABecLapT(const MLCellABecLapT<MF> &) = delete;
  MLCellABecLapT(MLCellABecLapT<MF> &&) = delete;
  MLCellABecLapT<MF> &operator=(const MLCellABecLapT<MF> &) = delete;
  MLCellABecLapT<MF> &operator=(MLCellABecLapT<MF> &&) = delete;

  void define(const Vector<Geometry> &a_geom, const Vector<BoxArray> &a_grids,
              const Vector<DistributionMapping> &a_dmap,
              const LPInfo &a_info = LPInfo(),
              const Vector<FabFactory<FAB> const *> &a_factory = {});

  void define(const Vector<Geometry> &a_geom, const Vector<BoxArray> &a_grids,
              const Vector<DistributionMapping> &a_dmap,
              const Vector<iMultiFab const *> &a_overset_mask,
              const LPInfo &a_info = LPInfo(),
              const Vector<FabFactory<FAB> const *> &a_factory = {});

  void define(const Vector<Geometry> &a_geom, const Vector<BoxArray> &a_grids,
              const Vector<DistributionMapping> &a_dmap,
              const Vector<iMultiFab const *> &a_face_mask_x,
              const Vector<iMultiFab const *> &a_face_mask_y,
              const Vector<iMultiFab const *> &a_face_mask_z,
              const LPInfo &a_info = LPInfo(),
              const Vector<FabFactory<FAB> const *> &a_factory = {});

  [[nodiscard]] iMultiFab const *getOversetMask(int amrlev, int mglev) const {
    return m_overset_mask[amrlev][mglev].get();
  }

  [[nodiscard]] bool needsUpdate() const override {
    return MLCellLinOpT<MF>::needsUpdate();
  }
  void update() override;

  void prepareForSolve() override;

  void setDirichletNodesToZero(int amrlev, int mglev, MF &mf) const override;

  void getFluxes(const Vector<Array<MF *, AMREX_SPACEDIM>> &a_flux,
                 const Vector<MF *> &a_sol, Location a_loc) const final;
  void getFluxes(const Vector<MF *> & /*a_flux*/,
                 const Vector<MF *> & /*a_sol*/) const final {
    amrex::Abort("MLCellABecLap::getFluxes: How did we get here?");
  }

  virtual RT getAScalar() const = 0;
  virtual RT getBScalar() const = 0;
  virtual MF const *getACoeffs(int amrlev, int mglev) const = 0;
  virtual Array<MF const *, AMREX_SPACEDIM> getBCoeffs(int amrlev,
                                                       int mglev) const = 0;

  void applyInhomogNeumannTerm(int amrlev, MF &rhs) const final;

  void addInhomogNeumannFlux(int amrlev,
                             const Array<MF *, AMREX_SPACEDIM> &grad,
                             MF const &sol, bool mult_bcoef) const final;

  void applyOverset(int amrlev, MF &rhs) const override;

#if defined(AMREX_USE_HYPRE) && (AMREX_SPACEDIM > 1)
  [[nodiscard]] std::unique_ptr<Hypre>
  makeHypre(Hypre::Interface hypre_interface) const override;
#endif

#ifdef AMREX_USE_PETSC
  [[nodiscard]] std::unique_ptr<PETScABecLap> makePETSc() const override;
#endif

protected:
  Vector<Vector<std::unique_ptr<iMultiFab>>> m_overset_mask;
  Vector<Vector<std::unique_ptr<iMultiFab>>> m_face_mask_x;
  Vector<Vector<std::unique_ptr<iMultiFab>>> m_face_mask_y;
  Vector<Vector<std::unique_ptr<iMultiFab>>> m_face_mask_z;

  LPInfo m_lpinfo_arg;

  [[nodiscard]] bool supportInhomogNeumannBC() const noexcept override {
    return true;
  }
};

template <typename MF>
void MLCellABecLapT<MF>::define(
    const Vector<Geometry> &a_geom, const Vector<BoxArray> &a_grids,
    const Vector<DistributionMapping> &a_dmap, const LPInfo &a_info,
    const Vector<FabFactory<FAB> const *> &a_factory) {
  MLCellLinOpT<MF>::define(a_geom, a_grids, a_dmap, a_info, a_factory);

  this->m_overset_mask.resize(this->m_num_amr_levels);
  for (int amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
    this->m_overset_mask[amrlev].resize(this->m_num_mg_levels[amrlev]);
  }

  this->m_face_mask_x.resize(this->m_num_amr_levels);
  this->m_face_mask_y.resize(this->m_num_amr_levels);
  this->m_face_mask_z.resize(this->m_num_amr_levels);
  for (int amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
    this->m_face_mask_x[amrlev].resize(this->m_num_mg_levels[amrlev]);
    this->m_face_mask_y[amrlev].resize(this->m_num_mg_levels[amrlev]);
    this->m_face_mask_z[amrlev].resize(this->m_num_mg_levels[amrlev]);
  }
}

template <typename MF>
void MLCellABecLapT<MF>::define(
    const Vector<Geometry> &a_geom, const Vector<BoxArray> &a_grids,
    const Vector<DistributionMapping> &a_dmap,
    const Vector<iMultiFab const *> &a_overset_mask, const LPInfo &a_info,
    const Vector<FabFactory<FAB> const *> &a_factory) {
  BL_PROFILE("MLCellABecLap::define(overset)");

  AMREX_ALWAYS_ASSERT(!this->hasHiddenDimension());

  this->m_lpinfo_arg = a_info;

  auto namrlevs = static_cast<int>(a_geom.size());
  this->m_overset_mask.resize(namrlevs);
  for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
    this->m_overset_mask[amrlev].push_back(
        std::make_unique<iMultiFab>(a_grids[amrlev], a_dmap[amrlev], 1, 1));
    iMultiFab::Copy(*(this->m_overset_mask[amrlev][0]), *a_overset_mask[amrlev],
                    0, 0, 1, 0);
    if (amrlev > 1) {
      AMREX_ALWAYS_ASSERT(amrex::refine(a_geom[amrlev - 1].Domain(), 2) ==
                          a_geom[amrlev].Domain());
    }
  }

  int amrlev = 0;
  Box dom = a_geom[0].Domain();
  for (int mglev = 1; mglev <= a_info.max_coarsening_level; ++mglev) {
    AMREX_ALWAYS_ASSERT(this->mg_coarsen_ratio == 2);
    iMultiFab const &fine = *(this->m_overset_mask[amrlev][mglev - 1]);
    if (dom.coarsenable(2) && fine.boxArray().coarsenable(2)) {
      dom.coarsen(2);
      auto crse = std::make_unique<iMultiFab>(
          amrex::coarsen(fine.boxArray(), 2), fine.DistributionMap(), 1, 1);
      ReduceOps<ReduceOpSum> reduce_op;
      ReduceData<int> reduce_data(reduce_op);
      using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(*crse, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.tilebox();
        Array4<int const> const &fmsk = fine.const_array(mfi);
        Array4<int> const &cmsk = crse->array(mfi);
        reduce_op.eval(bx, reduce_data,
                       [=] AMREX_GPU_HOST_DEVICE(Box const &b) -> ReduceTuple {
                         return {coarsen_overset_mask(b, cmsk, fmsk)};
                       });
      }
      ReduceTuple hv = reduce_data.value(reduce_op);
      if (amrex::get<0>(hv) == 0) {
        this->m_overset_mask[amrlev].push_back(std::move(crse));
      } else {
        break;
      }
    } else {
      break;
    }
  }
  int max_overset_mask_coarsening_level =
      this->m_overset_mask[amrlev].size() - 1;
  ParallelAllReduce::Min(max_overset_mask_coarsening_level,
                         ParallelContext::CommunicatorSub());
  this->m_overset_mask[amrlev].resize(max_overset_mask_coarsening_level + 1);

  LPInfo linfo = a_info;
  linfo.max_coarsening_level =
      std::min(a_info.max_coarsening_level, max_overset_mask_coarsening_level);

  MLCellLinOpT<MF>::define(a_geom, a_grids, a_dmap, linfo, a_factory);

  this->m_face_mask_x.resize(this->m_num_amr_levels);
  this->m_face_mask_y.resize(this->m_num_amr_levels);
  this->m_face_mask_z.resize(this->m_num_amr_levels);
  for (int amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
    this->m_face_mask_x[amrlev].resize(this->m_num_mg_levels[amrlev]);
    this->m_face_mask_y[amrlev].resize(this->m_num_mg_levels[amrlev]);
    this->m_face_mask_z[amrlev].resize(this->m_num_mg_levels[amrlev]);
  }

  amrlev = 0;
  for (int mglev = 1; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
    MF foo(this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 0,
           MFInfo().SetAlloc(false));
    if (!amrex::isMFIterSafe(*(this->m_overset_mask[amrlev][mglev]), foo)) {
      auto osm = std::make_unique<iMultiFab>(this->m_grids[amrlev][mglev],
                                             this->m_dmap[amrlev][mglev], 1, 1);
      osm->ParallelCopy(*(this->m_overset_mask[amrlev][mglev]));
      std::swap(osm, this->m_overset_mask[amrlev][mglev]);
    }
  }

  for (amrlev = 1; amrlev < this->m_num_amr_levels; ++amrlev) {
    for (int mglev = 1; mglev < this->m_num_mg_levels[amrlev];
         ++mglev) { // for ref_ratio 4
      this->m_overset_mask[amrlev].push_back(std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1));

#ifdef AMREX_USE_GPU
      if (Gpu::inLaunchRegion() &&
          this->m_overset_mask[amrlev][mglev]->isFusingCandidate()) {
        auto const &crsema = this->m_overset_mask[amrlev][mglev]->arrays();
        auto const &finema =
            this->m_overset_mask[amrlev][mglev - 1]->const_arrays();
        ParallelFor(
            *(this->m_overset_mask[amrlev][mglev]),
            [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
              coarsen_overset_mask(i, j, k, crsema[box_no], finema[box_no]);
            });
        if (!Gpu::inNoSyncRegion()) {
          Gpu::streamSynchronize();
        }
      } else
#endif
      {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*(this->m_overset_mask[amrlev][mglev]),
                        TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int> const &cmsk =
              this->m_overset_mask[amrlev][mglev]->array(mfi);
          Array4<int const> const fmsk =
              this->m_overset_mask[amrlev][mglev - 1]->const_array(mfi);
          AMREX_HOST_DEVICE_PARALLEL_FOR_3D(
              bx, i, j, k, { coarsen_overset_mask(i, j, k, cmsk, fmsk); });
        }
      }
    }
  }

  for (amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
    for (int mglev = 0; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
      this->m_overset_mask[amrlev][mglev]->setBndry(1);
      this->m_overset_mask[amrlev][mglev]->FillBoundary(
          this->m_geom[amrlev][mglev].periodicity());
    }
  }
}

template <typename MF>
void MLCellABecLapT<MF>::define(
    const Vector<Geometry> &a_geom, const Vector<BoxArray> &a_grids,
    const Vector<DistributionMapping> &a_dmap,
    const Vector<iMultiFab const *> &a_face_mask_x,
    const Vector<iMultiFab const *> &a_face_mask_y,
    const Vector<iMultiFab const *> &a_face_mask_z, const LPInfo &a_info,
    const Vector<FabFactory<FAB> const *> &a_factory) {
  BL_PROFILE("MLCellABecLap::define(overset)");

  AMREX_ALWAYS_ASSERT(!this->hasHiddenDimension());

  this->m_lpinfo_arg = a_info;

  // Initialize the finest multigrid level data for each meshing level.
  auto namrlevs = static_cast<int>(a_geom.size());
  this->m_face_mask_x.resize(namrlevs);
  this->m_face_mask_y.resize(namrlevs);
  this->m_face_mask_z.resize(namrlevs);
  for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
    this->m_face_mask_x[amrlev].push_back(
        std::make_unique<iMultiFab>(a_grids[amrlev], a_dmap[amrlev], 1, 1));
    this->m_face_mask_y[amrlev].push_back(
        std::make_unique<iMultiFab>(a_grids[amrlev], a_dmap[amrlev], 1, 1));
    this->m_face_mask_z[amrlev].push_back(
        std::make_unique<iMultiFab>(a_grids[amrlev], a_dmap[amrlev], 1, 1));
    iMultiFab::Copy(*(this->m_face_mask_x[amrlev][0]), *a_face_mask_x[amrlev],
                    0, 0, 1, 0);
    iMultiFab::Copy(*(this->m_face_mask_y[amrlev][0]), *a_face_mask_y[amrlev],
                    0, 0, 1, 0);
    iMultiFab::Copy(*(this->m_face_mask_z[amrlev][0]), *a_face_mask_z[amrlev],
                    0, 0, 1, 0);
    if (amrlev > 1) {
      AMREX_ALWAYS_ASSERT(amrex::refine(a_geom[amrlev - 1].Domain(), 2) ==
                          a_geom[amrlev].Domain());
    }
  }

  // TODO: This logic is only executed for amrlev 0. But the face mask is set at
  // level 1. Is this a problem?
  int amrlev = 0;
  Box dom = a_geom[0].Domain();
  for (int mglev = 1; mglev <= a_info.max_coarsening_level; ++mglev) {
    AMREX_ALWAYS_ASSERT(this->mg_coarsen_ratio == 2);
    iMultiFab const &fine_x = *(this->m_face_mask_x[amrlev][mglev - 1]);
    iMultiFab const &fine_y = *(this->m_face_mask_y[amrlev][mglev - 1]);
    iMultiFab const &fine_z = *(this->m_face_mask_z[amrlev][mglev - 1]);
    if (dom.coarsenable(2)) {
      bool foundCoarseFineMismatch = false;
      dom.coarsen(2);
      if (fine_x.boxArray().coarsenable(2)) {
        auto crse_x =
            std::make_unique<iMultiFab>(amrex::coarsen(fine_x.boxArray(), 2),
                                        fine_x.DistributionMap(), 1, 1);
        // TODO: We still need to determine how to handle coarse/fine mismatches
        // in coarsen_face_mask(). i.e. Do we have logic to determine if a path
        // exists through non-fixed faces at the finest level, and if so mark
        // the coarse level as not fixed. Or do we just say if any fine cells
        // are fixed, the corresponding coarse face is also fixed?
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*crse_x, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int const> const &fmsk = fine_x.const_array(mfi);
          Array4<int> const &cmsk = crse_x->array(mfi);
          reduce_op.eval(
              bx, reduce_data,
              [=] AMREX_GPU_HOST_DEVICE(Box const &b) -> ReduceTuple {
                return {coarsen_face_mask(b, cmsk, fmsk)};
              });
        }
        ReduceTuple hv = reduce_data.value(reduce_op);
        if (amrex::get<0>(hv) == 0) {
          this->m_face_mask_x[amrlev].push_back(std::move(crse_x));
        } else {
          foundCoarseFineMismatch = true;
        }
      }
      if (fine_y.boxArray().coarsenable(2)) {
        auto crse_y =
            std::make_unique<iMultiFab>(amrex::coarsen(fine_y.boxArray(), 2),
                                        fine_y.DistributionMap(), 1, 1);
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*crse_y, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int const> const &fmsk = fine_y.const_array(mfi);
          Array4<int> const &cmsk = crse_y->array(mfi);
          reduce_op.eval(
              bx, reduce_data,
              [=] AMREX_GPU_HOST_DEVICE(Box const &b) -> ReduceTuple {
                return {coarsen_face_mask(b, cmsk, fmsk)};
              });
        }
        ReduceTuple hv = reduce_data.value(reduce_op);
        if (amrex::get<0>(hv) == 0) {
          this->m_face_mask_y[amrlev].push_back(std::move(crse_y));
        } else {
          foundCoarseFineMismatch = true;
        }
      }
      if (fine_z.boxArray().coarsenable(2)) {
        auto crse_z =
            std::make_unique<iMultiFab>(amrex::coarsen(fine_z.boxArray(), 2),
                                        fine_z.DistributionMap(), 1, 1);
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<int> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*crse_z, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int const> const &fmsk = fine_z.const_array(mfi);
          Array4<int> const &cmsk = crse_z->array(mfi);
          reduce_op.eval(
              bx, reduce_data,
              [=] AMREX_GPU_HOST_DEVICE(Box const &b) -> ReduceTuple {
                return {coarsen_face_mask(b, cmsk, fmsk)};
              });
        }
        ReduceTuple hv = reduce_data.value(reduce_op);
        if (amrex::get<0>(hv) == 0) {
          this->m_face_mask_z[amrlev].push_back(std::move(crse_z));
        } else {
          foundCoarseFineMismatch = true;
        }
      }
      if (foundCoarseFineMismatch) {
        break;
      }
    } else {
      break;
    }
  }

  MLCellLinOpT<MF>::define(a_geom, a_grids, a_dmap, a_info, a_factory);

  // Create an overset mask of nullptrs that correspond to the number of levels
  // so that later pointer checks can be made without indexing out of bounds.
  this->m_overset_mask.resize(this->m_num_amr_levels);
  for (int amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
    this->m_overset_mask[amrlev].resize(this->m_num_mg_levels[amrlev]);
  }

  // Populate the multigrid level face data at the coarsest meshing level.
  amrlev = 0;
  for (int mglev = 1; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
    MF foo_x(this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 0,
             MFInfo().SetAlloc(false));
    // If BoxArrays and DistributionMaps used by the face masks and the linear
    // operator are not the same, make them the same.
    // Note that we assume that if one face mask is inconsistent, they all are.
    if (!amrex::isMFIterSafe(*(this->m_face_mask_x[amrlev][mglev]), foo_x)) {
      // Create iMultiFabs that have a BoxArray and DistributionMap that are
      // consistent with this linear operator.
      auto fm_x = std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1);
      auto fm_y = std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1);
      auto fm_z = std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1);

      // Copy the face mask data into the temporary iMultiFabs.
      fm_x->ParallelCopy(*(this->m_face_mask_x[amrlev][mglev]));
      fm_y->ParallelCopy(*(this->m_face_mask_y[amrlev][mglev]));
      fm_z->ParallelCopy(*(this->m_face_mask_z[amrlev][mglev]));

      // Swap the temporary iMultiFabs with the member face masks.
      std::swap(fm_x, this->m_face_mask_x[amrlev][mglev]);
      std::swap(fm_y, this->m_face_mask_y[amrlev][mglev]);
      std::swap(fm_z, this->m_face_mask_z[amrlev][mglev]);
    }
  }

  // Now for each finer meshing level, populate the corresponding coarser
  // multigrid level face data.
  for (amrlev = 1; amrlev < this->m_num_amr_levels; ++amrlev) {
    for (int mglev = 1; mglev < this->m_num_mg_levels[amrlev];
         ++mglev) { // for ref_ratio 4

      // Create the multigrid level face mask iMultiFabs
      this->m_face_mask_x[amrlev].push_back(std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1));

      this->m_face_mask_y[amrlev].push_back(std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1));

      this->m_face_mask_z[amrlev].push_back(std::make_unique<iMultiFab>(
          this->m_grids[amrlev][mglev], this->m_dmap[amrlev][mglev], 1, 1));

#ifdef AMREX_USE_GPU
      if (Gpu::inLaunchRegion() &&
          this->m_face_mask_x[amrlev][mglev]->isFusingCandidate()) {
        auto const &crsema = this->m_face_mask_x[amrlev][mglev]->arrays();
        auto const &finema =
            this->m_face_mask_x[amrlev][mglev - 1]->const_arrays();
        ParallelFor(
            *(this->m_face_mask_x[amrlev][mglev]),
            [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
              coarsen_face_mask_x(i, j, k, crsema[box_no], finema[box_no]);
            });
        if (!Gpu::inNoSyncRegion()) {
          Gpu::streamSynchronize();
        }
      } else
#endif
      {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        // Generate the coarse multigrid level X face mask data for a given
        // meshing level.
        for (MFIter mfi(*(this->m_face_mask_x[amrlev][mglev]),
                        TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int> const &cmsk =
              this->m_face_mask_x[amrlev][mglev]->array(mfi);
          Array4<int const> const fmsk =
              this->m_face_mask_x[amrlev][mglev - 1]->const_array(mfi);
          AMREX_HOST_DEVICE_PARALLEL_FOR_3D(
              bx, i, j, k, { coarsen_face_mask(i, j, k, cmsk, fmsk); });
        }
      }

#ifdef AMREX_USE_GPU
      if (Gpu::inLaunchRegion() &&
          this->m_face_mask_y[amrlev][mglev]->isFusingCandidate()) {
        auto const &crsema = this->m_face_mask_y[amrlev][mglev]->arrays();
        auto const &finema =
            this->m_face_mask_y[amrlev][mglev - 1]->const_arrays();
        ParallelFor(
            *(this->m_face_mask_y[amrlev][mglev]),
            [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
              coarsen_face_mask_y(i, j, k, crsema[box_no], finema[box_no]);
            });
        if (!Gpu::inNoSyncRegion()) {
          Gpu::streamSynchronize();
        }
      } else
#endif
      {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        // Generate the coarse multigrid level Y face mask data for a given
        // meshing level.
        for (MFIter mfi(*(this->m_face_mask_y[amrlev][mglev]),
                        TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int> const &cmsk =
              this->m_face_mask_y[amrlev][mglev]->array(mfi);
          Array4<int const> const fmsk =
              this->m_face_mask_y[amrlev][mglev - 1]->const_array(mfi);
          AMREX_HOST_DEVICE_PARALLEL_FOR_3D(
              bx, i, j, k, { coarsen_face_mask(i, j, k, cmsk, fmsk); });
        }
      }

#ifdef AMREX_USE_GPU
      if (Gpu::inLaunchRegion() &&
          this->m_face_mask_z[amrlev][mglev]->isFusingCandidate()) {
        auto const &crsema = this->m_face_mask_z[amrlev][mglev]->arrays();
        auto const &finema =
            this->m_face_mask_z[amrlev][mglev - 1]->const_arrays();
        ParallelFor(
            *(this->m_face_mask_z[amrlev][mglev]),
            [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
              coarsen_face_mask_z(i, j, k, crsema[box_no], finema[box_no]);
            });
        if (!Gpu::inNoSyncRegion()) {
          Gpu::streamSynchronize();
        }
      } else
#endif
      {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        // Generate the coarse multigrid level Z face mask data for a given
        // meshing level.
        for (MFIter mfi(*(this->m_face_mask_z[amrlev][mglev]),
                        TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
          const Box &bx = mfi.tilebox();
          Array4<int> const &cmsk =
              this->m_face_mask_z[amrlev][mglev]->array(mfi);
          Array4<int const> const fmsk =
              this->m_face_mask_z[amrlev][mglev - 1]->const_array(mfi);
          AMREX_HOST_DEVICE_PARALLEL_FOR_3D(
              bx, i, j, k, { coarsen_face_mask(i, j, k, cmsk, fmsk); });
        }
      }
    }
  }

  // For all meshing and multigrid levels, do boundary fills of the face mask
  // data.
  for (amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev) {
    for (int mglev = 0; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
      this->m_face_mask_x[amrlev][mglev]->setBndry(1);
      this->m_face_mask_x[amrlev][mglev]->FillBoundary(
          this->m_geom[amrlev][mglev].periodicity());

      this->m_face_mask_y[amrlev][mglev]->setBndry(1);
      this->m_face_mask_y[amrlev][mglev]->FillBoundary(
          this->m_geom[amrlev][mglev].periodicity());

      this->m_face_mask_z[amrlev][mglev]->setBndry(1);
      this->m_face_mask_z[amrlev][mglev]->FillBoundary(
          this->m_geom[amrlev][mglev].periodicity());
    }
  }
}

template <typename MF> void MLCellABecLapT<MF>::update() {
  if (MLCellLinOpT<MF>::needsUpdate()) {
    MLCellLinOpT<MF>::update();
  }
}

template <typename MF> void MLCellABecLapT<MF>::prepareForSolve() {
  MLCellLinOpT<MF>::prepareForSolve();
}

template <typename MF>
void MLCellABecLapT<MF>::setDirichletNodesToZero(int amrlev, int mglev,
                                                 MF &mf) const {
  auto const *omask = this->getOversetMask(amrlev, mglev);
  if (omask) {
    const int ncomp = this->getNComp();
    auto const &mskma = omask->const_arrays();
    auto const &ma = mf.arrays();
    ParallelFor(mf, IntVect(0), ncomp,
                [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k, int n) {
                  if (mskma[bno](i, j, k) == 0) {
                    ma[bno](i, j, k, n) = RT(0.0);
                  }
                });
    if (!Gpu::inNoSyncRegion()) {
      Gpu::streamSynchronize();
    }
  }
}

template <typename MF>
void MLCellABecLapT<MF>::getFluxes(
    const Vector<Array<MF *, AMREX_SPACEDIM>> &a_flux,
    const Vector<MF *> &a_sol, Location a_loc) const {
  BL_PROFILE("MLMG::getFluxes()");

  const int ncomp = this->getNComp();
  const RT betainv = RT(1.0) / getBScalar();
  const int nlevs = this->NAMRLevels();
  for (int alev = 0; alev < nlevs; ++alev) {
    this->compFlux(alev, a_flux[alev], *a_sol[alev], a_loc);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      this->unapplyMetricTerm(alev, 0, *a_flux[alev][idim]);
      if (betainv != RT(1.0)) {
        a_flux[alev][idim]->mult(betainv, 0, ncomp);
      }
    }
    this->addInhomogNeumannFlux(alev, a_flux[alev], *a_sol[alev], true);
  }
}

template <typename MF>
void MLCellABecLapT<MF>::applyInhomogNeumannTerm(int amrlev, MF &rhs) const {
  bool has_inhomog_neumann = this->hasInhomogNeumannBC();
  bool has_robin = this->hasRobinBC();

  if (!has_inhomog_neumann && !has_robin) {
    return;
  }

  int ncomp = this->getNComp();
  const int mglev = 0;

  const auto problo = this->m_geom[amrlev][mglev].ProbLoArray();
  const auto probhi = this->m_geom[amrlev][mglev].ProbHiArray();
  amrex::ignore_unused(probhi);
  const RT dxi = static_cast<RT>(this->m_geom[amrlev][mglev].InvCellSize(0));
  const RT dyi = static_cast<RT>(
      (AMREX_SPACEDIM >= 2) ? this->m_geom[amrlev][mglev].InvCellSize(1)
                            : Real(1.0));
  const RT dzi = static_cast<RT>(
      (AMREX_SPACEDIM == 3) ? this->m_geom[amrlev][mglev].InvCellSize(2)
                            : Real(1.0));
  const RT xlo = static_cast<RT>(problo[0]);
  const RT dx = static_cast<RT>(this->m_geom[amrlev][mglev].CellSize(0));
  const Box &domain = this->m_geom[amrlev][mglev].Domain();

  const RT beta = getBScalar();
  Array<MF const *, AMREX_SPACEDIM> const &bcoef = getBCoeffs(amrlev, mglev);
  FAB foo(Box(IntVect(0), IntVect(1)));
  bool has_bcoef = (bcoef[0] != nullptr);

  const auto &maskvals = this->m_maskvals[amrlev][mglev];
  const auto &bcondloc = *(this->m_bcondloc[amrlev][mglev]);
  const auto &bndry = *(this->m_bndry_sol[amrlev]);

  MFItInfo mfi_info;
  if (Gpu::notInLaunchRegion()) {
    mfi_info.SetDynamic(true);
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(rhs, mfi_info); mfi.isValid(); ++mfi) {
    const Box &vbx = mfi.validbox();
    auto const &rhsfab = rhs.array(mfi);

    const auto &bdlv = bcondloc.bndryLocs(mfi);
    const auto &bdcv = bcondloc.bndryConds(mfi);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      auto const bfab =
          (has_bcoef) ? bcoef[idim]->const_array(mfi) : foo.const_array();
      const Orientation olo(idim, Orientation::low);
      const Orientation ohi(idim, Orientation::high);
      const Box blo = amrex::adjCellLo(vbx, idim);
      const Box bhi = amrex::adjCellHi(vbx, idim);
      const auto &mlo = maskvals[olo].array(mfi);
      const auto &mhi = maskvals[ohi].array(mfi);
      const auto &bvlo = bndry.bndryValues(olo).array(mfi);
      const auto &bvhi = bndry.bndryValues(ohi).array(mfi);
      bool outside_domain_lo = !(domain.contains(blo));
      bool outside_domain_hi = !(domain.contains(bhi));
      if ((!outside_domain_lo) && (!outside_domain_hi)) {
        continue;
      }
      for (int icomp = 0; icomp < ncomp; ++icomp) {
        const BoundCond bctlo = bdcv[icomp][olo];
        const BoundCond bcthi = bdcv[icomp][ohi];
        const RT bcllo = bdlv[icomp][olo];
        const RT bclhi = bdlv[icomp][ohi];
        if (this->m_lobc_orig[icomp][idim] == LinOpBCType::inhomogNeumann &&
            outside_domain_lo) {
          if (idim == 0) {
            RT fac = beta * dxi;
            if (this->m_has_metric_term && !has_bcoef) {
#if (AMREX_SPACEDIM == 1)
              fac *= static_cast<RT>(problo[0] * problo[0]);
#elif (AMREX_SPACEDIM == 2)
              fac *= static_cast<RT>(problo[0]);
#endif
            }
            AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
              mllinop_apply_innu_xlo(i, j, k, rhsfab, mlo, bfab, bctlo, bcllo,
                                     bvlo, fac, has_bcoef, icomp);
            });
          } else if (idim == 1) {
            RT fac = beta * dyi;
            if (this->m_has_metric_term && !has_bcoef) {
              AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
                mllinop_apply_innu_ylo_m(i, j, k, rhsfab, mlo, bctlo, bcllo,
                                         bvlo, fac, xlo, dx, icomp);
              });
            } else {
              AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
                mllinop_apply_innu_ylo(i, j, k, rhsfab, mlo, bfab, bctlo, bcllo,
                                       bvlo, fac, has_bcoef, icomp);
              });
            }
          } else {
            RT fac = beta * dzi;
            AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
              mllinop_apply_innu_zlo(i, j, k, rhsfab, mlo, bfab, bctlo, bcllo,
                                     bvlo, fac, has_bcoef, icomp);
            });
          }
        }
        if (this->m_hibc_orig[icomp][idim] == LinOpBCType::inhomogNeumann &&
            outside_domain_hi) {
          if (idim == 0) {
            RT fac = beta * dxi;
            if (this->m_has_metric_term && !has_bcoef) {
#if (AMREX_SPACEDIM == 1)
              fac *= static_cast<RT>(probhi[0] * probhi[0]);
#elif (AMREX_SPACEDIM == 2)
              fac *= static_cast<RT>(probhi[0]);
#endif
            }
            AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
              mllinop_apply_innu_xhi(i, j, k, rhsfab, mhi, bfab, bcthi, bclhi,
                                     bvhi, fac, has_bcoef, icomp);
            });
          } else if (idim == 1) {
            RT fac = beta * dyi;
            if (this->m_has_metric_term && !has_bcoef) {
              AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
                mllinop_apply_innu_yhi_m(i, j, k, rhsfab, mhi, bcthi, bclhi,
                                         bvhi, fac, xlo, dx, icomp);
              });
            } else {
              AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
                mllinop_apply_innu_yhi(i, j, k, rhsfab, mhi, bfab, bcthi, bclhi,
                                       bvhi, fac, has_bcoef, icomp);
              });
            }
          } else {
            RT fac = beta * dzi;
            AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
              mllinop_apply_innu_zhi(i, j, k, rhsfab, mhi, bfab, bcthi, bclhi,
                                     bvhi, fac, has_bcoef, icomp);
            });
          }
        }

        if (has_robin) {
          // For Robin BC, see comments in AMReX_MLABecLaplacian.cpp above
          // function applyRobinBCTermsCoeffs.
          auto const &rbc =
              (*this->m_robin_bcval[amrlev])[mfi].const_array(icomp * 3);
          if (this->m_lobc_orig[icomp][idim] == LinOpBCType::Robin &&
              outside_domain_lo) {
            if (idim == 0) {
              RT fac = beta * dxi * dxi;
              AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
                RT A = rbc(i, j, k, 2) /
                       (rbc(i, j, k, 1) * dxi + rbc(i, j, k, 0) * RT(0.5));
                rhsfab(i + 1, j, k, icomp) +=
                    fac * bfab(i + 1, j, k, icomp) * A;
              });
            } else if (idim == 1) {
              RT fac = beta * dyi * dyi;
              AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
                RT A = rbc(i, j, k, 2) /
                       (rbc(i, j, k, 1) * dyi + rbc(i, j, k, 0) * RT(0.5));
                rhsfab(i, j + 1, k, icomp) +=
                    fac * bfab(i, j + 1, k, icomp) * A;
              });
            } else {
              RT fac = beta * dzi * dzi;
              AMREX_HOST_DEVICE_FOR_3D(blo, i, j, k, {
                RT A = rbc(i, j, k, 2) /
                       (rbc(i, j, k, 1) * dzi + rbc(i, j, k, 0) * RT(0.5));
                rhsfab(i, j, k + 1, icomp) +=
                    fac * bfab(i, j, k + 1, icomp) * A;
              });
            }
          }
          if (this->m_hibc_orig[icomp][idim] == LinOpBCType::Robin &&
              outside_domain_hi) {
            if (idim == 0) {
              RT fac = beta * dxi * dxi;
              AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
                RT A = rbc(i, j, k, 2) /
                       (rbc(i, j, k, 1) * dxi + rbc(i, j, k, 0) * RT(0.5));
                rhsfab(i - 1, j, k, icomp) += fac * bfab(i, j, k, icomp) * A;
              });
            } else if (idim == 1) {
              RT fac = beta * dyi * dyi;
              AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
                RT A = rbc(i, j, k, 2) /
                       (rbc(i, j, k, 1) * dyi + rbc(i, j, k, 0) * RT(0.5));
                rhsfab(i, j - 1, k, icomp) += fac * bfab(i, j, k, icomp) * A;
              });
            } else {
              RT fac = beta * dzi * dzi;
              AMREX_HOST_DEVICE_FOR_3D(bhi, i, j, k, {
                RT A = rbc(i, j, k, 2) /
                       (rbc(i, j, k, 1) * dzi + rbc(i, j, k, 0) * RT(0.5));
                rhsfab(i, j, k - 1, icomp) += fac * bfab(i, j, k, icomp) * A;
              });
            }
          }
        }
      }
    }
  }
}

template <typename MF>
void MLCellABecLapT<MF>::addInhomogNeumannFlux(
    int amrlev, const Array<MF *, AMREX_SPACEDIM> &grad, MF const &sol,
    bool mult_bcoef) const {
  /*
   * if (mult_bcoef == true)
   *     grad is -bceof*grad phi
   * else
   *     grad is grad phi
   */
  RT fac = mult_bcoef ? RT(-1.0) : RT(1.0);

  bool has_inhomog_neumann = this->hasInhomogNeumannBC();
  bool has_robin = this->hasRobinBC();

  if (!has_inhomog_neumann && !has_robin) {
    return;
  }

  int ncomp = this->getNComp();
  const int mglev = 0;

  const auto dxinv = this->m_geom[amrlev][mglev].InvCellSize();
  const Box domain = this->m_geom[amrlev][mglev].growPeriodicDomain(1);

  Array<MF const *, AMREX_SPACEDIM> bcoef = {
      AMREX_D_DECL(nullptr, nullptr, nullptr)};
  if (mult_bcoef) {
    bcoef = getBCoeffs(amrlev, mglev);
  }

  const auto &bndry = *this->m_bndry_sol[amrlev];

  MFItInfo mfi_info;
  if (Gpu::notInLaunchRegion()) {
    mfi_info.SetDynamic(true);
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(sol, mfi_info); mfi.isValid(); ++mfi) {
    Box const &vbx = mfi.validbox();
    for (OrientationIter orit; orit.isValid(); ++orit) {
      const Orientation ori = orit();
      const int idim = ori.coordDir();
      const Box &ccb = amrex::adjCell(vbx, ori);
      const Dim3 os = IntVect::TheDimensionVector(idim).dim3();
      const RT dxi = static_cast<RT>(dxinv[idim]);
      if (!domain.contains(ccb)) {
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          auto const &phi = sol.const_array(mfi, icomp);
          auto const &bv =
              bndry.bndryValues(ori).multiFab().const_array(mfi, icomp);
          auto const &bc = bcoef[idim] ? bcoef[idim]->const_array(mfi, icomp)
                                       : Array4<RT const>{};
          auto const &f = grad[idim]->array(mfi, icomp);
          if (ori.isLow()) {
            if (this->m_lobc_orig[icomp][idim] == LinOpBCType::inhomogNeumann) {
              AMREX_HOST_DEVICE_FOR_3D(ccb, i, j, k, {
                int ii = i + os.x;
                int jj = j + os.y;
                int kk = k + os.z;
                RT b = bc ? bc(ii, jj, kk) : RT(1.0);
                f(ii, jj, kk) = fac * b * bv(i, j, k);
              });
            } else if (this->m_lobc_orig[icomp][idim] == LinOpBCType::Robin) {
              auto const &rbc =
                  (*this->m_robin_bcval[amrlev])[mfi].const_array(icomp * 3);
              AMREX_HOST_DEVICE_FOR_3D(ccb, i, j, k, {
                int ii = i + os.x;
                int jj = j + os.y;
                int kk = k + os.z;
                RT tmp = RT(1.0) /
                         (rbc(i, j, k, 1) * dxi + rbc(i, j, k, 0) * RT(0.5));
                RT RA = rbc(i, j, k, 2) * tmp;
                RT RB =
                    (rbc(i, j, k, 1) * dxi - rbc(i, j, k, 0) * RT(0.5)) * tmp;
                RT b = bc ? bc(ii, jj, kk) : RT(1.0);
                f(ii, jj, kk) =
                    fac * b * dxi * ((RT(1.0) - RB) * phi(ii, jj, kk) - RA);
              });
            }
          } else {
            if (this->m_hibc_orig[icomp][idim] == LinOpBCType::inhomogNeumann) {
              AMREX_HOST_DEVICE_FOR_3D(ccb, i, j, k, {
                RT b = bc ? bc(i, j, k) : RT(1.0);
                f(i, j, k) = fac * b * bv(i, j, k);
              });
            } else if (this->m_hibc_orig[icomp][idim] == LinOpBCType::Robin) {
              auto const &rbc =
                  (*this->m_robin_bcval[amrlev])[mfi].const_array(icomp * 3);
              AMREX_HOST_DEVICE_FOR_3D(ccb, i, j, k, {
                RT tmp = RT(1.0) /
                         (rbc(i, j, k, 1) * dxi + rbc(i, j, k, 0) * RT(0.5));
                RT RA = rbc(i, j, k, 2) * tmp;
                RT RB =
                    (rbc(i, j, k, 1) * dxi - rbc(i, j, k, 0) * RT(0.5)) * tmp;
                RT b = bc ? bc(i, j, k) : RT(1.0);
                f(i, j, k) =
                    fac * b * dxi *
                    (RA + (RB - RT(1.0)) * phi(i - os.x, j - os.y, k - os.z));
              });
            }
          }
        }
      }
    }
  }
}

template <typename MF>
void MLCellABecLapT<MF>::applyOverset(int amrlev, MF &rhs) const {
  if (m_overset_mask[amrlev][0]) {
    const int ncomp = this->getNComp();
#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion() &&
        m_overset_mask[amrlev][0]->isFusingCandidate()) {
      auto const &osma = m_overset_mask[amrlev][0]->const_arrays();
      auto const &rhsa = rhs.arrays();
      ParallelFor(*m_overset_mask[amrlev][0], IntVect(0), ncomp,
                  [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k,
                                       int n) noexcept {
                    if (osma[box_no](i, j, k) == 0) {
                      rhsa[box_no](i, j, k, n) = RT(0.0);
                    }
                  });
      if (!Gpu::inNoSyncRegion()) {
        Gpu::streamSynchronize();
      }
    } else
#endif
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(*m_overset_mask[amrlev][0], TilingIfNotGPU());
           mfi.isValid(); ++mfi) {
        const Box &bx = mfi.tilebox();
        auto const &rfab = rhs.array(mfi);
        auto const &osm = m_overset_mask[amrlev][0]->const_array(mfi);
        AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, ncomp, i, j, k, n, {
          if (osm(i, j, k) == 0) {
            rfab(i, j, k, n) = RT(0.0);
          }
        });
      }
    }
  }
}

#if defined(AMREX_USE_HYPRE) && (AMREX_SPACEDIM > 1)
template <typename MF>
std::unique_ptr<Hypre>
MLCellABecLapT<MF>::makeHypre(Hypre::Interface hypre_interface) const {
  if constexpr (!std::is_same<MF, MultiFab>()) {
    amrex::Abort("MLCellABecLap Hypre interface only supports MultiFab");
  } else {
    const BoxArray &ba = this->m_grids[0].back();
    const DistributionMapping &dm = this->m_dmap[0].back();
    const Geometry &geom = this->m_geom[0].back();
    const auto &factory = *(this->m_factory[0].back());
    MPI_Comm comm = this->BottomCommunicator();

    const int mglev = this->NMGLevels(0) - 1;

    auto om = getOversetMask(0, mglev);

    auto hypre_solver =
        amrex::makeHypre(ba, dm, geom, comm, hypre_interface, om);

    hypre_solver->setScalars(getAScalar(), getBScalar());

    auto ac = getACoeffs(0, mglev);
    if (ac) {
      hypre_solver->setACoeffs(*ac);
    } else {
      MultiFab alpha(ba, dm, 1, 0, MFInfo(), factory);
      alpha.setVal(0.0);
      hypre_solver->setACoeffs(alpha);
    }

    auto bc = getBCoeffs(0, mglev);
    if (bc[0]) {
      hypre_solver->setBCoeffs(bc);
    } else {
      Array<MultiFab, AMREX_SPACEDIM> beta;
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        beta[idim].define(amrex::convert(ba, IntVect::TheDimensionVector(idim)),
                          dm, 1, 0, MFInfo(), factory);
        beta[idim].setVal(1.0);
      }
      hypre_solver->setBCoeffs(amrex::GetArrOfConstPtrs(beta));
    }
    hypre_solver->setIsMatrixSingular(this->isBottomSingular());

    return hypre_solver;
  }
  return nullptr;
}
#endif

#ifdef AMREX_USE_PETSC
template <typename MF>
std::unique_ptr<PETScABecLap> MLCellABecLapT<MF>::makePETSc() const {
  if constexpr (!std::is_same<MF, MultiFab>()) {
    amrex::Abort("MLCellABecLap PETSc interface only supports MultiFab");
  } else {
    const BoxArray &ba = this->m_grids[0].back();
    const DistributionMapping &dm = this->m_dmap[0].back();
    const Geometry &geom = this->m_geom[0].back();
    const auto &factory = *(this->m_factory[0].back());
    MPI_Comm comm = this->BottomCommunicator();

    auto petsc_solver = makePetsc(ba, dm, geom, comm);

    petsc_solver->setScalars(getAScalar(), getBScalar());

    const int mglev = this->NMGLevels(0) - 1;
    auto ac = getACoeffs(0, mglev);
    if (ac) {
      petsc_solver->setACoeffs(*ac);
    } else {
      MultiFab alpha(ba, dm, 1, 0, MFInfo(), factory);
      alpha.setVal(0.0);
      petsc_solver->setACoeffs(alpha);
    }

    auto bc = getBCoeffs(0, mglev);
    if (bc[0]) {
      petsc_solver->setBCoeffs(bc);
    } else {
      Array<MultiFab, AMREX_SPACEDIM> beta;
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        beta[idim].define(amrex::convert(ba, IntVect::TheDimensionVector(idim)),
                          dm, 1, 0, MFInfo(), factory);
        beta[idim].setVal(1.0);
      }
      petsc_solver->setBCoeffs(amrex::GetArrOfConstPtrs(beta));
    }
    return petsc_solver;
  }
  return nullptr;
}
#endif

extern template class MLCellABecLapT<MultiFab>;

using MLCellABecLap = MLCellABecLapT<MultiFab>;

} // namespace amrex

#endif
