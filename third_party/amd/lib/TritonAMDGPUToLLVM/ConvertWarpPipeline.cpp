/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "Analysis/AMDGPUAllocation.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cstdlib>
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "convert-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTWARPPIPELINE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

// Construct a virtual block describing a pipeline cluster's buffer R/W set.
// Walks recursively so that LDS effects inside nested non-loop regions
// (scf.if / tt.reduce / tt.scan / etc.) are accounted for.  Loops (scf.for /
// scf.while) cannot legally appear inside a cluster, so this walk never has
// to reason about iteration-multiplied effects.
static BlockInfo buildBlockInfoFromBlock(Block *block, Allocation *allocation) {
  BlockInfo info;
  block->walk([&](MemoryEffectOpInterface mei) {
    Operation *op = mei.getOperation();
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effs;
    mei.getEffects(effs);
    for (auto &eff : effs) {
      Value v = eff.getValue();
      if (!v)
        continue;
      for (auto bufId : allocation->getAllBufferIdsWithAliases(v)) {
        if (bufId == Allocation::InvalidBufferId)
          continue;
        auto interval = allocation->getAllocatedInterval(bufId);
        auto slice = AllocationSlice(v, interval, bufId);
        if (isa<MemoryEffects::Write>(eff.getEffect()))
          info.syncWriteSlices[slice].insert(op);
        else if (isa<MemoryEffects::Read>(eff.getEffect()))
          info.syncReadSlices[slice].insert(op);
      }
    }
  });
  return info;
}

// Pre-existing barrier/wait ops that may legally appear at cluster
// boundaries (between stages or before/after a pipeline).  Mirrors
// isPipelineIgnorable in WarpPipeliner.cpp plus the ROCDL-lowered forms that
// can appear after intermediate passes.
static bool isWarpPipelineIgnorableBarrier(Operation *op) {
  return isa<ROCDL::BarrierOp, gpu::BarrierOp, triton::gpu::AsyncWaitOp,
             triton::amdgpu::AsyncWaitOp, triton::amdgpu::AsyncTDMWait,
             triton::amdgpu::AsyncTDMIntrinsicWait>(op);
}

// True if `op` reads or writes memory (a load / store / async-copy).
static bool readsOrWritesMemory(Operation *op) {
  auto mei = dyn_cast<MemoryEffectOpInterface>(op);
  return mei && (mei.hasEffect<MemoryEffects::Read>() ||
                 mei.hasEffect<MemoryEffects::Write>());
}

// True if `exec` is a stage created by the warp-pipeline frontend.
static bool isPipelineStage(scf::ExecuteRegionOp exec) {
  return exec && exec->hasAttr("triton.warp_pipeline.stage");
}

// dyn_cast<scf::ExecuteRegionOp> + warp_pipeline.stage marker check.
// Returns null when `op` is not a pipeline stage.
static scf::ExecuteRegionOp getPipelineStage(Operation *op) {
  auto exec = dyn_cast_or_null<scf::ExecuteRegionOp>(op);
  return isPipelineStage(exec) ? exec : nullptr;
}

// Validate the body of a `pipelined_for` loop.  After WarpPipeliner the body
// must consist of: a sequence of pipeline-stage execute_regions, optional
// pre-existing barrier/wait ops between (or before/after) those stages, and
// a terminator scf.yield -- nothing else.  Emits an error and returns
// failure on any deviation.  Side-effect free: leaves the IR untouched so
// callers can fail fast before mutating anything.
static LogicalResult validatePipelinedForBody(scf::ForOp forOp) {
  std::map<int, Operation *> existingBarrierMap;
  int numClusters = 0;
  for (auto &op : *forOp.getBody()) {
    if (auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (!isPipelineStage(exeOp))
        return op.emitError(
            "non-warp-pipeline scf.execute_region inside pipelined_for body");
      ++numClusters;
    } else if (isWarpPipelineIgnorableBarrier(&op)) {
      if (existingBarrierMap.count(numClusters))
        return op.emitError("multiple pre-existing barriers between pipeline "
                            "stages; insert a dummy stage instead");
      existingBarrierMap[numClusters] = &op;
    } else if (isa<scf::YieldOp>(op)) {
      continue;
    } else {
      return op.emitError("unexpected op inside pipelined_for body; only "
                          "warp-pipeline stages and barrier/wait ops are "
                          "allowed");
    }
  }
  if (numClusters < 2)
    return forOp.emitError(
        "pipelined_for body must contain at least two pipeline stages");
  if (existingBarrierMap.count(0) && existingBarrierMap.count(numClusters))
    return forOp.emitError("pipelined_for body has both top-of-loop and "
                           "bottom-of-loop pre-existing barriers");
  return success();
}

// Pairwise LDS-dependency analysis between pipeline clusters.
//
// `circular` selects the index topology used by the analysis:
//   * true  — the schedule wraps modulo N.  Used by loop pipelines (scf.for)
//             where the wrap-around represents iter-i feeding iter-(i+1).
//   * false — the schedule is straight-line, indices stay in [0, N).  Used
//             by flat (unrolled) pipelines.
// The rest of this comment uses "circular" / "linear" exclusively, since
// the analysis only cares about topology and not about the source IR kind.
//
// LAYOUT
// ------
//   cluster:   c0    c1    c2    ...    c_{N-1}
//   bars:    b0    b1    b2    b3   ...        b_{N-1}        (b_i sits
//                                                              before c_i)
//
//   * circular: b0 is the wrap-around barrier inside the loop body —
//     sitting between c_{N-1} of one iteration and c0 of the next.
//   * linear:   b0 has no physical slot (no barrier exists before the first
//     cluster), and the schedule never wraps around.
//
// GOAL
// ----
//   For every ordered pair (src, dst) whose LDS effects intersect, guarantee
//   that the schedule has at least one LOCAL (ds_wait + s_barrier) barrier
//   somewhere on the path src → dst.  If no existing slot on the path is
//   LOCAL, mark one as LOCAL.
//
// PLACEMENT CHOICE
// ----------------
//   When forced to place a LOCAL barrier we pick:
//     dist == 1 → bars[dst]      (the only slot between src and dst)
//     dist >  1 → bars[dst - 1]  (the second-rightmost slot on the path)
//   The `dst - 1` choice is somewhat arbitrary — any slot in (src, dst] is
//   correct for memory ordering — and is preserved here to match upstream
//   behavior and existing tests.
//
// COVERAGE CHECK
// --------------
//   A pair is "covered" if any slot in (src, barrierLoc] is already LOCAL.
//   Note that bars[dst] is intentionally NOT consulted when dist > 1; this
//   mirrors the placement choice (we never look at, nor place into, the
//   slot owned by the adjacent (dst-1, dst) pair).
//
// ITERATION ORDER
// ---------------
//   We sweep `dist` from 1 up to `maxDist`:
//     * circular: maxDist = N.  dist == N is the self-loop (src == dst),
//       which captures iter-i write vs iter-(i+1) read across the
//       wrap-around when only one cluster touches the buffer.
//     * linear:   maxDist = N - 1.  No wrap.
//   Walking by increasing distance ensures the shorter-range LOCAL
//   barriers we just placed are visible when checking longer-range pairs,
//   skipping many redundant placements.
static void analyzePipelineDependencies(ArrayRef<BlockInfo> clusterInfo,
                                        SmallVectorImpl<bool> &bars,
                                        Allocation *allocation, bool circular) {
  const int N = clusterInfo.size();
  const int maxDist = circular ? N : N - 1;

  // Modular wrap; a no-op in linear mode where indices stay in range.
  auto wrap = [&](int i) -> int { return circular ? (i % N + N) % N : i; };

  // Returns true if any barrier slot in (src, stop] is already LOCAL.
  // The walk starts at `src + 1` and advances one slot at a time, wrapping
  // modulo N in circular mode; it terminates as soon as it finds a LOCAL
  // slot or reaches `stop`.
  auto isCovered = [&](int src, int stop) -> bool {
    for (int i = src + 1;; i++) {
      const int idx = wrap(i);
      if (bars[idx])
        return true;
      if (idx == stop)
        return false;
    }
  };

  for (int dist = 1; dist <= maxDist; dist++) {
    // In linear mode, src + dist must stay in range.  In circular mode all
    // src values are valid and dst wraps modulo N.
    const int srcEnd = circular ? N : N - dist;
    for (int src = 0; src < srcEnd; src++) {
      const int dst = wrap(src + dist);
      const int barrierLoc = (dist == 1) ? dst : wrap(dst - 1);
      if (isCovered(src, barrierLoc))
        continue;
      if (!clusterInfo[src].isIntersected(
              clusterInfo[dst], mlir::triton::AMD::membarFilter, allocation))
        continue;
      bars[barrierLoc] = true;
      LDBG("cluster " << src << " need fence to " << dst
                      << " placing barrier at " << barrierLoc);
    }
  }
}

static void emitClusterBarrier(OpBuilder &r, Location loc, bool needLocal) {
  ROCDL::SchedBarrier::create(r, loc, ROCDL::SchedGroupMask::none);
  if (needLocal)
    mlir::triton::gpu::BarrierOp::create(r, loc, triton::gpu::AddrSpace::Local);
  else
    ROCDL::SBarrierOp::create(r, loc);
  ROCDL::SchedBarrier::create(r, loc, ROCDL::SchedGroupMask::none);
}

static void emitClusterPriority(OpBuilder &r, Location loc,
                                Operation *clusterOp, bool anyHasPriority) {
  if (auto intAttr = clusterOp->getAttrOfType<IntegerAttr>(
          "triton.warp_pipeline.priority")) {
    ROCDL::SetPrioOp::create(r, loc, intAttr.getInt());
  } else if (anyHasPriority) {
    // Reset to default when other stages use priority.
    ROCDL::SetPrioOp::create(r, loc, 0);
  }
}

// PROTOTYPE(dot-coissue / Path A): inside a DOT stage cluster, emit an
// over-provisioned SCHED_GROUP_BARRIER sequence  [MFMA 1][VALU K] x N  at the
// very top of the cluster's region.  The AMDGPU IGroupLP machine-scheduler
// mutation then interleaves each MFMA with ~K co-issuable VALU ops — the
// optimal "MFMA || 6-unpacked" schedule proven in mfma_coissue_scheduling.md.
//
// This pass runs BEFORE tt.dot -> MFMA lowering, so we do not yet know the exact
// MFMA count and over-provision N; groups the solver cannot fill are dropped.
// VALU ops are still unpacked here (packing is the late SIPreEmitPeephole
// peephole), so a VALU group fills with unpacked math — exactly the ops we want
// hidden under the MFMA co-issue window.
//
// Env-gated so the change is inert unless explicitly enabled:
//   TRITON_HIP_DOT_COISSUE=1        enable
//   TRITON_HIP_DOT_COISSUE_VALU=K   VALU ops per MFMA group   (default 6)
//   TRITON_HIP_DOT_COISSUE_NMFMA=N  MFMA groups (over-provision, default 24)
// Uses raw getenv (not triton's whitelisted getBoolEnv) to keep the knob local
// to this prototype.
//
// Strategy = "even interleave": spread the region's co-issuable VALU evenly
// across its MFMAs so the MFMA cadence stays uniform (no end-of-stage leftover
// -> no cool-down / jump-start stall). Per-MFMA group is
//     [MFMA 1] [VALU k_valu] [TRANS k_trans]
//
// MEASURED (1x16320, scalarized): the count-matched "K = round(V/M) per region
// with a separate TRANS group for v_exp" idea does NOT win — a uniform
// OVER-demand (VALU 8 for both DOT1 and DOT2, no TRANS group) is best (~1066):
//   - Lowering DOT2 VALU to its true count (~4) is WORSE (1057) than demanding 8.
//     Over-demanding is what forces the tail to zero; matching the count leaves
//     a tail. So the right knob is "demand more than available", not "match V/M".
//   - Grouping the transcendentals with a TRANS mask is WORSE (VALU8+TRANS2 =
//     1052; +TRANS1 = 1016). v_exp carries softmax dependency chains; pinning it
//     to a rigid per-MFMA cadence fights the scheduler. Leave it to float — the
//     VALU mask already excludes it, so a plain [VALU K] never touches it.
// Hence the defaults: VALU 8 for both stages, TRANS 0. The per-stage / TRANS
// knobs are retained for experimentation only.
// Auto-derive the split parameters from the TTGIR region (no hardcoding).
//   M = #MFMA of the region's tt.dot(s), from instr_shape + warpsPerCTA + tile:
//       numRepM*numRepN*numRepK, no kWidth needed.
//   V = Sum getTotalElemsPerThread over co-issuable VALU elementwise ops, PLUS a
//       tt.reduce term: opsPerCombine * (elemsPerThread(in) - elemsPerThread(out))
//       — the intra-thread combine chain (add/mul reduce -> 1 op/combine;
//       max/min -> the 2-3 ops its combine region expands to).
//   E = Sum getTotalElemsPerThread over transcendental (exp) ops.
struct DotStageCounts {
  int M = 0, V = 0, E = 0;
};
static DotStageCounts countDotStage(Block *clusterBlock) {
  DotStageCounts c;
  auto elems = [](Type t) -> int {
    if (auto rt = dyn_cast<RankedTensorType>(t))
      return (int)triton::gpu::getTotalElemsPerThread(rt);
    return 0;
  };
  clusterBlock->walk([&](Operation *op) {
    if (auto dot = dyn_cast<mlir::triton::DotOp>(op)) {
      auto aTy = dyn_cast<RankedTensorType>(dot.getA().getType());
      auto dTy = dyn_cast<RankedTensorType>(dot.getType());
      if (!aTy || !dTy)
        return;
      auto mfma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(dTy.getEncoding());
      if (!mfma)
        return;
      ArrayRef<unsigned> instr = mfma.getInstrShape();   // [mDim, nDim, kDim]
      ArrayRef<unsigned> warps = mfma.getWarpsPerCTA();  // [warpsM, warpsN]
      auto dShape = dTy.getShape();
      if (instr.size() < 3 || warps.size() < 2 || dShape.size() < 2 ||
          instr[0] == 0 || instr[1] == 0 || instr[2] == 0)
        return;
      int64_t tileM = dShape[dShape.size() - 2];
      int64_t tileN = dShape[dShape.size() - 1];
      int64_t Kdim = aTy.getShape().back(); // contraction dim
      auto cdiv = [](int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; };
      int64_t repM = cdiv(tileM, (int64_t)instr[0] * warps[0]);
      int64_t repN = cdiv(tileN, (int64_t)instr[1] * warps[1]);
      int64_t repK = Kdim / (int64_t)instr[2];
      c.M += (int)(repM * repN * repK);
    } else if (isa<math::Exp2Op, math::ExpOp>(op)) {
      c.E += elems(op->getResult(0).getType());
    } else if (auto red = dyn_cast<mlir::triton::ReduceOp>(op)) {
      // The reduce term (opt-in) OVER-counts: (inE-outE) assumes register
      // combines, but these reductions are mostly cross-lane shuffles (not
      // 4-cyc VALU), so it inflates V and pushes K1 high. Off by default.
      if (!std::getenv("TRITON_HIP_COISSUE_REDUCE"))
        return;
      int opsPerCombine = 0;
      op->getRegion(0).walk([&](Operation *cop) {
        if (isa<arith::AddFOp, arith::MulFOp, arith::SubFOp, arith::MaximumFOp,
                arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
                arith::DivFOp, arith::CmpFOp, arith::SelectOp>(cop))
          opsPerCombine++;
      });
      int inE = op->getNumOperands() > 0 ? elems(op->getOperand(0).getType()) : 0;
      int outE = op->getNumResults() > 0 ? elems(op->getResult(0).getType()) : 0;
      c.V += opsPerCombine * std::max(0, inE - outE);
    } else if (isa<arith::MulFOp, arith::AddFOp, arith::SubFOp, arith::TruncFOp,
                   arith::ExtFOp, arith::MaximumFOp, arith::MaxNumFOp,
                   arith::MinimumFOp, arith::MinNumFOp, arith::DivFOp,
                   math::FmaOp>(op)) {
      // Scalars inside reduce combine regions have no tensor type -> elems()==0,
      // so they are naturally excluded here (counted via the reduce term above).
      c.V += elems(op->getResult(0).getType());
    }
  });
  return c;
}

static void emitDotCoIssueGroups(OpBuilder &b, Location loc, Block *clusterBlock,
                                 StringRef stage) {
  const char *en = std::getenv("TRITON_HIP_DOT_COISSUE");
  if (!en || en[0] == '0' || en[0] == '\0')
    return;
  auto intEnv = [](const char *name, int dflt) -> int {
    if (const char *v = std::getenv(name)) {
      int x = std::atoi(v);
      if (x >= 0)
        return x;
    }
    return dflt;
  };
  // SCHED_GROUP_BARRIER groups are formed by searching UPWARD from the barrier,
  // so the sequence sits at the END of the region. OpBuilder advances past each
  // new op, so sequential creates keep forward order.
  b.setInsertionPoint(clusterBlock->getTerminator());
  const uint32_t syncId = 0;
  auto emitGroup = [&](ROCDL::SchedGroupMask mask, int size) {
    ROCDL::SchedGroupBarrier::create(b, loc, ROCDL::SchedGroupMask::mfma_wmma,
                                     /*size=*/1u, syncId);
    if (size > 0)
      ROCDL::SchedGroupBarrier::create(b, loc, mask, (uint32_t)size, syncId);
  };
  // Auto-derived split (TRITON_HIP_DOT_COISSUE_AUTO=1): compute M,V,E from the
  // TTGIR region and derive K1/K2/g0/g1, instead of the hardcoded reference.
  if (const char *au = std::getenv("TRITON_HIP_DOT_COISSUE_AUTO");
      au && au[0] == '1') {
    DotStageCounts c = countDotStage(clusterBlock);
    if (c.M <= 0)
      return;
    auto ceilDiv = [](int a, int d) { return (a + d - 1) / d; };
    int K1 = std::max(1, ceilDiv(c.V + 2 * c.E, c.M));
    llvm::errs() << "[dot-coissue auto] stage=" << stage << " M=" << c.M
                 << " V=" << c.V << " E=" << c.E << " K1=" << K1;
    if (c.E == 0) {
      llvm::errs() << " -> [VALU " << K1 << "] x" << c.M << "\n";
      for (int j = 0; j < c.M; ++j)
        emitGroup(ROCDL::SchedGroupMask::valu, K1);
    } else {
      int K2 = std::max(1, ceilDiv(K1, 2));
      int g0 = std::min(c.M, std::max(1, (c.V + K1 / 2) / K1)); // round(V/K1)
      int g1 = c.M - g0;
      llvm::errs() << " K2=" << K2 << " g0=" << g0 << " g1=" << g1
                   << " -> [VALU " << K1 << "]x" << g0 << " + [TRANS " << K2
                   << "]x" << g1 << "\n";
      for (int j = 0; j < g0; ++j)
        emitGroup(ROCDL::SchedGroupMask::valu, K1);
      for (int j = 0; j < g1; ++j)
        emitGroup(ROCDL::SchedGroupMask::transcendental, K2);
    }
    return;
  }
  if (stage.starts_with("dot1")) { // QK: no transcendentals -> uniform [MFMA][VALU K]
    const int kValu = intEnv("TRITON_HIP_DOT1_VALU", 8);
    const int nGroups = intEnv("TRITON_HIP_DOT_COISSUE_NMFMA", 24);
    for (int j = 0; j < nGroups; ++j)
      emitGroup(ROCDL::SchedGroupMask::valu, kValu);
  } else {
    // dot2 / PV: SPLIT the MFMAs — some cover VALU, others cover exp, so each
    // MFMA gets a homogeneous co-execute load and the cadence stays uniform in
    // *cycles* (K2 exp @ 8cyc == K1 valu @ 4cyc when K2 = K1/2). group0 MFMAs
    // take [VALU K1]; group1 = M-group0 take [TRANS K2]. Derived (LZ): pretend
    // 1 exp = 2 valu, K1 = round((V + 2E)/M), K2 = K1/2, group0 = round(V/K1).
    //   focus shape (M16, V54, E33): K1=8, K2=4, group0=7, group1=9.
    const int k1 = intEnv("TRITON_HIP_DOT2_VALU", 8);
    const int k2 = intEnv("TRITON_HIP_DOT2_TRANS", 4);
    const int g0 = intEnv("TRITON_HIP_DOT2_G0", 7);  // VALU-covering MFMAs
    const int g1 = intEnv("TRITON_HIP_DOT2_G1", 9);  // exp-covering MFMAs
    const bool expFirst = intEnv("TRITON_HIP_DOT2_EXPFIRST", 0) != 0;
    if (expFirst) {
      for (int j = 0; j < g1; ++j)
        emitGroup(ROCDL::SchedGroupMask::transcendental, k2);
      for (int j = 0; j < g0; ++j)
        emitGroup(ROCDL::SchedGroupMask::valu, k1);
    } else {
      for (int j = 0; j < g0; ++j)
        emitGroup(ROCDL::SchedGroupMask::valu, k1);
      for (int j = 0; j < g1; ++j)
        emitGroup(ROCDL::SchedGroupMask::transcendental, k2);
    }
  }
}

// Wrap a pre-existing barrier op (e.g. async_wait) with sched_barriers so the
// backend scheduler cannot move ops across it, and emit the cluster's
// priority just before the barrier.  Used in place of inserting a fresh
// cluster barrier when one already exists at the cluster boundary.
static void wrapExistingBarrier(OpBuilder &b, Location loc,
                                Operation *clusterOp,
                                Operation *existingBarrier,
                                bool anyHasPriority) {
  b.setInsertionPoint(existingBarrier);
  emitClusterPriority(b, loc, clusterOp, anyHasPriority);
  ROCDL::SchedBarrier::create(b, loc, ROCDL::SchedGroupMask::none);
  b.setInsertionPointAfter(existingBarrier);
  ROCDL::SchedBarrier::create(b, loc, ROCDL::SchedGroupMask::none);
}

// Emit pre-barrier, thread-ID partitioning, and phase-shift cond_barrier.
// Returns warpLow (for reconverge) and warpHigh (consumed by phase shift).
static std::pair<Value, Value>
emitPipelinePrelude(OpBuilder &b, Location loc, int threadsPerPipelineGroup) {
  // Flush any pending shared-memory (LDS) dependencies before entering the
  // warp-pipelined region.  Without this barrier ModuleMembarAnalysis may
  // later insert a barrier inside the first pipeline stage, which would
  // break the carefully tuned pipeline timing.
  mlir::triton::gpu::BarrierOp::create(b, loc, triton::gpu::AddrSpace::Local);

  auto i32ty = b.getIntegerType(32);
  auto workIDX = ROCDL::ThreadIdXOp::create(b, loc, i32ty);
  auto constZero = arith::ConstantIntOp::create(b, loc, 0, 32);
  auto constWarpSize =
      arith::ConstantIntOp::create(b, loc, threadsPerPipelineGroup, 32);
  auto warpIDX = arith::DivSIOp::create(b, loc, workIDX, constWarpSize);
  auto warpLow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                       warpIDX, constZero);
  auto warpHigh = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                        warpIDX, constZero);
  mlir::triton::amdgpu::CondBarrierOp::create(b, loc, warpHigh);

  return {warpLow, warpHigh};
}

// Emit priority reset and reconverge cond_barrier after a pipeline.
static void emitPipelinePostlude(OpBuilder &b, Location loc,
                                 bool anyHasPriority, Value warpLow) {
  if (anyHasPriority)
    ROCDL::SetPrioOp::create(b, loc, 0);
  mlir::triton::amdgpu::CondBarrierOp::create(b, loc, warpLow);
}

class ConvertPipelinedForPattern : public OpRewritePattern<scf::ForOp> {
public:
  ConvertPipelinedForPattern(MLIRContext *ctx, ModuleAllocation &moduleAlloc,
                             int threadsPerPipelineGroup)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2),
        moduleAllocation(moduleAlloc),
        threadsPerPipelineGroup(threadsPerPipelineGroup) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only handle loops that the frontend marked with pipelined_for.
    if (!forOp->getAttr("triton.warp_pipeline.pipelined_for"))
      return rewriter.notifyMatchFailure(forOp, "no pipelined_for");

    // Look up allocation info as in original pass.  Bail out *before* we
    // mutate the IR so a soft match-failure cannot leave a half-converted
    // loop behind (no marker, no barriers).
    auto func = forOp->getParentOfType<mlir::triton::FuncOp>();
    Allocation *allocation = moduleAllocation.getFuncData(func);
    if (!allocation)
      return rewriter.notifyMatchFailure(forOp, "no Allocation for function");

    forOp->removeAttr("triton.warp_pipeline.pipelined_for");
    emitPipelinedFor(rewriter, forOp.getLoc(), forOp, allocation,
                     threadsPerPipelineGroup);
    return success();
  }

private:
  void emitPipelinedFor(PatternRewriter &b, Location loc, scf::ForOp forOp,
                        Allocation *allocation,
                        int threadsPerPipelineGroup) const {
    // 1. Pre-barrier, thread partitioning, and phase shift.
    b.setInsertionPoint(forOp);
    auto [warpLow, warpHigh] =
        emitPipelinePrelude(b, loc, threadsPerPipelineGroup);

    // 2. Walk the (already-validated) body once to collect clusters and any
    // pre-existing inter-cluster barriers (e.g. from prefetch patterns).
    SmallVector<Block *> clusterBlocks;
    SmallVector<Operation *> clusterOps;
    SmallVector<bool> bars;
    std::map<int, Operation *> existingBarrierMap;
    Operation *terminatorOp = nullptr;

    for (auto &op : *forOp.getBody()) {
      if (auto exeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        exeOp.setNoInline(false);
        clusterOps.push_back(&op);
        clusterBlocks.push_back(&exeOp->getRegion(0).front());
        bars.push_back(false);
      } else if (isWarpPipelineIgnorableBarrier(&op)) {
        existingBarrierMap[clusterBlocks.size()] = &op;
      } else if (isa<scf::YieldOp>(op)) {
        terminatorOp = &op;
      }
    }

    SmallVector<BlockInfo> clusterInfo;
    for (auto cb : clusterBlocks)
      clusterInfo.push_back(buildBlockInfoFromBlock(cb, allocation));
    int numClusters = clusterInfo.size();

    // Check if any cluster has explicit priority.
    bool anyHasPriority = llvm::any_of(clusterOps, [](Operation *op) {
      return op->hasAttr("triton.warp_pipeline.priority");
    });
    LDBG("total clusters : " << numClusters);

    // Normally, we don't expect a pipelined loop begins with a barrier
    // but sometimes required by memory prefetching pattern.
    auto topBar = existingBarrierMap.find(0);
    auto bottomBar = existingBarrierMap.find(numClusters);
    bool hasTopBarrier = topBar != existingBarrierMap.end();
    if (bottomBar != existingBarrierMap.end()) {
      // validatePipelinedForBody guarantees we cannot have both top and
      // bottom barriers, so rotating bottom -> 0 is unambiguous.
      assert(!hasTopBarrier &&
             "validatePipelinedForBody should have rejected this");
      existingBarrierMap[0] = bottomBar->second;
      existingBarrierMap.erase(bottomBar);
    }

    // 3. Circular dependency analysis (wrap-around for loop pipelines).
    analyzePipelineDependencies(clusterInfo, bars, allocation,
                                /*circular=*/true);

    // 4. Materializing final cluster-scope barriers.  For each cluster index:
    //  • If there is a pre-existing barrier at that location, we wrap it with
    //    sched_barriers so that backend scheduling cannot move operations
    //    across it.
    //  • If no barrier exists but `bars[i]` is true, we insert a new cluster
    //    barrier (SchedBarrier + Local/SBarrier + SchedBarrier).
    //    The “local” variant is chosen when cluster-to-cluster memory
    //    dependence requires local-scope synchronization.
    //  • Cluster 0 is a special case: if no top-of-loop barrier existed,
    //    the first cluster barrier must be inserted just before the loop’s
    //    terminator, forming the wrap-around dependency.
    // PROTOTYPE(top-barrier): place the loop-carried wrap-around barrier
    // (cluster 0, no pre-existing top barrier) at the TOP of the loop body
    // (before cluster 0) instead of just before the yield. The top barrier's
    // setprio primes cluster 0's priority every iteration, so the separate
    // before-loop priming is dropped. Effect: the loop-control (induction /
    // back-edge) runs after the last stage with no trailing barrier, and the
    // first stage's MFMA leads the region right after the top barrier.
    (void)terminatorOp;
    for (int i = 0; i < numClusters; i++) {
      if (auto exBar = existingBarrierMap.find(i);
          exBar != existingBarrierMap.end()) {
        // FIXME: If bars[i] is true, wrapping a non-LOCAL pre-existing
        // barrier is not enough to satisfy LDS ordering.  For now we rely on
        // the producer to place such barriers only where no local fence is
        // needed.
        wrapExistingBarrier(b, loc, clusterOps[i], exBar->second,
                            anyHasPriority);
      } else {
        // before cluster i (i == 0 -> top of the loop body)
        b.setInsertionPoint(clusterOps[i]);
        emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
        emitClusterBarrier(b, loc, /*needLocal=*/bars[i]);
      }
    }

    // PROTOTYPE(dot-coissue / Path A): inside each DOT stage cluster, inject
    // per-MFMA co-issue scheduling groups so the machine scheduler interleaves
    // MFMAs with unpacked VALU (see emitDotCoIssueGroups). Env-gated; inert by
    // default.
    for (int i = 0; i < numClusters; i++) {
      auto stage = clusterOps[i]->getAttrOfType<StringAttr>(
          "triton.warp_pipeline.stage");
      if (stage && stage.getValue().starts_with("dot"))
        emitDotCoIssueGroups(b, loc, clusterBlocks[i], stage.getValue());
    }

    // 5. Post-loop priority reset and reconverge.
    b.setInsertionPointAfter(forOp);
    emitPipelinePostlude(b, loc, anyHasPriority, warpLow);
  }

  ModuleAllocation &moduleAllocation;
  int threadsPerPipelineGroup;
};

class InlineWarpPipelineExecuteRegionPattern
    : public OpRewritePattern<scf::ExecuteRegionOp> {
public:
  InlineWarpPipelineExecuteRegionPattern(MLIRContext *ctx)
      : OpRewritePattern<scf::ExecuteRegionOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp exec,
                                PatternRewriter &rewriter) const override {
    if (exec->getAttr("no_inline"))
      return rewriter.notifyMatchFailure(exec, "explicit no_inline");

    // Only inline the stages created by the warp-pipeline frontend.
    if (!isPipelineStage(exec))
      return rewriter.notifyMatchFailure(exec, "not a warp-pipeline stage");

    // Make sure this pattern is applied after transforming pipelined forOp
    if (auto forOp = dyn_cast<scf::ForOp>(exec->getParentOp()))
      if (forOp->getAttr("triton.warp_pipeline.pipelined_for"))
        return rewriter.notifyMatchFailure(exec,
                                           "parent forOp not converted yet");

    // Expect a single-block region.
    Region &reg = exec.getRegion();
    if (!llvm::hasSingleElement(reg))
      return rewriter.notifyMatchFailure(exec, "expected single-block region");

    // Inline region.
    Block *block = &reg.front();

    // Emit a scheduling barrier after every memory op in the stage so the
    // backend scheduler keeps them in program order.
    for (Operation &op : llvm::make_early_inc_range(*block))
      if (readsOrWritesMemory(&op)) {
        rewriter.setInsertionPointAfter(&op);
        ROCDL::SchedBarrier::create(
            rewriter, op.getLoc(),
            ROCDL::SchedGroupMask::non_mem_non_sideeffect);
      }

    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, exec, {});
    rewriter.replaceOp(exec, results);
    rewriter.eraseOp(terminator);

    return success();
  }
};

// Process a flat (non-loop) sequence of warp-pipeline execute_regions.
// Unlike the loop case there is no wrap-around: dependencies are strictly
// linear from the first stage to the last.
//
// Emitted IR:
//   ttg.barrier local               (pre-barrier)
//   <thread ID arith>
//   cond_barrier(warpHigh)           (phase shift)
//   [s_setprio P0]
//   execute_region { stage 0 }
//   [s_setprio P1]  sched+barrier    (cluster barrier)
//   execute_region { stage 1 }
//   ...
//   [s_setprio 0]
//   cond_barrier(warpLow)            (reconverge)
//
static void emitPipelinedFlat(SmallVector<scf::ExecuteRegionOp> &clusterOps,
                              Allocation *allocation,
                              int threadsPerPipelineGroup) {
  Location loc = clusterOps.front().getLoc();
  OpBuilder b(clusterOps.front().getContext());
  int numClusters = clusterOps.size();

  // 1. Pre-barrier and phase shift before the first execute_region.
  b.setInsertionPoint(clusterOps.front());
  auto [warpLow, warpHigh] =
      emitPipelinePrelude(b, loc, threadsPerPipelineGroup);

  // 2. Collect cluster info.
  SmallVector<Block *> clusterBlocks;
  SmallVector<bool> bars(numClusters, false);

  for (auto exec : clusterOps) {
    exec.setNoInline(false);
    clusterBlocks.push_back(&exec->getRegion(0).front());
  }

  SmallVector<BlockInfo> clusterInfo;
  for (auto *cb : clusterBlocks)
    clusterInfo.push_back(buildBlockInfoFromBlock(cb, allocation));

  bool anyHasPriority = llvm::any_of(clusterOps, [](scf::ExecuteRegionOp op) {
    return op->hasAttr("triton.warp_pipeline.priority");
  });

  // 3. Linear dependency analysis (no wrap-around for flat pipelines).
  analyzePipelineDependencies(clusterInfo, bars, allocation,
                              /*circular=*/false);

  // 4. Materialize cluster barriers.
  //    Cluster 0 gets only its priority (inserted after cond_barrier above).
  //    Clusters 1..N get priority + cluster barrier, unless a pre-existing
  //    barrier op (e.g., async_wait) already exists between the clusters —
  //    in that case, wrap it with sched_barriers instead of adding a new one.
  emitClusterPriority(b, loc, clusterOps[0], anyHasPriority);

  for (int i = 1; i < numClusters; i++) {
    Operation *existingBarrier = nullptr;
    for (Operation *op = clusterOps[i - 1]->getNextNode();
         op && op != clusterOps[i].getOperation(); op = op->getNextNode()) {
      if (isWarpPipelineIgnorableBarrier(op)) {
        existingBarrier = op;
        break;
      }
    }

    if (existingBarrier) {
      // FIXME: If bars[i] is true, wrapping a non-LOCAL pre-existing barrier
      // is not enough to satisfy LDS ordering.  For now we rely on the
      // producer to place such barriers only where no local fence is needed.
      wrapExistingBarrier(b, loc, clusterOps[i], existingBarrier,
                          anyHasPriority);
    } else {
      b.setInsertionPoint(clusterOps[i]);
      emitClusterPriority(b, loc, clusterOps[i], anyHasPriority);
      emitClusterBarrier(b, loc, /*needLocal=*/bars[i]);
    }
  }

  // 5. Post-sequence reconverge.
  b.setInsertionPointAfter(clusterOps.back());
  emitPipelinePostlude(b, loc, anyHasPriority, warpLow);
}

// Walk the module for flat warp-pipeline execute_region sequences
// (produced by WarpPipeliner::createFlatPipeline) and emit phase-shift
// barriers around them.
static void processUnrolledPipelineRegions(ModuleOp m,
                                           ModuleAllocation &moduleAllocation,
                                           int threadsPerPipelineGroup) {
  m.walk([&](triton::FuncOp funcOp) {
    Allocation *allocation = moduleAllocation.getFuncData(funcOp);
    if (!allocation)
      return;

    // NOTE: We only iterate the function's top-level blocks; flat-pipeline
    // execute_regions inside nested non-loop regions (e.g. scf.if bodies)
    // are not collected.  WarpPipeliner's flat-pipeline frontend has the
    // same scope, so the two stay in sync.
    for (Block &block : funcOp.getBody()) {
      // Collect contiguous sequences of flat warp-pipeline execute_regions,
      // splitting at any non-ignorable, non-pipeline op.
      SmallVector<SmallVector<scf::ExecuteRegionOp>> sequences;
      SmallVector<scf::ExecuteRegionOp> current;

      for (auto &op : block) {
        if (auto exec = getPipelineStage(&op)) {
          current.push_back(exec);
          continue;
        }
        if (isWarpPipelineIgnorableBarrier(&op))
          continue;
        if (!current.empty()) {
          sequences.push_back(std::move(current));
          current.clear();
        }
      }
      if (!current.empty())
        sequences.push_back(std::move(current));

      for (auto &seq : sequences) {
        if (seq.size() < 2)
          continue;
        LDBG("processing flat pipeline with " << seq.size() << " stages");
        emitPipelinedFlat(seq, allocation, threadsPerPipelineGroup);
      }
    }
  });
}

// Return true if `op` is intra-pipeline glue between two clusters — the
// sequence emitted by emitClusterBarrier/emitClusterPriority and any
// pre-existing barrier op that emitPipelinedFlat wraps with sched_barriers.
// Defined in terms of isWarpPipelineIgnorableBarrier so the two sets stay in
// sync; the extra cases below are the ones we emit ourselves.
static bool isIntraPipelineGlue(Operation *op) {
  return isWarpPipelineIgnorableBarrier(op) ||
         isa<ROCDL::SchedBarrier, ROCDL::SetPrioOp, ROCDL::SBarrierOp,
             triton::gpu::BarrierOp>(op);
}

// Walk backward from `exec` past `sched_barrier` / `s_setprio` and check
// whether the first non-glue op is a LOCAL `triton::gpu::BarrierOp`.
// Any other barrier kind (s_barrier, async_wait, …) is treated as
// non-LOCAL for the purposes of LDS-dependency coverage.
static bool hasLocalBarrierBefore(Operation *exec) {
  for (Operation *scan = exec->getPrevNode(); scan;
       scan = scan->getPrevNode()) {
    if (isa<ROCDL::SchedBarrier, ROCDL::SetPrioOp>(scan))
      continue;
    if (auto barrier = dyn_cast<triton::gpu::BarrierOp>(scan))
      return barrier.hasLocal();
    return false;
  }
  return false;
}

// Collect execute_region clusters and their materialized barrier flags from
// a converted pipelined for-loop body.  After ConvertPipelinedForPattern the
// loop body contains: [priority] [barrier] execute_region ... [barrier] yield.
// bars[0] corresponds to the wrap-around barrier (before yield); bars[i] for
// i > 0 is the barrier immediately preceding cluster i.
// Returns false if the loop body doesn't match the expected pattern.
static bool collectLoopClusters(scf::ForOp forOp,
                                SmallVectorImpl<Block *> &blocks,
                                SmallVectorImpl<bool> &bars) {
  Operation *yieldOp = forOp.getBody()->getTerminator();
  if (!yieldOp)
    return false;
  for (auto &op : *forOp.getBody()) {
    if (auto exec = getPipelineStage(&op)) {
      blocks.push_back(&exec->getRegion(0).front());
      bars.push_back(false);
    }
  }
  if (blocks.empty())
    return false;

  int K = blocks.size();
  // bars[0]: wrap-around barrier immediately before the yield.
  // Pattern: [s_setprio] sched_barrier (ttg.barrier_local|s_barrier)
  //          sched_barrier yield
  Operation *op = yieldOp->getPrevNode();
  if (op && isa<ROCDL::SchedBarrier>(op)) {
    op = op->getPrevNode();
    if (auto barrier = dyn_cast_or_null<triton::gpu::BarrierOp>(op))
      bars[0] = barrier.hasLocal();
  }

  // bars[1..K-1]: barrier immediately preceding each cluster's execute_region.
  for (int i = 1; i < K; i++)
    bars[i] = hasLocalBarrierBefore(blocks[i]->getParentOp());
  return true;
}

// Collect execute_region clusters and their preceding barrier flags from a
// flat (unrolled) pipeline starting at `firstExec`.  After emitPipelinedFlat
// the sequence looks like:
//   exec { b_0 } [s_setprio] sched_barrier (barrier) sched_barrier exec { b_1 }
//   ...
// bars[0] is always false (no barrier before the first cluster); bars[i] for
// i > 0 is the barrier between b_{i-1} and b_i.
static bool collectFlatClusters(scf::ExecuteRegionOp firstExec,
                                SmallVectorImpl<Block *> &blocks,
                                SmallVectorImpl<bool> &bars) {
  if (!isPipelineStage(firstExec))
    return false;
  blocks.push_back(&firstExec->getRegion(0).front());
  bars.push_back(false);

  for (Operation *op = firstExec->getNextNode(); op; op = op->getNextNode()) {
    if (auto exec = getPipelineStage(op)) {
      blocks.push_back(&exec->getRegion(0).front());
      bars.push_back(hasLocalBarrierBefore(op));
      continue;
    }
    // Walk past cluster barriers / priority / pre-existing barriers that
    // emitPipelinedFlat may have wrapped with sched_barriers.  Anything
    // else (e.g. cond_barrier postlude, unrelated ops) terminates the
    // flat sequence.
    if (isIntraPipelineGlue(op))
      continue;
    break;
  }
  return true;
}

// Dispatch to collectLoopClusters / collectFlatClusters based on the kind of
// the next pipeline.  The resulting bars follow the same convention as
// collectLoopClusters: bars[0] is either a wrap-around (loop) or false (flat);
// bars[i>0] is the barrier preceding cluster i.
static bool collectNextPipelineClusters(Operation *startOp,
                                        SmallVectorImpl<Block *> &blocks,
                                        SmallVectorImpl<bool> &bars) {
  if (auto forOp = dyn_cast<scf::ForOp>(startOp))
    return collectLoopClusters(forOp, blocks, bars);
  if (auto exec = dyn_cast<scf::ExecuteRegionOp>(startOp))
    return collectFlatClusters(exec, blocks, bars);
  return false;
}

// Check whether merging two pipelines is safe to do without inserting a
// barrier at the boundary.  Enumerates *only* cross-pipeline pairs
// (a_i, b_j) and verifies each intersected pair already has a LOCAL barrier
// on its path in the merged schedule.
//
// Why not reuse analyzePipelineDependencies on the merged sequence?
//   The merged linear analysis would also visit intra-A and intra-B pairs,
//   and may try to flip an internal slot (e.g. between a_0 and a_1) when
//   A's IR has only a non-LOCAL pre-existing barrier (such as
//   amdg.async_tdm_wait).  Such slots are A's own responsibility — A
//   already accepted that wait as sufficient for intra-warp ordering — and
//   re-flipping them is a false positive that prevents elimination on
//   otherwise safe kernels.  Restricting the sweep to cross-pipeline pairs
//   sidesteps the ambiguity entirely.
//
// Cross-warp concurrency vs intra-warp ordering:
//   With a one-stage phase offset the only truly concurrent cross-warp
//   pair at the boundary is (a_{K-1}, b_0).  All other (a_i, b_j) pairs
//   execute sequentially within a single warp.  Both kinds, however,
//   require a LOCAL barrier on AMD: the concurrent pair needs cross-warp
//   sync, and the sequential pair needs ds_wait to order async ds_read /
//   ds_write within a warp (pre-existing async_tdm_wait does *not*
//   guarantee ds completion ordering in general).  So the coverage check
//   uses LOCAL-only mergedBars uniformly across all pairs.
//
// Layout of mergedBars (linear, LOCAL-only):
//   i < K      A's internal LOCAL barriers (loopBars[i]).
//   i == K     boundary seed; set to loopBars[0] because A's wrap-around
//              physically sits at the bottom of A's loop body and, when
//              LOCAL, is the most recent LDS sync the merged schedule
//              inherits as it crosses into B.
//   i > K      B's internal LOCAL barriers (nextBars[i - K]).  nextBars[0]
//              is skipped (flat B has no slot before b_0; loop B's
//              wrap-around lives inside B's body, irrelevant here).
static bool isCrossPipelineSafe(ArrayRef<Block *> loopBlocks,
                                ArrayRef<bool> loopBars,
                                ArrayRef<Block *> nextBlocks,
                                ArrayRef<bool> nextBars,
                                Allocation *allocation) {
  int K = loopBlocks.size();
  int M = nextBlocks.size();
  assert(!loopBars.empty() &&
         "expected at least one cluster in the prior loop");

  SmallVector<BlockInfo> mergedInfo;
  for (auto *b : loopBlocks)
    mergedInfo.push_back(buildBlockInfoFromBlock(b, allocation));
  for (auto *b : nextBlocks)
    mergedInfo.push_back(buildBlockInfoFromBlock(b, allocation));

  SmallVector<bool> mergedBars;
  mergedBars.reserve(K + M);
  for (bool b : loopBars)
    mergedBars.push_back(b);
  mergedBars.push_back(loopBars[0]); // boundary, seeded by A's wrap-around
  for (int i = 1; i < M; i++)
    mergedBars.push_back(nextBars[i]);

  // True if any slot in (src, stop] is LOCAL.  Linear topology, no wrap.
  auto isCovered = [&](int src, int stop) {
    for (int i = src + 1; i <= stop; i++)
      if (mergedBars[i])
        return true;
    return false;
  };

  // Sweep cross-pipeline pairs only.  Placement choice mirrors
  // analyzePipelineDependencies (dist == 1 → dst, dist > 1 → dst - 1).
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < M; j++) {
      int src = i, dst = K + j;
      int dist = dst - src;
      int barrierLoc = (dist == 1) ? dst : dst - 1;
      if (isCovered(src, barrierLoc))
        continue;
      if (!mergedInfo[src].isIntersected(
              mergedInfo[dst], mlir::triton::AMD::membarFilter, allocation))
        continue;
      LDBG("cross-pipeline LDS dep (a_"
           << i << ", b_" << j << ") uncovered at slot " << barrierLoc);
      return false;
    }
  }
  return true;
}

// Eliminate redundant conditional barriers between consecutive warp-pipelined
// regions.  When two pipelines are back-to-back with no intervening
// operations, the post-loop reconverge (cond_barrier warpLow) and the
// pre-pipeline phase shift (cond_barrier warpHigh) cancel out — the phase
// from the first pipeline naturally carries over.
//
// The prelude's ttg.barrier local (see emitPipelinePrelude) exists to flush
// pending LDS state so ModuleMembarAnalysis won't insert barriers inside
// pipeline stages.  When the post-loop cond_barrier is immediately followed
// by this barrier and cross-pipeline dependency analysis confirms no LDS
// hazard at the boundary, the barrier is also redundant.
//
// When the two pipelines merge, the phase offset causes stages from different
// pipelines to execute concurrently (e.g., warp0 runs b0 while warp1 runs
// a_{K-1}).  The cross-pipeline analysis checks all pairs (a_i, b_j) for LDS
// conflicts, accounting for barriers already placed by each pipeline's own
// dependency analysis.
//
// The "next pipeline" can be either another scf.for or a flat (unrolled)
// pipeline represented as a sequence of scf.execute_region ops.
// TODO: This could be generalized to flat-to-loop / flat-to-flat boundaries,
// but those cases cannot reuse a prior loop's wrap-around barrier as the
// boundary seed and are not expected to matter for common codegen.
//
// Before:                              After:
//   scf.for { loop 1 }                  scf.for { loop 1 }
//   [s_setprio 0]                       [s_setprio 0]
//   cond_barrier(warpLow)   ← erase    <thread ID arith> (dead, cleaned later)
//   ttg.barrier local       ← erase    [s_setprio P]
//   <thread ID arith>                   scf.for / execute_region { pipeline 2 }
//   cond_barrier(warpHigh)  ← erase
//   [s_setprio P]
//   scf.for / execute_region { pipeline 2 }
//
static void eliminateRedundantCondBarriers(ModuleOp m,
                                           ModuleAllocation &moduleAllocation) {
  SmallVector<Operation *> toErase;

  m.walk([&](triton::FuncOp funcOp) {
    Allocation *allocation = moduleAllocation.getFuncData(funcOp);
    if (!allocation)
      return;

    for (Block &block : funcOp.getBody()) {
      SmallVector<triton::amdgpu::CondBarrierOp> condBarriers;
      for (auto &op : block)
        if (auto cb = dyn_cast<triton::amdgpu::CondBarrierOp>(&op))
          condBarriers.push_back(cb);

      for (size_t i = 0; i + 1 < condBarriers.size(); i++) {
        auto postLoopCB = condBarriers[i];
        auto preLoopCB = condBarriers[i + 1];

        // The post-loop cond_barrier must be preceded by a scf.for
        // (possibly with an intervening s_setprio reset).
        Operation *prev = postLoopCB->getPrevNode();
        if (prev && isa<ROCDL::SetPrioOp>(prev))
          prev = prev->getPrevNode();
        if (!isa_and_nonnull<scf::ForOp>(prev)) {
          LDBG("post-loop cond_barrier not preceded by scf.for; skipping");
          continue;
        }
        auto prevFor = cast<scf::ForOp>(prev);

        // The pre-loop cond_barrier must be followed by a warp-pipelined
        // scf.for or a flat pipeline execute_region (possibly with an
        // intervening s_setprio).
        Operation *next = preLoopCB->getNextNode();
        if (next && isa<ROCDL::SetPrioOp>(next))
          next = next->getNextNode();
        bool nextIsPipeline =
            isa_and_nonnull<scf::ForOp>(next) || getPipelineStage(next);
        if (!nextIsPipeline) {
          LDBG("pre-loop cond_barrier not followed by a warp-pipeline; "
               "skipping");
          continue;
        }

        // The post-loop cond_barrier must be immediately followed by the
        // prelude's ttg.barrier local — this proves no operations were
        // inserted between the two pipelines.
        auto preBarrier =
            dyn_cast_or_null<triton::gpu::BarrierOp>(postLoopCB->getNextNode());
        if (!preBarrier || !preBarrier.hasLocal()) {
          LDBG("post-loop cond_barrier not immediately followed by prelude "
               "ttg.barrier local; skipping");
          continue;
        }

        // Cross-pipeline LDS dependency analysis.  When the phase carries
        // over, stages from different pipelines execute concurrently at the
        // boundary.  We must verify that no uncovered LDS conflict exists.
        SmallVector<Block *> loopBlocks, nextBlocks;
        SmallVector<bool> loopBars, nextBars;
        if (!collectLoopClusters(prevFor, loopBlocks, loopBars)) {
          LDBG("could not collect prior loop's clusters; skipping");
          continue;
        }
        if (!collectNextPipelineClusters(next, nextBlocks, nextBars)) {
          LDBG("could not collect next pipeline's clusters; skipping");
          continue;
        }
        if (!isCrossPipelineSafe(loopBlocks, loopBars, nextBlocks, nextBars,
                                 allocation)) {
          LDBG("cross-pipeline LDS dependency at boundary — keeping barriers");
          continue;
        }

        LDBG("eliminating redundant barriers between back-to-back pipelines");
        toErase.push_back(postLoopCB);
        toErase.push_back(preBarrier);
        toErase.push_back(preLoopCB);
        i++;
      }
    }
  });

  for (auto *op : llvm::reverse(toErase))
    op->erase();
}

struct ConvertWarpPipeline
    : public mlir::triton::impl::ConvertWarpPipelineBase<ConvertWarpPipeline> {

public:
  ConvertWarpPipeline(StringRef gfxArch)
      : ConvertWarpPipelineBase<ConvertWarpPipeline>() {
    this->gfxArch = gfxArch.str();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    mlir::triton::AMD::TargetInfo targetInfo(gfxArch.getValue());
    size_t partitionSize = targetInfo.getSharedMemoryPartitionSize();
    auto allocationFn = [&targetInfo](Operation *op) -> unsigned {
      return mlir::triton::AMD::AMDAllocationAnalysisScratchSizeFn(op,
                                                                   targetInfo);
    };
    ModuleAllocation moduleAllocation(m, allocationFn, partitionSize);

    if (targetInfo.getISAFamily() == triton::amdgpu::ISAFamily::Unknown) {
      m.emitError("unsupported target: '") << gfxArch.getValue() << "'";
      return signalPassFailure();
    }
    // Thread count of one warp-pipeline group.
    // A block runs on 4 SIMDs with 2 warps per SIMD. Warp-pipelining splits
    // these warps into two groups (one warp per SIMD) that execute different
    // stages at different times.
    int threadsPerPipelineGroup = targetInfo.getWarpSize() * 4;

    // Up-front structural validation: catch malformed pipelined_for bodies
    // before any rewrite mutates the IR.  Errors are emitted at the
    // offending op; we bail out hard rather than producing half-converted
    // IR.
    bool malformed = false;
    m.walk([&](scf::ForOp forOp) {
      if (!forOp->getAttr("triton.warp_pipeline.pipelined_for"))
        return;
      if (failed(validatePipelinedForBody(forOp)))
        malformed = true;
    });
    if (malformed)
      return signalPassFailure();

    RewritePatternSet patternFor(&getContext());
    RewritePatternSet patternInline(&getContext());
    patternFor.add<ConvertPipelinedForPattern>(&getContext(), moduleAllocation,
                                               threadsPerPipelineGroup);
    patternInline.add<InlineWarpPipelineExecuteRegionPattern>(&getContext());

    if (failed(applyPatternsGreedily(m, std::move(patternFor))))
      return signalPassFailure();

    // Flat (unrolled) pipeline regions are still wrapped in execute_regions
    // with no_inline=true from WarpPipeliner.  Process them before inlining.
    processUnrolledPipelineRegions(m, moduleAllocation,
                                   threadsPerPipelineGroup);

    // Must run after patternFor and flat processing (all regions converted,
    // barriers inserted) but before patternInline (inlining execute_regions
    // would flatten the IR and obscure the cond_barrier adjacency we rely on).
    eliminateRedundantCondBarriers(m, moduleAllocation);

    if (failed(applyPatternsGreedily(m, std::move(patternInline))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::AMD {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertWarpPipelinePass(StringRef gfxArch) {
  return std::make_unique<ConvertWarpPipeline>(gfxArch);
}
} // namespace mlir::triton::AMD
