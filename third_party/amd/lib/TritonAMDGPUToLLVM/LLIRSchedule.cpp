//===- LLIRSchedule.cpp - gfx950 LLIR scheduler ---------------------------===//
//
// An LLVM-IR FunctionPass that interleaves MFMA matrix-core instructions with
// memory ops that are independent of them (not the loads that feed those very
// MFMAs), so the matrix unit stays busy across a GEMM's hot loop. It runs before
// register allocation and is opt-in: the Python compiler invokes it only for
// gfx950 and only when the caller passes schedule_hint="gemm-4waves" (see
// HIPBackend.make_llir).
//
// Outline:
//   * analyzeBBMFMA  - split each block into regions; a region boundary is an
//                      MFMA that follows a memory op since the region began.
//                      By construction an MFMA's input loads land in an earlier
//                      region, so intra-region reordering is dependency-safe.
//   * scheduleBB     - for every region, hoist MFMA-input prep, sink MFMA
//                      results, then space the region's MFMAs around its memory
//                      anchors (throughput-based for LR/LW and GR).
//   * run            - schedule every block transactionally: snapshot, schedule,
//                      verifyFunction, and revert just that block if the result
//                      is invalid. runLLIRSchedulePass adds a whole-function
//                      rollback on top, so a bad schedule never reaches codegen.
//
// The pass returns true iff it scheduled at least one region; the caller uses
// that to keep LLVM's machine schedulers disabled only when it took effect, and
// to gate the paired MFMA register-allocation flags.
//
//===----------------------------------------------------------------------===//

#include "TritonAMDGPUToLLVM/MfmaUtility.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "tritonamdgpu-llir-schedule"

namespace {

using namespace llvm;

// Classification of an instruction for scheduling purposes.
enum class SchedKind { MFMA, GR, LR, LW, Other };

// LDS resides in address space 3 on AMDGPU.
constexpr unsigned kLDSAddressSpace = 3;

// Structures used for region analysis/scheduling
struct AnchorInst {
  Instruction *I = nullptr;
  SchedKind Kind = SchedKind::Other;
};

struct MFMARegionInfo {
  Instruction *RegionStart = nullptr;
  unsigned TotalMFMA = 0;
};

using MFMARegionList = SmallVector<MFMARegionInfo, 8>;
using BBMFMAAnalysisMap = DenseMap<const BasicBlock *, MFMARegionList>;

struct BBRegion {
  BasicBlock *BB = nullptr;
  Instruction *Begin = nullptr; // First instruction in region (inclusive)
  Instruction *End =
      nullptr; // First instruction of next region or nullptr (exclusive)
};

struct MFMARegionCollectResult {
  SmallVector<Instruction *, 16> Hoist;
  SmallVector<Instruction *, 16> Sink;
  Instruction *LastAnchor = nullptr;
  SmallVector<AnchorInst, 32> Anchors;
  SmallVector<Instruction *, 32> MFMAInsts;
};

// Utilities grouped for clarity
struct Utils {
  static bool isMFMAorWMMA(const Instruction &I) {
    // Shared matrix-core predicate (also used by the scalarize-packed-fops
    // pass). gfx950 only exposes MFMA, but the helper is family-agnostic.
    return mlir::triton::AMD::isMFMAorWMMA(I);
  }

  static bool isLDSLoadInst(const Instruction *I) {
    auto *LI = dyn_cast<LoadInst>(I);
    return LI && (LI->getPointerAddressSpace() == kLDSAddressSpace);
  }

  static bool isHoistTransparentInst(const Instruction &I) {
    return isa<ShuffleVectorInst>(I) || isa<InsertElementInst>(I);
  }

  static bool isSinkTransparentInst(const Instruction &I) {
    return isa<ExtractElementInst>(I);
  }

  static SchedKind classifySchedInst(Instruction &I) {
    if (isMFMAorWMMA(I))
      return SchedKind::MFMA;

    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->isIntrinsic()) {
          StringRef Name = F->getName();
          // GR: buffer.load (into regs), buffer.load.lds / .async.lds,
          //     raw.ptr.buffer.store (gmem store from regs)
          if (Name.contains("buffer.load") ||
              Name.contains("raw.ptr.buffer.store"))
            return SchedKind::GR;
          // LR: ds_read (ds.read.*) or ds_load (ds.load.*)
          if (Name.contains("ds.read") || Name.contains("ds.load"))
            return SchedKind::LR;
        }
      }
    }

    // LR: load from LDS (addrspace 3)
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->getPointerAddressSpace() == kLDSAddressSpace)
        return SchedKind::LR;
    }

    // LW: store to LDS (addrspace 3)
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->getPointerAddressSpace() == kLDSAddressSpace)
        return SchedKind::LW;
    }

    return SchedKind::Other;
  }

  static iterator_range<BasicBlock::iterator>
  instructionsInRegion(const BBRegion &R) {
    BasicBlock *BB = R.BB;
    // Begin is now inclusive (region starts at this instruction)
    auto ItBegin = R.Begin ? R.Begin->getIterator() : BB->begin();
    auto ItEnd = R.End ? R.End->getIterator() : BB->end();
    return make_range(ItBegin, ItEnd);
  }

  static unsigned getMFMACycles(const Instruction &I) {
    if (!isMFMAorWMMA(I))
      return 0;
    const auto *CI = cast<CallInst>(&I);
    const Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return 0;
    StringRef Name = Callee->getName();

    // Scaled f8f6f4 MFMAs: the cost depends on the operand formats encoded in
    // cbsz (arg 3) and blgp (arg 4).
    if (Name.contains("mfma.scale.f32.16x16x128.f8f6f4")) {
      // both operands f4 -> 16 cycles, otherwise (either operand f8) -> 32.
      if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
        if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
          return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 16
                                                                          : 32;
      return 32; // Fallback if cbsz/blgp are not constants
    }
    if (Name.contains("mfma.scale.f32.32x32x64.f8f6f4")) {
      // both operands f4 -> 32 cycles, otherwise (either operand f8) -> 64.
      if (auto *CbszC = dyn_cast<ConstantInt>(CI->getArgOperand(3)))
        if (auto *BlgpC = dyn_cast<ConstantInt>(CI->getArgOperand(4)))
          return (CbszC->getZExtValue() > 1 && BlgpC->getZExtValue() > 1) ? 32
                                                                          : 64;
      return 64; // Fallback if cbsz/blgp are not constants
    }

    // Fixed-cost MFMAs.
    static constexpr struct {
      StringRef Name;
      unsigned Cycles;
    } kFixedCycles[] = {
        {"mfma.f32.16x16x32.f16", 16},  {"mfma.f32.16x16x32.bf16", 16},
        {"mfma.i32.16x16x64.i8", 16},   {"mfma.f32.32x32x16.f16", 32},
        {"mfma.f32.32x32x16.bf16", 32}, {"mfma.i32.32x32x32.i8", 32},
    };
    for (const auto &Entry : kFixedCycles)
      if (Name.contains(Entry.Name))
        return Entry.Cycles;

    // Unknown / unmodeled shape: the scheduler bails on this region and leaves
    // it to the default LLVM schedulers (such kernels are not perf-critical).
    return 0;
  }

  // Width in bits of the value moved by an LDS-access anchor.
  static unsigned getLDSAccessBits(const Instruction *I) {
    if (const auto *LI = dyn_cast<LoadInst>(I))
      return LI->getType()->getPrimitiveSizeInBits();
    if (const auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand()->getType()->getPrimitiveSizeInBits();
    if (const auto *CI = dyn_cast<CallInst>(I))
      return CI->getType()->getPrimitiveSizeInBits();
    return 0;
  }

  // LDS instruction throughput during steady state, which is proportional to the
  // access bits.
  static unsigned getLDSCoverCycles(const Instruction *I, unsigned MFMACycles) {
    unsigned Bits = getLDSAccessBits(I);
    return Bits ? (Bits / 8) : MFMACycles;
  }

  // MFMAs to emit at this LDS access under a throughput model: reads and writes
  // share the one LDS issue port, so we carry a running cycle balance across the
  // region's accesses and emit floor(balance / MFMACycles) MFMAs here, keeping
  // the remainder for the next access.
  static unsigned takeMFMAsForLDS(const Instruction *I, unsigned MFMACycles,
                                  unsigned &AccumCycles) {
    AccumCycles += getLDSCoverCycles(I, MFMACycles);
    unsigned N = AccumCycles / MFMACycles; // floor; carry the remainder
    AccumCycles -= N * MFMACycles;
    return N;
  }
};

// Region analysis and scheduling logic grouped into a helper class
class LLIRScheduler {
public:
  explicit LLIRScheduler() = default;

  // Roll a block back to a pre-scheduling snapshot: erase the instructions the
  // scheduler inserted (the region-comment void inline-asm calls, which have no
  // uses) and restore the recorded instruction order.
  static void restoreBlock(BasicBlock &BB,
                           const SmallVectorImpl<Instruction *> &snapshot) {
    SmallPtrSet<const Instruction *, 32> orig(snapshot.begin(), snapshot.end());
    SmallVector<Instruction *, 8> inserted;
    for (Instruction &I : BB)
      if (!orig.count(&I))
        inserted.push_back(&I);
    for (Instruction *I : inserted) {
      if (!I->use_empty())
        I->replaceAllUsesWith(PoisonValue::get(I->getType()));
      I->eraseFromParent();
    }
    for (size_t i = 1; i < snapshot.size(); ++i)
      snapshot[i]->moveAfter(snapshot[i - 1]);
  }

  // Schedule every block in the function. Region detection + the per-region
  // structural invariant make this safe: a block with no eligible MFMA region
  // is simply left untouched (so no loop-finding heuristic is needed, and an
  // odd prologue / multiple loops / no loop are all handled uniformly).
  // Each block is scheduled transactionally: if its schedule fails
  // verification (e.g. an epilogue the main logic can't safely interleave),
  // only that block is rolled back, so good blocks keep their schedule.
  // Returns true if any region was scheduled.
  bool run(Function &F) {
    LLVM_DEBUG(dbgs() << "Pre-RA scheduler analyzing function: " << F.getName()
                      << "\n");
    BBMFMAAnalysisMap BBMFMAMap;
    bool scheduled = false;
    for (BasicBlock &BB : F) {
      LLVM_DEBUG(dbgs() << "BB: " << BB.getName() << "\n");
      analyzeBBMFMA(BB, BBMFMAMap);

      // Snapshot the block so we can revert just this block on failure.
      SmallVector<Instruction *, 64> snapshot;
      for (Instruction &I : BB)
        snapshot.push_back(&I);

      if (!scheduleBB(BB, BBMFMAMap))
        continue;

      if (verifyFunction(F, nullptr)) {
        // This block's schedule is invalid; bail gracefully on it alone.
        LLVM_DEBUG(dbgs() << "  reverting unschedulable block "
                          << BB.getName() << "\n");
        restoreBlock(BB, snapshot);
      } else {
        scheduled = true;
      }
    }
    return scheduled;
  }

private:

  // Split a basic block into MFMA regions in a single program-order pass,
  // recording each region's first MFMA (its RegionStart) and its MFMA count.
  // A new region opens at every MFMA that follows a memory op (GR/LR/LW) seen
  // since the region began; by construction an MFMA's input loads land in an
  // earlier region, so intra-region reordering is dependency-safe.
  static void analyzeBBMFMA(BasicBlock &BB, BBMFMAAnalysisMap &Out) {
    MFMARegionList Regions;
    unsigned CurRegion = 0;
    bool SeenMemoryOps = false;
    bool InRegion = false;

    for (Instruction &I : BB) {
      SchedKind SK = Utils::classifySchedInst(I);
      if (SK == SchedKind::GR || SK == SchedKind::LR || SK == SchedKind::LW)
        SeenMemoryOps = true;

      if (!Utils::isMFMAorWMMA(I))
        continue;

      // A memory op since this region's MFMAs began signals that the next MFMA
      // consumes freshly-loaded data and must open a new region.
      if (SeenMemoryOps && InRegion) {
        CurRegion++;
        InRegion = false;
      }
      if (!InRegion) {
        InRegion = true;
        // Memory ops seen *before* this region's first MFMA are the region's own
        // setup (their data feeds these MFMAs), not a boundary marker; clear the
        // flag so they don't spuriously split off the second MFMA on their own.
        SeenMemoryOps = false;
        if (CurRegion >= Regions.size())
          Regions.resize(CurRegion + 1);
        Regions[CurRegion].RegionStart = &I; // first MFMA is the region start
      }
      Regions[CurRegion].TotalMFMA++;
    }

    if (Regions.empty())
      return;

    LLVM_DEBUG({
      for (unsigned i = 0; i < Regions.size(); ++i)
        dbgs() << "Region " << i << ": total MFMA: " << Regions[i].TotalMFMA
               << "\n";
    });

    Out[&BB] = std::move(Regions);
  }

  static bool feedsMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      for (User *U : V->users()) {
        if (auto *UI = dyn_cast<Instruction>(U)) {
          if (Utils::isMFMAorWMMA(*UI))
            return true;
          if (Utils::isHoistTransparentInst(*UI))
            Worklist.push_back(UI);
        }
      }
    }
    return false;
  }

  static bool definedByMFMA(Instruction *I) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;

    Worklist.push_back(I);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      if (auto *DefI = dyn_cast<Instruction>(V)) {
        if (Utils::isMFMAorWMMA(*DefI))
          return true;

        if (Utils::isSinkTransparentInst(*DefI)) {
          for (Value *Op : DefI->operands())
            Worklist.push_back(Op);
        }
      }
    }
    return false;
  }

  static MFMARegionCollectResult
  collectMFMAAndTransparentInstsInRegion(const BBRegion &R) {
    MFMARegionCollectResult Res;

    // All instructions in this region. Hoisting moves a prep to the region start
    // (right after R.Begin); an operand defined later inside the region would
    // then sit after its use, so this set lets us keep such preps in place.
    SmallPtrSet<const Instruction *, 32> RegionInsts;
    for (Instruction &I : Utils::instructionsInRegion(R))
      RegionInsts.insert(&I);

    // Preps cleared for hoisting so far (collected in program order). An operand
    // that is a same-region prep we already hoisted stays ahead of its use.
    SmallPtrSet<const Instruction *, 16> Hoisted;

    for (Instruction &I : Utils::instructionsInRegion(R)) {
      SchedKind K = Utils::classifySchedInst(I);
      if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW) {
        Res.LastAnchor = &I;
        Res.Anchors.push_back({&I, K});
        continue;
      }

      if (K == SchedKind::MFMA) {
        Res.MFMAInsts.push_back(&I);
        continue;
      }

      if (Utils::isHoistTransparentInst(I)) {
        // Hoisting moves I to right after the region start, so it is safe only if
        // every operand still dominates that position: operands defined before
        // the region already do, R.Begin does, and a prep we are also hoisting
        // keeps its relative order ahead of I. An operand defined inside the
        // region that we are NOT hoisting — an LR/LW/GR anchor, or a prep we
        // rejected — would end up after its use, so I must stay put. This covers
        // both shuffle and insertelement, all anchor kinds, and multi-hop chains.
        bool safeToHoist = true;
        for (Value *Op : I.operands()) {
          auto *OpI = dyn_cast<Instruction>(Op);
          if (!OpI || OpI == R.Begin)
            continue;
          if (RegionInsts.count(OpI) && !Hoisted.count(OpI)) {
            safeToHoist = false;
            break;
          }
        }
        if (safeToHoist && feedsMFMA(&I)) {
          Res.Hoist.push_back(&I);
          Hoisted.insert(&I);
        }
        continue;
      }

      if (isa<ExtractElementInst>(I)) {
        if (definedByMFMA(&I))
          Res.Sink.push_back(&I);
      }
    }

    return Res;
  }

  static MFMARegionCollectResult
  preprocessMFMAInstsInRegion(const BBRegion &R) {
    auto Res = collectMFMAAndTransparentInstsInRegion(R);

    if (Res.Hoist.empty() && Res.Sink.empty())
      return Res;

    Instruction *HoistPos = R.Begin; // Region start (the region's first MFMA)
    Instruction *SinkPos = Res.LastAnchor; // last anchor in region

    if (HoistPos)
      for (Instruction *I : llvm::reverse(Res.Hoist)) {
        // Don't hoist R.Begin after itself
        if (I != HoistPos)
          I->moveAfter(HoistPos);
      }

    // Sinking needs a trailing anchor to move past. A region with MFMAs but no
    // GR/LR/LW anchor (LastAnchor stays null) has no valid sink point, so leave
    // the extractelements in place rather than dereferencing a null insert
    // position. Such a region carries no anchors, so scheduleMFMAWithSpacing
    // bails on it anyway.
    // Sink ALL MFMA-result extractelements to just past the region's last
    // anchor. This is required, not just opportunistic: it clears them out of
    // the MFMA run so the subsequent interleaving (which only reorders
    // instructions before LastAnchor) cannot move an MFMA past one of its own
    // result extracts. A region with MFMAs but no anchor has no sink point, so
    // the SinkPos null-guard leaves those extracts in place (scheduleMFMAWith-
    // Spacing bails on such a region anyway). Any genuinely unsafe sink is
    // caught by the per-block verifyFunction rollback.
    if (SinkPos)
      for (Instruction *I : llvm::reverse(Res.Sink))
        I->moveAfter(SinkPos);

    return Res;
  }

  static StringRef schedKindName(SchedKind K) {
    switch (K) {
    case SchedKind::GR:
      return "GR";
    case SchedKind::LR:
      return "LR";
    case SchedKind::LW:
      return "LW";
    case SchedKind::MFMA:
      return "mfma";
    case SchedKind::Other:
      return "other";
    }
    llvm_unreachable("unknown SchedKind");
  }

  // Helper: move N MFMAs after InsertPt using moveAfter.
  // moveAfter naturally produces correct order: each new MFMA goes right
  // after InsertPt, pushing previous ones further away.
  // Result: InsertPt, MFMA[N-K], ..., MFMA[N-2], MFMA[N-1]
  static unsigned moveMFMAsAfter(SmallVectorImpl<Instruction *> &MFMAInsts,
                                 unsigned &MFMAIdx, unsigned Count,
                                 Instruction *InsertPt) {
    unsigned moved = 0;
    for (unsigned j = 0; j < Count && MFMAIdx > 0; ++j) {
      MFMAInsts[--MFMAIdx]->moveAfter(InsertPt);
      moved++;
    }
    return moved;
  }

  // Interleave MFMA with anchor instructions using moveAfter.
  //
  // Pure throughput model — every count is "how many MFMAs of compute cover this
  // memory op's issue-port occupancy needs", never a latency-hiding reorder:
  //   - GR (global load): ceil(64 / mfma_cycles) MFMAs (1 if immediately
  //                       followed by an LR).
  //   - LR / LW (LDS read/write): floor(carried LDS-cycle balance / mfma_cycles)
  //                       MFMAs. Reads and writes share the one LDS issue port,
  //                       so both use the same width-proportional pairing.
  //   - 2 MFMAs drain the tail; any leftover compute (more MFMAs than the memory
  //     ops demand cover for) is split evenly between the region's head and tail,
  //     with an odd MFMA favoring the head.
  static void scheduleMFMAWithSpacing(SmallVectorImpl<AnchorInst> &Anchors,
                                      SmallVectorImpl<Instruction *> &MFMAInsts) {
    if (Anchors.empty())
      return;
    if (MFMAInsts.empty())
      return;

    unsigned mfmaCycles = Utils::getMFMACycles(*MFMAInsts.front());
    if (mfmaCycles == 0)
      return;
    // A global load occupies the global-load path for ~64 cycles, so it needs
    // 64 cycles of MFMA cover — ceil(64 / mfma_cycles) MFMAs (4 for a 16-cycle
    // MFMA, 2 for 32-cycle). Same throughput basis as the LDS-access pairing.
    unsigned mfmaPerGR = (64 + mfmaCycles - 1) / mfmaCycles;

    unsigned MFMAIdx = MFMAInsts.size();
    unsigned Total = MFMAIdx;

    // Count anchors by kind. ldsBudget is the total MFMA cover the region's LDS
    // accesses demand, floor(sum(cycles_per_access) / mfma_cycles) — the
    // throughput ratio over reads *and* writes alike, so cheap accesses share an
    // MFMA and wide ones draw several.
    unsigned numGR = 0, numGRBeforeLR = 0, totalLDSCycles = 0;
    for (size_t j = 0; j < Anchors.size(); ++j) {
      if (Anchors[j].Kind == SchedKind::GR) {
        numGR++;
        if (j + 1 < Anchors.size() && Anchors[j + 1].Kind == SchedKind::LR)
          numGRBeforeLR++;
      } else if (Anchors[j].Kind == SchedKind::LR ||
                 Anchors[j].Kind == SchedKind::LW) {
        totalLDSCycles += Utils::getLDSCoverCycles(Anchors[j].I, mfmaCycles);
      }
    }
    unsigned ldsBudget = totalLDSCycles / mfmaCycles;

    // gfx950 scheduling:
    //   GR: mfmaPerGR MFMAs each (except GR→LR gets 1)
    //   LR/LW: cycle-paired at the true throughput ratio (takeMFMAsForLDS) —
    //          cheap accesses share an MFMA, wide ones draw several
    //   2 MFMAs drain the end; leftover compute is split evenly head/tail
    unsigned grBudget = mfmaPerGR * (numGR - numGRBeforeLR);
    unsigned needed = grBudget + numGRBeforeLR + ldsBudget + 2;
    unsigned leftover = (Total > needed) ? Total - needed : 0;

    // Surplus compute (more MFMAs than the memory ops need cover for) is split
    // evenly between the region's head and tail; an odd MFMA favors the head.
    unsigned tailLeftover = leftover / 2;               // floor → tail
    unsigned headLeftover = leftover - tailLeftover;    // ceil → head

    LLVM_DEBUG(dbgs() << "  MFMA budget: total=" << Total << ", needed=" << needed
                      << ", leftover=" << leftover << " (head=" << headLeftover
                      << ", tail=" << tailLeftover << ")\n");

    // The 2-MFMA tail drain plus the tail's share of the surplus. The head's
    // share is whatever stays unmoved at the front after the reverse walk.
    unsigned MFMAAtEnd =
        moveMFMAsAfter(MFMAInsts, MFMAIdx, 2 + tailLeftover, Anchors.back().I);
    // Running LDS-cycle balance carried across the region's LDS accesses (LR and
    // LW, processed in reverse) so the MFMA:access pairing follows the true
    // throughput ratio.
    unsigned ldsAccum = 0;
    DenseMap<SchedKind, unsigned> MFMAPerAnchorKind;

    for (int i = static_cast<int>(Anchors.size()) - 1; i >= 0 && MFMAIdx > 0;
         --i) {
      size_t idx = static_cast<size_t>(i);
      Instruction *InsertPt = Anchors[idx].I;
      SchedKind Kind = Anchors[idx].Kind;

      unsigned Count = 0;
      if (Kind == SchedKind::LR || Kind == SchedKind::LW) {
        // Cycle model: emit floor(balance / mfma_cycles) MFMAs, carrying the
        // remainder — cheap accesses share an MFMA, wide ones draw several.
        // Reads and writes use the same shared LDS-cycle balance.
        Count = Utils::takeMFMAsForLDS(Anchors[idx].I, mfmaCycles, ldsAccum);
      } else if (Kind == SchedKind::GR) {
        bool followedByLR = (idx + 1 < Anchors.size() &&
                             Anchors[idx + 1].Kind == SchedKind::LR);
        Count = followedByLR ? 1 : mfmaPerGR;
      }

      [[maybe_unused]] unsigned moved =
          moveMFMAsAfter(MFMAInsts, MFMAIdx, Count, InsertPt);
      LLVM_DEBUG(MFMAPerAnchorKind[Kind] += moved);
    }

    LLVM_DEBUG({
      dbgs() << "  MFMA insertion summary: total=" << Total
             << ", at_front=" << MFMAIdx << ", at_end=" << MFMAAtEnd;
      for (auto &KV : MFMAPerAnchorKind) {
        dbgs() << ", after_" << schedKindName(KV.first) << "=" << KV.second;
      }
      dbgs() << "\n";
    });
  }

  // Insert an inline asm comment before the given instruction.
  // Emit a non-side-effecting inline-asm comment (a pure annotation, NOT a
  // reorder barrier): the scheduler already disables the machine schedulers, so
  // the region markers shouldn't artificially constrain instruction movement.
  static void insertAsmComment(Instruction *IP, const std::string &Comment) {
    LLVMContext &Ctx = IP->getContext();
    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(IP);
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    InlineAsm *IA =
        InlineAsm::get(FTy, ";; " + Comment, "", /*hasSideEffects=*/false);
    Builder.CreateCall(IA);
  }

  static bool scheduleBB(BasicBlock &BB, const BBMFMAAnalysisMap &Analysis) {
    auto It = Analysis.find(&BB);
    if (It == Analysis.end())
      return false;

    const MFMARegionList &Regions = It->second;

    unsigned NumRegions = Regions.size();
    unsigned ScheduledRegionIdx = 0;

    for (unsigned i = 0; i < NumRegions; ++i) {
      const MFMARegionInfo &R = Regions[i];
      if (!R.RegionStart)
        continue;

      if (R.TotalMFMA != 0) {
        BBRegion bbR;
        bbR.BB = &BB;
        bbR.Begin = Regions[i].RegionStart;
        bbR.End = (i + 1 < NumRegions) ? Regions[i + 1].RegionStart : nullptr;

        // Schedulability check, performed BEFORE any mutation: bail on a region
        // whose MFMA shape we don't model (getMFMACycles == 0) or that has no
        // memory anchor to interleave the MFMAs against. Skipping here -- before
        // preprocessMFMAInstsInRegion moves anything -- ensures the pass only
        // reports success (and only trips the misched-disable / AGPR-form flags)
        // for regions it actually schedules; unmodeled kernels are left to the
        // default LLVM schedulers.
        unsigned MFMACycles = 0;
        bool SeenMFMA = false, HasAnchor = false;
        for (Instruction &I : Utils::instructionsInRegion(bbR)) {
          SchedKind K = Utils::classifySchedInst(I);
          if (K == SchedKind::GR || K == SchedKind::LR || K == SchedKind::LW)
            HasAnchor = true;
          else if (K == SchedKind::MFMA && !SeenMFMA) {
            SeenMFMA = true;
            MFMACycles = Utils::getMFMACycles(I);
          }
        }
        if (MFMACycles == 0 || !HasAnchor)
          continue;

        MFMARegionCollectResult Res = preprocessMFMAInstsInRegion(bbR);

        // --- Build region comment ---
        std::string Comment;
        raw_string_ostream OS(Comment);

        // Count anchors by kind
        unsigned numGR = 0, numLR = 0, numLW = 0;
        for (auto &A : Res.Anchors) {
          if (A.Kind == SchedKind::GR)
            numGR++;
          else if (A.Kind == SchedKind::LR)
            numLR++;
          else if (A.Kind == SchedKind::LW)
            numLW++;
        }
        OS << "Region " << ScheduledRegionIdx << ": " << Res.MFMAInsts.size()
           << " mfma, " << numGR << " GR, " << numLR << " LR, " << numLW << " LW";
        ScheduledRegionIdx++;

        insertAsmComment(bbR.Begin, Comment);

        LLVM_DEBUG({
          dbgs() << "Cluster " << i << " structure:";
          SchedKind RunKind = SchedKind::Other;
          unsigned RunCount = 0;
          for (Instruction &Inst : Utils::instructionsInRegion(bbR)) {
            SchedKind K = Utils::classifySchedInst(Inst);
            if (K != SchedKind::MFMA && K != SchedKind::GR &&
                K != SchedKind::LR && K != SchedKind::LW)
              continue;
            if (K == RunKind) {
              RunCount++;
            } else {
              if (RunCount > 0)
                dbgs() << " " << RunCount << " " << schedKindName(RunKind);
              RunKind = K;
              RunCount = 1;
            }
          }
          if (RunCount > 0)
            dbgs() << " " << RunCount << " " << schedKindName(RunKind);
          dbgs() << "\n";
        });

        scheduleMFMAWithSpacing(Res.Anchors, Res.MFMAInsts);
      }
    }
    return ScheduledRegionIdx > 0;
  }

};

// Pass wrapper

struct LLIRSchedulePass : FunctionPass {
  static char ID;
  LLIRScheduler Scheduler;
  // Set to true when the pass actually scheduled at least one region. The caller
  // uses it to decide whether LLVM's machine schedulers should be disabled; if
  // nothing was scheduled, they must stay enabled.
  bool *DidSchedule = nullptr;

  LLIRSchedulePass(bool *DidSchedule = nullptr)
      : FunctionPass(ID), DidSchedule(DidSchedule) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // The pass only reorders instructions within blocks and inserts intra-block
    // instructions; it never changes the CFG. It needs no analyses.
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override {
    if (F.isDeclaration())
      return false;
    bool scheduled = Scheduler.run(F);
    if (DidSchedule)
      *DidSchedule = scheduled;
    return scheduled;
  }
};

} // end anonymous namespace

char LLIRSchedulePass::ID = 0;

namespace mlir::triton::AMD {

bool runLLIRSchedulePass(llvm::Function &F) {
  // Snapshot the unscheduled function so we can roll back if the scheduled IR
  // fails verification. The scheduler reorders instructions heuristically; a bad
  // schedule must never reach codegen. Rolling back to the pristine body (rather
  // than asserting, which is compiled out in release builds) keeps the kernel
  // correct by falling back to the default, machine-scheduled lowering.
  llvm::ValueToValueMapTy vmap;
  llvm::Function *backup = llvm::CloneFunction(&F, vmap);
  backup->setName(F.getName() + ".llirsched.bak");

  bool didSchedule = false;
  {
    llvm::legacy::FunctionPassManager FPM(F.getParent());
    FPM.add(new LLIRSchedulePass(&didSchedule));
    FPM.doInitialization();
    FPM.run(F);
    FPM.doFinalization();
  }

  if (llvm::verifyFunction(F, &llvm::errs())) {
    llvm::errs() << "LLIR schedule pass produced invalid IR; rolling back to the "
                    "unscheduled function (machine scheduling stays enabled).\n";
    // Replace F's invalid body with the pristine backup body: drop F's blocks,
    // splice in the backup's, then remap the backup's args onto F's.
    // deleteBody() forces external linkage, so save and restore F's linkage.
    auto origLinkage = F.getLinkage();
    F.deleteBody();
    F.splice(F.end(), backup);
    F.setLinkage(origLinkage);
    for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
      backup->getArg(i)->replaceAllUsesWith(F.getArg(i));
    backup->eraseFromParent();
    return false; // bailed out: caller leaves the machine schedulers enabled
  }

  backup->eraseFromParent();
  return didSchedule;
}

} // namespace mlir::triton::AMD
