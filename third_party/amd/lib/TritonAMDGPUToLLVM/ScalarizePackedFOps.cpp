#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"

#define DEBUG_TYPE "tritonamdgpu-scalarize-packed-fops"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

bool isMFMAorWMMA(Instruction &inst) {
  auto *callInst = llvm::dyn_cast<CallInst>(&inst);
  if (!callInst)
    return false;
  // E.g., tail call void asm sideeffect "s_waitcnt lgkmcnt(0) ", ""()
  if (callInst->isInlineAsm())
    return false;
  Function *calledFunc = callInst->getCalledFunction();
  if (!calledFunc->isIntrinsic())
    return false;
  StringRef intrinName = calledFunc->getName();
  if (intrinName.contains("mfma") || intrinName.contains("wmma"))
    return true;
  return false;
}

// llvm.fmuladd / llvm.fma on a vector type. These are CALLS, not BinaryOperators,
// so the m_BinOp path below never sees them and they survive as v_pk_fma_f32 --
// which costs 8 cycles and, being packed, cannot co-issue with an MFMA at all.
// Measured on the gfx950 FAv4 kernel: 3 x v_pk_fma left in the PV stage = 12
// cycles of un-hideable VALU plus a distorted co-exec group, the last thing
// keeping that stage off its MFMA-bound floor.
static Intrinsic::ID getScalarizableFPIntrinsic(Instruction *inst) {
  auto *callInst = dyn_cast<CallInst>(inst);
  if (!callInst || callInst->isInlineAsm())
    return Intrinsic::not_intrinsic;
  Function *callee = callInst->getCalledFunction();
  if (!callee || !callee->isIntrinsic())
    return Intrinsic::not_intrinsic;
  Intrinsic::ID id = callee->getIntrinsicID();
  if (id == Intrinsic::fmuladd || id == Intrinsic::fma)
    return id;
  return Intrinsic::not_intrinsic;
}

bool maybeReplaceVectorFMAWithScalarFMAs(Instruction *inst,
                                         IRBuilder<> &builder) {
  Intrinsic::ID id = getScalarizableFPIntrinsic(inst);
  if (id == Intrinsic::not_intrinsic)
    return false;
  auto *callInst = cast<CallInst>(inst);
  auto *vecTy = dyn_cast<VectorType>(callInst->getType());
  if (!vecTy)
    return false;
  assert(!vecTy->isScalableTy() && "expected fixed-len vector");
  Type *elemTy = vecTy->getElementType();
  builder.SetInsertPoint(inst);
  Value *newVec = llvm::UndefValue::get(vecTy);
  for (int i = 0, e = vecTy->getElementCount().getFixedValue(); i < e; ++i) {
    Value *a = builder.CreateExtractElement(callInst->getArgOperand(0), i);
    Value *b = builder.CreateExtractElement(callInst->getArgOperand(1), i);
    Value *c = builder.CreateExtractElement(callInst->getArgOperand(2), i);
    Value *res = builder.CreateIntrinsic(elemTy, id, {a, b, c});
    // Keep fast-math flags so the scalar ops still contract/reassociate exactly
    // as the packed op was allowed to.
    if (auto *newCall = dyn_cast<CallInst>(res))
      newCall->copyFastMathFlags(callInst);
    newVec = builder.CreateInsertElement(newVec, res, i);
  }
  inst->replaceAllUsesWith(newVec);
  return true;
}

bool maybeReplaceVectorFOpWithScalarFOps(Instruction *inst,
                                         IRBuilder<> &builder) {
  Value *lhs, *rhs;
  if (!match(inst, m_BinOp(m_Value(lhs), m_Value(rhs))))
    return false;
  auto *VecLhs = dyn_cast<VectorType>(lhs->getType());
  if (!VecLhs)
    return false;
  assert(!VecLhs->isScalableTy() && "expected fixed-len vector");
  builder.SetInsertPoint(inst);
  Value *newVec = llvm::UndefValue::get(VecLhs);
  for (int i = 0; i < VecLhs->getElementCount().getFixedValue(); ++i) {
    Value *newLhs = builder.CreateExtractElement(lhs, i);
    Value *newRhs = builder.CreateExtractElement(rhs, i);
    Value *res;
    if (inst->getOpcode() == Instruction::FMul)
      res = builder.CreateFMul(newLhs, newRhs);
    else if (inst->getOpcode() == Instruction::FAdd)
      res = builder.CreateFAdd(newLhs, newRhs);
    else if (inst->getOpcode() == Instruction::FSub)
      res = builder.CreateFSub(newLhs, newRhs);
    else
      llvm::report_fatal_error("only fadd, fmul, fsub supported");
    newVec = builder.CreateInsertElement(newVec, res, i);
  }
  LLVM_DEBUG(dbgs() << "ScalarizePackedFOps: Replacing: " << inst << '\n');
  LLVM_DEBUG(dbgs() << "                     With: " << newVec << '\n');
  inst->replaceAllUsesWith(newVec);
  return true;
}

//  This Pass scalarizes vector `fmul`s and `fadd`s in basic blocks that contain
//  MFMAs. The point/purpose/value of doing is that these get codegened to
//  "packed" ops (`v_pk_mul_f32`/`v_pk_add_f32`) and while packed ops use
//  separate VALUs from MFMA tensor cores (no problem there), the instructions
//  themselves cannot be *issued* in parallel, thus there is a performance cost
//  to having such packed ops "near" MFMAs. Concretely/specifically this
//  eliminates `v_pk_mul_f32`/`v_pk_add_f32` operations in the final asm in bbs
//  with MFMAs.
//
//  Note, these "scalar" floating point ops will still get lowered to vector
//  instructions like `v_mul_f32_e32 v1, v163, v114` and
//  `v_add_u32_e32 v1, s16, v12`, just not the "packed" variants.
//
//  Note, these vectorized `fmul`s aren't actually emitted by triton per se -
//  they are introduced/inserted by the VectorCombine::foldPermuteOfBinops
//  pattern during the `optimize_module` pipeline (hence why this LLVM pass
//  needs to follow that pipeline).
struct ScalarizePackedFOps : FunctionPass {
  ScalarizePackedFOps() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    IRBuilder<> builder(F.getContext());
    bool changed = false;
    SmallVector<Instruction *> instsToErase;
    for (BasicBlock &BB : F) {
      if (!llvm::any_of(BB, isMFMAorWMMA))
        continue;
      for (Instruction &inst : BB) {
        // fmuladd/fma intrinsics on vectors -> per-element scalar intrinsics
        // (otherwise they stay v_pk_fma_f32; see getScalarizableFPIntrinsic).
        if (maybeReplaceVectorFMAWithScalarFMAs(&inst, builder)) {
          instsToErase.push_back(&inst);
          changed = true;
          continue;
        }
        if (inst.getOpcode() != Instruction::FMul &&
            inst.getOpcode() != Instruction::FAdd &&
            inst.getOpcode() != Instruction::FSub)
          continue;
        if (maybeReplaceVectorFOpWithScalarFOps(&inst, builder)) {
          instsToErase.push_back(&inst);
          changed = true;
        }
      }
    }

    if (changed) {
      for (Instruction *inst : instsToErase) {
        if (inst)
          inst->eraseFromParent();
      }
    }

    // We don't do anything with this but this is a virtual function override
    // and the signature requires it.
    return changed;
  }

  static char ID;
};

} // end anonymous namespace

char ScalarizePackedFOps::ID = 0;

namespace mlir::triton::AMD {
void runScalarizePackedFOpsPass(Function &F) {
  ScalarizePackedFOps pass;
  pass.runOnFunction(F);
  // If there are no errors, the function returns false.
  assert(!llvm::verifyFunction(F) &&
         "expected function to verify successfully");
}
} // namespace mlir::triton::AMD
