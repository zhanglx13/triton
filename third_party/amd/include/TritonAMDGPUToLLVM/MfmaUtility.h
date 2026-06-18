#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MFMAUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MFMAUTILITY_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

namespace mlir::triton::AMD {

// True iff `I` is an AMDGPU matrix-core intrinsic call (MFMA or WMMA). A given
// target exposes only one of the two families, so callers never need to tell
// them apart. Inline-asm calls and non-intrinsic calls are rejected.
inline bool isMFMAorWMMA(const llvm::Instruction &I) {
  const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
  if (!CI || CI->isInlineAsm())
    return false;
  const llvm::Function *Callee = CI->getCalledFunction();
  if (!Callee || !Callee->isIntrinsic())
    return false;
  llvm::StringRef Name = Callee->getName();
  return Name.contains("mfma") || Name.contains("wmma");
}

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MFMAUTILITY_H_
