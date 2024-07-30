#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"

using llvm::MapVector;
using namespace mlir;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"


namespace {

    bool canHoistAndAggregate(triton::LoadOp loadOp, int ub) {
	auto loadTensorType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
	auto shape = loadTensorType.getShape();
	llvm::outs() << "Loaded tensor shape: " << shape[0] << " by " << shape[1] << "\n";
	// Assume each load has one use, which is the convert_layout #blocked to #dotOp
	Operation *use = *loadOp.getResult().getUsers().begin();
	auto cvt = llvm::dyn_cast<ttg::ConvertLayoutOp>(use);
	auto tensorType =
	    dyn_cast<RankedTensorType>(cvt.getResult().getType());
	auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
	    tensorType.getEncoding());
	auto opIdx = dotOpEnc.getOpIdx();
	llvm::outs() << "opIdx = " << opIdx << "\n";
	// Assume we don't want to hoist load for operand B
	if (opIdx == 1)
	    return false;

	// Assume fp16 element type
	// Also assume operand B does not use LDS
	int total_LDS = shape[0] * shape[1] * ub * 2;
	assert(total_LDS <= 65536 && "BLOCK_M * K is too large");

	return true;
    }




    void findValidLoads(scf::ForOp forOp, SetVector<Operation *> &validLoads, int ub) {
    for (Operation &op : forOp) {
	if (auto loadOp = dyn_cast<triton::LoadOp>(&op)){
	    llvm::outs() << "Found a loadOp: ";
	    loadOp.dump();
	    if (canHoistAndAggregate(loadOp, ub))
		validLoads.insert(loadOp);
	}
    }
}

    int isUpperBoundConstant(scf::ForOp forOp) {
	auto ub = forOp.getUpperBound();
	if (auto constant = dyn_cast<arith::ConstantOp>(ub.getDefiningOp())) {
	    return cast<IntegerAttr>(constant.getValue()).getInt();
	}
	else {
	    llvm::outs() << "Non constant upper bound??\n";
	    return 0;
	}
    }

// Stream Pipeline
struct AggregateLoad : public TritonAMDGPUAggregateLoadBase<AggregateLoad> {
  AggregateLoad() = default;

  void runOnOperation() override {

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
	// We need K to be constant, i.e. the upper bound of the loop is a constant
	auto ub = isUpperBoundConstant(forOp);
	if(!ub)
	    return;
	llvm::outs() << "Found a for loop with constant upper bound " << ub << "\n";
	SetVector<Operation *> validLoads;
	findValidLoads(forOp, validLoads, ub);
    });
  }
};
}


std::unique_ptr<Pass> mlir::createTritonAMDGPUAggregateLoadPass() {
    return std::make_unique<AggregateLoad>();
}
