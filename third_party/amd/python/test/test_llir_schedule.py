"""Tests for the gfx950 LLIR pre-RA scheduler.

The scheduler is an opt-in LLVM-IR FunctionPass: it runs only when the caller
passes schedule_hint="gemm-4waves" and the target is gfx950. When it takes
effect it also forces MFMA accumulators into AGPRs (amdgpu-agpr-alloc=256 +
amdgpu-mfma-vgpr-form=0) to free VGPRs.

These tests compile a pipelined fp16 GEMM (gemm_fp16.ttgir) through the AMD
backend and assert on the resulting LLVM IR / assembly, mirroring the approach
used by test_scalarize_packed_fops.py and test_llvm_fn_attrs.py. The fixture is
a software-pipelined hot loop (MFMAs interleaved with LDS reads / async global
loads) so the scheduler has real regions to interleave.
"""
from pathlib import Path

import pytest
import triton

current_target = triton.runtime.driver.active.get_current_target()
if current_target is None or current_target.arch != "gfx950":
    pytest.skip("LLIR scheduler is gfx950-only", allow_module_level=True)

GEMM_TTIR = str(Path(__file__).parent / "gemm_fp16.ttgir")

# Function attribute the scheduler adds when it takes effect (RA flag #1).
AGPR_ATTR = '"amdgpu-agpr-alloc"="256"'
# Marker the scheduler emits at the start of each scheduled region.
REGION_MARKER = ";; Region"


def _compile(**options):
    return triton.compile(GEMM_TTIR, target=current_target, options=options)


def test_scheduler_off_by_default():
    # Without the opt-in hint the pass must not run and must leave codegen
    # untouched: no AGPR-alloc attr, no region markers.
    k = _compile()
    assert k.metadata.llir_scheduled is False
    assert AGPR_ATTR not in k.asm["llir"]
    assert REGION_MARKER not in k.asm["amdgcn"]


def test_unrelated_hint_does_not_trigger():
    # schedule_hint carries other tokens too; only "gemm-4waves" enables this pass.
    k = _compile(schedule_hint="memory-bound-attention")
    assert k.metadata.llir_scheduled is False
    assert AGPR_ATTR not in k.asm["llir"]


def test_gemm_4waves_schedules_and_couples_ra_flags():
    k = _compile(schedule_hint="gemm-4waves")
    # The pass found an eligible MFMA region and scheduled it.
    assert k.metadata.llir_scheduled is True
    # RA flag coupling: the AGPR-alloc attr appears only when scheduling happened.
    assert AGPR_ATTR in k.asm["llir"]
    # The scheduler annotates each region and interleaves MFMA with the loads.
    amdgcn = k.asm["amdgcn"]
    assert REGION_MARKER in amdgcn
    assert "v_mfma" in amdgcn


def test_token_parsing_is_csv_and_case_insensitive():
    # schedule_hint is parsed as a comma-separated, lower-cased set, so the
    # token is recognized among others and regardless of case.
    k = _compile(schedule_hint="some-other-hint, GEMM-4Waves")
    assert k.metadata.llir_scheduled is True
    assert AGPR_ATTR in k.asm["llir"]
