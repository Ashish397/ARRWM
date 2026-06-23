
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 16384},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_arange_bitwise_and_bitwise_or_eq_expand_ge_index_lt_permute_sum_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'FE032B48F6119FCDCB54C77114B7A8669BD76F063FCC36FA1DD87062D1860F25', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1373584, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_arange_bitwise_and_bitwise_or_eq_expand_ge_index_lt_permute_sum_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 85849
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 293
    x0 = (xindex % 293)
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 128
        r0_2 = (r0_index % 128)
        tmp5 = tl.load(in_ptr0 + (r0_3 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r0_3 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r0_3 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr3 + (r0_3 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr4 + (r0_3 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r0_3 + 128*x1
        tmp1 = r0_2 + 128*x0
        tmp2 = tmp0 == tmp1
        tmp3 = tl.full([1, 1], 32781, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp6 = tmp1 < tmp5
        tmp7 = tmp4 & tmp6
        tmp8 = tmp2 | tmp7
        tmp9 = tmp0 >= tmp3
        tmp11 = tmp1 < tmp10
        tmp13 = tmp1 >= tmp12
        tmp14 = tmp11 & tmp13
        tmp16 = tmp1 < tmp15
        tmp18 = tmp1 >= tmp17
        tmp19 = tmp16 & tmp18
        tmp20 = tmp14 | tmp19
        tmp21 = tmp9 & tmp20
        tmp22 = tmp8 | tmp21
        tmp23 = tmp22.to(tl.int64)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp25, xmask)
