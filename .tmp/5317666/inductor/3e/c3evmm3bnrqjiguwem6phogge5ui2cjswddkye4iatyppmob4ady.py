# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /scratch/u6ex/as1748.u6ex/ARRWM/.tmp/5317666/inductor/vq/cvqmnitekvqszk6zxmomm6bkesfava76drlqmvxkk6e6u33cnibb.py
# Topologically Sorted Source Nodes: [m, eye_mask, n, lt, clean_mask, index, lt_1, or_2, ge_2, noise_mask, index_1, lt_2, index_2, ge, C1, index_3, lt_3, index_4, ge_1, C2, or_1, batched_outputs, batched_outputs_3, mask, mask_2, mask_3, mask_block_sum], Original ATen: [aten.arange, aten.view, aten.eq, aten.lt, aten.index, aten.bitwise_and, aten.bitwise_or, aten.ge, aten.expand, aten.permute, aten.sum]
# Source node to ATen node mapping:
#   C1 => bitwise_and_1
#   C2 => bitwise_and_2
#   batched_outputs => bitwise_or_2
#   batched_outputs_3 => expand
#   clean_mask => bitwise_and, view_1
#   eye_mask => eq, view_7
#   ge => ge, view_3
#   ge_1 => ge_1, view_5
#   ge_2 => ge_2
#   index => index
#   index_1 => index_1
#   index_2 => index_2
#   index_3 => index_3
#   index_4 => index_4
#   lt => lt
#   lt_1 => lt_1, view
#   lt_2 => lt_2, view_2
#   lt_3 => lt_3, view_4
#   m => iota_2
#   mask => expand_1
#   mask_2 => view_8
#   mask_3 => permute
#   mask_block_sum => sum_1
#   n => iota_3
#   noise_mask => bitwise_and_3, view_6
#   or_1 => bitwise_or
#   or_2 => bitwise_or_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[37504][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "i64[37504][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "i64[37504][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "i64[37504][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "i64[37504][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %iota_2 : Tensor "i64[37504][1]cuda:0"[num_users=8] = call_function[target=torch.ops.prims.iota.default](args = (37504,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_7 : Tensor "i64[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [37504, 1]), kwargs = {})
#   %iota_3 : Tensor "i64[37504][1]cuda:0"[num_users=6] = call_function[target=torch.ops.prims.iota.default](args = (37504,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_7, %iota_3), kwargs = {})
#   %lt : Tensor "b8[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_2, 32781), kwargs = {})
#   %view_1 : Tensor "b8[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%lt, [37504, 1]), kwargs = {})
#   %index : Tensor "i64[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%iota_2]), kwargs = {})
#   %view : Tensor "i64[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [37504, 1]), kwargs = {})
#   %lt_1 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_3, %view), kwargs = {})
#   %bitwise_and : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%view_1, %lt_1), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%eq, %bitwise_and), kwargs = {})
#   %ge_2 : Tensor "b8[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_2, 32781), kwargs = {})
#   %view_6 : Tensor "b8[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%ge_2, [37504, 1]), kwargs = {})
#   %index_1 : Tensor "i64[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%iota_2]), kwargs = {})
#   %view_2 : Tensor "i64[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [37504, 1]), kwargs = {})
#   %lt_2 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_3, %view_2), kwargs = {})
#   %index_2 : Tensor "i64[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%iota_2]), kwargs = {})
#   %view_3 : Tensor "i64[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_2, [37504, 1]), kwargs = {})
#   %ge : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%iota_3, %view_3), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt_2, %ge), kwargs = {})
#   %index_3 : Tensor "i64[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg3_1, [%iota_2]), kwargs = {})
#   %view_4 : Tensor "i64[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_3, [37504, 1]), kwargs = {})
#   %lt_3 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_3, %view_4), kwargs = {})
#   %index_4 : Tensor "i64[37504][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg4_1, [%iota_2]), kwargs = {})
#   %view_5 : Tensor "i64[37504, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_4, [37504, 1]), kwargs = {})
#   %ge_1 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%iota_3, %view_5), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt_3, %ge_1), kwargs = {})
#   %bitwise_or : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_1, %bitwise_and_2), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%view_6, %bitwise_or), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[37504, 37504][37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or_1, %bitwise_and_3), kwargs = {})
#   %expand : Tensor "b8[1, 37504, 37504][1406550016, 37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%bitwise_or_2, [1, 37504, 37504]), kwargs = {})
#   %expand_1 : Tensor "b8[1, 1, 37504, 37504][1406550016, 1406550016, 37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%expand, [1, 1, 37504, 37504]), kwargs = {})
#   %view_8 : Tensor "b8[1, 1, 293, 128, 293, 128][1406550016, 1406550016, 4800512, 37504, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_1, [1, 1, 293, 128, 293, 128]), kwargs = {})
#   %permute : Tensor "b8[1, 1, 293, 293, 128, 128][1406550016, 1406550016, 4800512, 128, 37504, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %sum_1 : Tensor "i64[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [-2, -1]), kwargs = {})
#   return %sum_1
triton_red_fused_arange_bitwise_and_bitwise_or_eq_expand_ge_index_lt_permute_sum_view_0 = async_compile.triton('triton_red_fused_arange_bitwise_and_bitwise_or_eq_expand_ge_index_lt_permute_sum_view_0', '''
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
''', device_str='cuda')


# kernel path: /scratch/u6ex/as1748.u6ex/ARRWM/.tmp/5317666/inductor/an/canxbq66iwarzxybwkqv54qmg42nxjlzqlsfvyi6qmvcduqhbuja.py
# Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   dense_mask_4 => full_default_3
# Graph fragment:
#   %full_default_3 : Tensor "i32[1, 1, 293, 294][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 293, 294], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %index_put_1
triton_poi_fused_new_zeros_1 = async_compile.triton('triton_poi_fused_new_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'FE032B48F6119FCDCB54C77114B7A8669BD76F063FCC36FA1DD87062D1860F25', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 689136}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 86142
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /scratch/u6ex/as1748.u6ex/ARRWM/.tmp/5317666/inductor/xe/cxebyiafm64bhcca2wjlgjl6im57zhzccr42norm2n5q2oz3hblh.py
# Topologically Sorted Source Nodes: [gt, lt_4, partial_blocks, partial_blocks_1, dense_mask, col_indices, full_blocks, full_blocks_1, dense_mask_1, col_indices_1, dense_mask_2, setitem, arange_4, row_indices, col_range, num_blocks_in_row, child_3, unsqueeze_1, index_mask, child_4, valid_indices, dense_mask_4, setitem_1, arange_6, row_indices_1, col_range_1, num_blocks_in_row_1, child_7, unsqueeze_3, index_mask_1, child_8, valid_indices_1], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort, aten.eq, aten.new_zeros, aten.arange, aten.unsqueeze, aten.sum, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
# Source node to ATen node mapping:
#   arange_4 => iota_4
#   arange_6 => iota_8
#   child_3 => convert_element_type_3
#   child_4 => convert_element_type_4
#   child_7 => convert_element_type_6
#   child_8 => convert_element_type_7
#   col_indices => sort
#   col_indices_1 => sort_1
#   col_range => iota_5
#   col_range_1 => iota_9
#   dense_mask => convert_element_type_2
#   dense_mask_1 => convert_element_type_5
#   dense_mask_2 => full_default
#   dense_mask_4 => full_default_3
#   full_blocks => eq_1
#   full_blocks_1 => convert_element_type_1
#   gt => gt
#   index_mask => lt_5
#   index_mask_1 => lt_6
#   lt_4 => lt_4
#   num_blocks_in_row => sum_2
#   num_blocks_in_row_1 => sum_3
#   partial_blocks => bitwise_and_4
#   partial_blocks_1 => convert_element_type
#   row_indices => unsqueeze
#   row_indices_1 => unsqueeze_7
#   setitem => full_default_2, index_put, iota_6, iota_7, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6
#   setitem_1 => full_default_5, index_put_1, iota_10, iota_11, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_9
#   unsqueeze_1 => unsqueeze_1
#   unsqueeze_3 => unsqueeze_8
#   valid_indices => full_default_1, where
#   valid_indices_1 => full_default_4, where_1
# Graph fragment:
#   %sum_1 : Tensor "i64[1, 1, 293, 293][85856, 85856, 293, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0" = PlaceHolder[target=sum_3]
#   %buf2 : Tensor "i16[1, 1, 293, 293][85888, 85888, 293, 1]cuda:0" = PlaceHolder[target=buf2]
#   %convert_element_type_3 : Tensor "i32[1, 1, 293][293, 293, 1]cuda:0" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_4 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0" = PlaceHolder[target=convert_element_type_4]
#   %index_put : Tensor "i32[1, 1, 293, 294][86144, 86144, 294, 1]cuda:0" = PlaceHolder[target=index_put]
#   %buf4 : Tensor "i16[1, 1, 293, 293][85888, 85888, 293, 1]cuda:0" = PlaceHolder[target=buf4]
#   %convert_element_type_6 : Tensor "i32[1, 1, 293][293, 293, 1]cuda:0" = PlaceHolder[target=convert_element_type_6]
#   %convert_element_type_7 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0" = PlaceHolder[target=convert_element_type_7]
#   %index_put_1 : Tensor "i32[1, 1, 293, 294][86144, 86144, 294, 1]cuda:0" = PlaceHolder[target=index_put_1]
#   %gt : Tensor "b8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %lt_4 : Tensor "b8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %lt_4), kwargs = {})
#   %convert_element_type : Tensor "i8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_4, torch.int8), kwargs = {})
#   %convert_element_type_2 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.int32), kwargs = {})
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.stable](args = (%convert_element_type_2,), kwargs = {stable: True, descending: True})
#   %eq_1 : Tensor "b8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %convert_element_type_1 : Tensor "i8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq_1, torch.int8), kwargs = {})
#   %convert_element_type_5 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.int32), kwargs = {})
#   %sort_1 : [num_users=1] = call_function[target=torch.ops.aten.sort.stable](args = (%convert_element_type_5,), kwargs = {stable: True, descending: True})
#   %full_default : Tensor "i32[1, 1, 293, 294][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 293, 294], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_7 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_7, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, -1), kwargs = {})
#   %iota_6 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_6, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %iota_4 : Tensor "i32[293][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (293,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i32[293, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_4, -1), kwargs = {})
#   %iota_5 : Tensor "i32[293][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (293,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %sum_2 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_2, [-1]), kwargs = {})
#   %convert_element_type_3 : Tensor "i32[1, 1, 293][293, 293, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.int32), kwargs = {})
#   %unsqueeze_1 : Tensor "i32[1, 1, 293, 1][293, 293, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 3), kwargs = {})
#   %lt_5 : Tensor "b8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_5, %unsqueeze_1), kwargs = {})
#   %convert_element_type_4 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1, torch.int32), kwargs = {})
#   %full_default_1 : Tensor "i32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 293), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_5, %convert_element_type_4, %full_default_1), kwargs = {})
#   %full_default_2 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "i32[1, 1, 293, 294][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%unsqueeze_6, %unsqueeze_3, %unsqueeze, %where], %full_default_2), kwargs = {})
#   %full_default_3 : Tensor "i32[1, 1, 293, 294][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 293, 294], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_11 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_11 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_11, -1), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %iota_10 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_9 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_10, -1), kwargs = {})
#   %unsqueeze_10 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, -1), kwargs = {})
#   %iota_8 : Tensor "i32[293][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (293,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %unsqueeze_7 : Tensor "i32[293, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_8, -1), kwargs = {})
#   %iota_9 : Tensor "i32[293][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (293,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %sum_3 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_5, [-1]), kwargs = {})
#   %convert_element_type_6 : Tensor "i32[1, 1, 293][293, 293, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.int32), kwargs = {})
#   %unsqueeze_8 : Tensor "i32[1, 1, 293, 1][293, 293, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_6, 3), kwargs = {})
#   %lt_6 : Tensor "b8[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_9, %unsqueeze_8), kwargs = {})
#   %convert_element_type_7 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_3, torch.int32), kwargs = {})
#   %full_default_4 : Tensor "i32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 293), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_6, %convert_element_type_7, %full_default_4), kwargs = {})
#   %full_default_5 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : Tensor "i32[1, 1, 293, 294][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_3, [%unsqueeze_13, %unsqueeze_10, %unsqueeze_7, %where_1], %full_default_5), kwargs = {})
#   return %buf2,%buf4,%sum_2,%sum_3,%convert_element_type_3,%convert_element_type_6,%convert_element_type_4,%buf9,%convert_element_type_7,%buf16
triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_2 = async_compile.triton('triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr4': '*i32', 'out_ptr5': '*i32', 'out_ptr6': '*i32', 'out_ptr7': '*i32', 'out_ptr8': '*i32', 'out_ptr9': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_2', 'mutated_arg_names': ['out_ptr7', 'out_ptr9'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 6, 'num_reduction': 2, 'backend_hash': 'FE032B48F6119FCDCB54C77114B7A8669BD76F063FCC36FA1DD87062D1860F25', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_2(in_ptr0, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 293
    r0_numel = 293
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 293*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tl.full([1, 1], 16384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.int8)
    tmp7 = tmp6.to(tl.int32)
    tmp8 = r0_1
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, rnumel, 1, stable=True, descending=True)
    tmp14 = tmp0 == tmp3
    tmp15 = tmp14.to(tl.int8)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp18, tmp19, = triton_helpers.sort_with_index(tmp17, tmp11, rnumel, 1, stable=True, descending=True)
    tmp20 = tmp7.to(tl.int64)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.int64)
    tmp25 = tmp16.to(tl.int64)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.int64)
    tmp30 = tmp24.to(tl.int32)
    tmp31 = tmp29.to(tl.int32)
    tmp32 = tmp13.to(tl.int64)
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tmp8 < tmp30
    tmp35 = tl.full([1, 1], 293, tl.int32)
    tmp36 = tl.where(tmp34, tmp33, tmp35)
    tmp37 = tl.full([1, 1], 294, tl.int32)
    tmp38 = tmp36 + tmp37
    tmp39 = tmp36 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp36)
    tl.device_assert(((0 <= tmp40) & (tmp40 < 294)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp40 < 294")
    tmp42 = tl.full([1, 1], 1, tl.int32)
    tmp43 = tmp19.to(tl.int64)
    tmp44 = tmp43.to(tl.int32)
    tmp45 = tmp8 < tmp31
    tmp46 = tl.where(tmp45, tmp44, tmp35)
    tmp47 = tmp46 + tmp37
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 294)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp49 < 294")
    tl.store(out_ptr4 + (x0), tmp30, xmask)
    tl.store(out_ptr5 + (x0), tmp31, xmask)
    tl.store(out_ptr6 + (r0_1 + 293*x0), tmp33, r0_mask & xmask)
    tl.store(out_ptr7 + (tl.broadcast_to(tmp40 + 294*x0, [XBLOCK, R0_BLOCK])), tmp42, r0_mask & xmask)
    tl.store(out_ptr8 + (r0_1 + 293*x0), tmp44, r0_mask & xmask)
    tl.store(out_ptr9 + (tl.broadcast_to(tmp49 + 294*x0, [XBLOCK, R0_BLOCK])), tmp42, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/u6ex/as1748.u6ex/ARRWM/.tmp/5317666/inductor/yt/cyty54djelfxmch4msrjgnwjlksmv2i44b7bqppywxomsnkn4u66.py
# Topologically Sorted Source Nodes: [batched_outputs_4, transpose, col_indices_2, q_indices], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort, aten._to_copy]
# Source node to ATen node mapping:
#   batched_outputs_4 => clone_4, slice_2
#   col_indices_2 => sort_2
#   q_indices => clone_6, convert_element_type_9
#   transpose => permute_1
# Graph fragment:
#   %buf9 : Tensor "i32[1, 1, 293, 294][86144, 86144, 294, 1]cuda:0" = PlaceHolder[target=buf9]
#   %buf11 : Tensor "i16[1, 1, 293, 293][85888, 85888, 293, 1]cuda:0" = PlaceHolder[target=buf11]
#   %slice_2 : Tensor "i32[1, 1, 293, 293][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put, 3, 0, 293), kwargs = {})
#   %clone_4 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_1 : Tensor "i32[1, 1, 293, 293][85849, 85849, 1, 293]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%clone_4, [0, 1, 3, 2]), kwargs = {})
#   %sort_2 : [num_users=1] = call_function[target=torch.ops.aten.sort.stable](args = (%permute_1,), kwargs = {stable: True, descending: True})
#   %convert_element_type_9 : Tensor "i32[1, 1, 293, 293][85849, 85849, 1, 293]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_5, torch.int32), kwargs = {})
#   %clone_6 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_9,), kwargs = {memory_format: torch.contiguous_format})
#   return %buf11,%clone_6
triton_per_fused__to_copy_clone_slice_sort_transpose_3 = async_compile.triton('triton_per_fused__to_copy_clone_slice_sort_transpose_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clone_slice_sort_transpose_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'FE032B48F6119FCDCB54C77114B7A8669BD76F063FCC36FA1DD87062D1860F25', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 686792}}
)
@triton.jit
def triton_per_fused__to_copy_clone_slice_sort_transpose_3(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 293
    r0_numel = 293
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 294*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = r0_1
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, rnumel, 1, stable=True, descending=True)
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr1 + (r0_1 + 293*x0), tmp8, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /scratch/u6ex/as1748.u6ex/ARRWM/.tmp/5317666/inductor/ti/ctiox4ztj6jz6rm32li6rbwizl6xz7dyv2bsd53em7egeab2ehbq.py
# Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, num_blocks_in_row_3], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum]
# Source node to ATen node mapping:
#   batched_outputs_6 => clone_7, slice_4
#   num_blocks_in_row_3 => sum_5
#   transpose_1 => permute_2
# Graph fragment:
#   %buf16 : Tensor "i32[1, 1, 293, 294][86144, 86144, 294, 1]cuda:0" = PlaceHolder[target=buf16]
#   %slice_4 : Tensor "i32[1, 1, 293, 293][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put_1, 3, 0, 293), kwargs = {})
#   %clone_7 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_2 : Tensor "i32[1, 1, 293, 293][85849, 85849, 1, 293]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%clone_7, [0, 1, 3, 2]), kwargs = {})
#   %sum_5 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_2, [-1]), kwargs = {})
#   return %buf20
triton_red_fused_clone_slice_sum_transpose_4 = async_compile.triton('triton_red_fused_clone_slice_sum_transpose_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_slice_sum_transpose_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'FE032B48F6119FCDCB54C77114B7A8669BD76F063FCC36FA1DD87062D1860F25', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 14064, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_clone_slice_sum_transpose_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 879
    r0_numel = 98
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 293
    x0 = (xindex % 293)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 98*x1
        tmp1 = tl.full([1, 1], 293, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 294*r0_2 + 28812*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp3.to(tl.int64)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /scratch/u6ex/as1748.u6ex/ARRWM/.tmp/5317666/inductor/lj/cljnvfywrtecdjczzcvvbi5rn75ous3fyl3et422gx6cyu7fugti.py
# Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
# Source node to ATen node mapping:
#   batched_outputs_6 => clone_7, slice_4
#   full_q_num_blocks => convert_element_type_10
#   num_blocks_in_row_3 => sum_5
#   transpose_1 => permute_2
# Graph fragment:
#   %buf20 : Tensor "i64[1, 1, 293, 3][879, 879, 1, 293]cuda:0" = PlaceHolder[target=buf20]
#   %sum_5 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0" = PlaceHolder[target=sum_5]
#   %slice_4 : Tensor "i32[1, 1, 293, 293][86142, 86142, 294, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put_1, 3, 0, 293), kwargs = {})
#   %clone_7 : Tensor "i32[1, 1, 293, 293][85849, 85849, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_2 : Tensor "i32[1, 1, 293, 293][85849, 85849, 1, 293]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%clone_7, [0, 1, 3, 2]), kwargs = {})
#   %sum_5 : Tensor "i64[1, 1, 293][293, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_2, [-1]), kwargs = {})
#   %convert_element_type_10 : Tensor "i32[1, 1, 293][293, 293, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_5, torch.int32), kwargs = {})
#   return %sum_5,%convert_element_type_10
triton_per_fused__to_copy_clone_slice_sum_transpose_5 = async_compile.triton('triton_per_fused__to_copy_clone_slice_sum_transpose_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clone_slice_sum_transpose_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'FE032B48F6119FCDCB54C77114B7A8669BD76F063FCC36FA1DD87062D1860F25', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9376, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_clone_slice_sum_transpose_5(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 293
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 293*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.int64)
    tmp5 = tmp4.to(tl.int32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (37504, ), (1, ))
        assert_size_stride(arg1_1, (37504, ), (1, ))
        assert_size_stride(arg2_1, (37504, ), (1, ))
        assert_size_stride(arg3_1, (37504, ), (1, ))
        assert_size_stride(arg4_1, (37504, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 1, 293, 293), (85856, 85856, 293, 1), torch.int64)
            # Topologically Sorted Source Nodes: [m, eye_mask, n, lt, clean_mask, index, lt_1, or_2, ge_2, noise_mask, index_1, lt_2, index_2, ge, C1, index_3, lt_3, index_4, ge_1, C2, or_1, batched_outputs, batched_outputs_3, mask, mask_2, mask_3, mask_block_sum], Original ATen: [aten.arange, aten.view, aten.eq, aten.lt, aten.index, aten.bitwise_and, aten.bitwise_or, aten.ge, aten.expand, aten.permute, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_arange_bitwise_and_bitwise_or_eq_expand_ge_index_lt_permute_sum_view_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf0, 85849, 16384, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
            buf15 = empty_strided_cuda((1, 1, 293, 294), (86144, 86144, 294, 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
            stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_1.run(buf15, 86142, stream=stream0)
            buf8 = empty_strided_cuda((1, 1, 293, 294), (86144, 86144, 294, 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
            stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_1.run(buf8, 86142, stream=stream0)
            buf6 = empty_strided_cuda((1, 1, 293), (293, 293, 1), torch.int32)
            buf13 = empty_strided_cuda((1, 1, 293), (293, 293, 1), torch.int32)
            buf7 = empty_strided_cuda((1, 1, 293, 293), (85849, 85849, 293, 1), torch.int32)
            buf14 = empty_strided_cuda((1, 1, 293, 293), (85849, 85849, 293, 1), torch.int32)
            # Topologically Sorted Source Nodes: [gt, lt_4, partial_blocks, partial_blocks_1, dense_mask, col_indices, full_blocks, full_blocks_1, dense_mask_1, col_indices_1, dense_mask_2, setitem, arange_4, row_indices, col_range, num_blocks_in_row, child_3, unsqueeze_1, index_mask, child_4, valid_indices, dense_mask_4, setitem_1, arange_6, row_indices_1, col_range_1, num_blocks_in_row_1, child_7, unsqueeze_3, index_mask_1, child_8, valid_indices_1], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort, aten.eq, aten.new_zeros, aten.arange, aten.unsqueeze, aten.sum, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_2.run(buf0, buf6, buf13, buf7, buf8, buf14, buf15, 293, 293, stream=stream0)
            del buf0
            buf23 = empty_strided_cuda((1, 1, 293, 293), (85849, 85849, 293, 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_4, transpose, col_indices_2, q_indices], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_clone_slice_sort_transpose_3.run(buf8, buf23, 293, 293, stream=stream0)
            buf19 = empty_strided_cuda((1, 1, 293, 293), (85849, 85849, 293, 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, col_indices_3, full_q_indices], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_clone_slice_sort_transpose_3.run(buf15, buf19, 293, 293, stream=stream0)
            buf20 = empty_strided_cuda((1, 1, 293, 3), (879, 879, 1, 293), torch.int64)
            # Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, num_blocks_in_row_3], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_clone_slice_sum_transpose_4.run(buf15, buf20, 879, 98, stream=stream0)
            del buf15
            buf22 = empty_strided_cuda((1, 1, 293), (293, 293, 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_clone_slice_sum_transpose_5.run(buf20, buf22, 293, 3, stream=stream0)
            buf24 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [batched_outputs_4, transpose, num_blocks_in_row_2], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_clone_slice_sum_transpose_4.run(buf8, buf24, 879, 98, stream=stream0)
            del buf8
            buf26 = empty_strided_cuda((1, 1, 293), (293, 293, 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_4, transpose, num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_clone_slice_sum_transpose_5.run(buf24, buf26, 293, 3, stream=stream0)
            del buf24
        return (buf19, buf22, buf23, buf26, buf14, buf13, buf7, buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((37504, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((37504, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((37504, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((37504, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((37504, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
