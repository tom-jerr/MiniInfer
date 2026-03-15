import torch
import time
import triton
import triton.language as tl


def flash_attn_triton(q, k, v, causal=True, sm_scale=1):
  # shape constraints
  Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
  assert Lq == Lk and Lk == Lv
  assert Lk in {16, 32, 64, 128}

  o = torch.empty_like(q)

  BLOCK_M = 128
  BLOCK_N = 64
  # NOTE: 对于flash attention 2, 外层循环的q可以并行处理, 因此每个thread需要计算正确的offset
  # 一个q, k, v的shape往往是(bs, head, seqlen, dim)
  # 对于(bs, head)中的每个元素都分配一个thread
  # 对于seqlen / BLOCK_M个的q分块, 每个分块再分配一个thread
  grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
  # NOTE:
  # L.shape = (bs * head, seqlen)
  # L记录了所有的分母和mi(m_i + tl.math.log2(l_i)), 用于后续的backward
  L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
  # 设置适当的wrap以提升性能
  num_warps = 4 if Lk <= 64 else 8
  _fwd_kernel[grid](
    q,
    k,
    v,
    sm_scale,
    L,
    o,
    q.stride(0),
    q.stride(1),
    q.stride(2),
    q.stride(3),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    k.stride(3),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    v.stride(3),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    o.stride(3),
    q.shape[0],
    q.shape[1],
    q.shape[2],
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
    DIM=Lk,
    IS_CAUSAL=causal,
    num_warps=num_warps,
    num_stages=4,
  )

  return o


@triton.jit
def _fwd_kernel(
  Q,
  K,
  V,
  sm_scale,
  # L记录了所有的分母和mi, 用于后续的backward
  L,
  O,
  stride_q_bs,
  stride_q_head,
  stride_q_seqlen,
  stride_q_dim,
  stride_k_bs,
  stride_k_head,
  stride_k_seqlen,
  stride_k_dim,
  stride_v_bs,
  stride_v_head,
  stride_v_seqlen,
  stride_v_dim,
  stride_o_bs,
  stride_o_head,
  stride_o_seqlen,
  stride_o_dim,
  BS,
  HEAD,
  SEQLEN,
  # BLOCK_M用于做 Q 的分块
  BLOCK_M: tl.constexpr,
  DIM: tl.constexpr,
  # BLOCK_N用于做 K 和 V 的分块
  BLOCK_N: tl.constexpr,
  IS_CAUSAL: tl.constexpr,
):
  # grid = (cdiv(seqlen, BLOCK_M), bs * head)
  # triton.language.program_id(axis) axis is The axis of the 3D launch grid
  # Q分块的起始地址
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  # NOTE:
  # base = off_bs_head * stride_q_head找到正确的(bs, head)位置
  # strides: 步长, advance时直接使用步数, 会自动根据步长计算跳过的元素
  # offsets表示parent block (seqlen, dim)中怎么偏移来获取小块
  # block_shape=(BLOCK_M, DIM)表示parent block的shape
  # order表示用什么顺序读取存储来构造所需的shape
  qkv_base_offset = off_bs_head * stride_q_head

  Q_block_ptr = tl.make_block_ptr(
    base=Q + qkv_base_offset,
    shape=(SEQLEN, DIM),
    strides=(stride_q_seqlen, stride_q_dim),
    offsets=(start_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, DIM),
    order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
    # base offset to skip to the right (bs, head)
    base=K + qkv_base_offset,
    # the shape of parent
    # NOTE: make_block_ptr读入时将K转置了
    shape=(DIM, SEQLEN),
    strides=(stride_k_dim, stride_k_seqlen),
    # 每个Q需要遍历整个的k和v
    offsets=(0, 0),
    # K根据BLOCK_N分块
    block_shape=(DIM, BLOCK_N),
    # 读入K的转置
    order=(0, 1),
  )
  V_block_ptr = tl.make_block_ptr(
    # base offset to skip to the right (bs, head)
    base=V + qkv_base_offset,
    # the shape of parent
    shape=(SEQLEN, DIM),
    strides=(stride_v_seqlen, stride_v_dim),
    # 每个Q需要遍历整个的k和v
    offsets=(0, 0),
    # K根据BLOCK_N分块
    block_shape=(BLOCK_N, DIM),
    order=(1, 0),
  )
  # initialize offsets
  # NOTE: BLOCK_M表示Q的分块大小, BLOCK_N表示k, v的分块大小
  off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  off_n = tl.arange(0, BLOCK_N)
  # initialize pointers
  # NOTE: 一次处理一个(BLOCK_M, dim)的q, 而max和分母的sum都只需要一维, 即(BLOCK_M, 1)
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  # 分母累加的sum, 每行的sum是一样的, 所以只需要一维然后广播即可
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  o_i = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

  # scale sm_scale by log_2(e) and use
  # 2^x instead of exp in the loop because CSE and LICM
  # don't work as expected with `exp` in the loop
  # CSE(common subexpression elimination), LICM(loop invariant code motion)是编译器里的东西
  qk_scale = sm_scale * 1.44269504

  # load q: stay in SRAM throughout
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)

  lo = 0  # loop over k, v
  # NOTE:: CAUSAL就是常说的不能看到后面的文本的自回归模型
  hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN

  for start_n in range(lo, hi, BLOCK_N):
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    if IS_CAUSAL:
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    qk += tl.dot(q, k)

    m_new = tl.maximum(m_i, tl.max(qk, 1))
    exp = tl.math.exp2(m_i - m_new)
    p_ij = tl.math.exp2(qk - m_new[:, None])
    l_new = l_i * exp + tl.sum(p_ij, 1)
    o_new = o_i * exp[:, None] + tl.dot(p_ij.to(tl.float16), v)

    m_i = m_new
    l_i = l_new
    o_i = o_new

    # NOTE: 计算下一个k, v的分块
    # 因为k已经转置(dim, seqlen), 所以算下一批seq的k时是增加k的第二个维度
    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

  # write back the output
  o_i = o_i / l_i[:, None]

  # NOTE: 将分母和mi保存到L中, 用于后续的backward
  # L.shape = (bs * head, seqlen), 因为每一行的分母和mi是相同的
  # off_bs_head = bs * head
  l_ptr = L + off_bs_head * SEQLEN + off_m
  # write [BLOCK_M] of data to L
  tl.store(l_ptr, m_i + tl.math.log2(l_i))
  # write back O
  O_block_ptr = tl.make_block_ptr(
    base=O + qkv_base_offset,
    shape=(SEQLEN, DIM),
    strides=(stride_o_seqlen, stride_o_dim),
    offsets=(start_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, DIM),
    order=(1, 0),
  )
  tl.store(O_block_ptr, o_i.to(tl.float16))


def ref_attn(q, k, v, causal=True, sm_scale=1):
  SEQLEN = q.shape[-2]
  M = torch.tril(torch.ones((SEQLEN, SEQLEN), device="cuda"))
  p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
  if causal:
    p[:, :, M == 0] = float("-inf")
  p = torch.softmax(p.float(), dim=-1).half()
  ref_out = torch.matmul(p, v)
  return ref_out


def causal_test(BS, HEAD, SEQLEN, DIM, causal):
  dtype = torch.float16
  torch.manual_seed(20)
  q = (
    torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda")
    .normal_(mean=0.0, std=0.5)
    .requires_grad_()
  )
  k = (
    torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda")
    .normal_(mean=0.0, std=0.5)
    .requires_grad_()
  )
  v = (
    torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda")
    .normal_(mean=0.0, std=0.5)
    .requires_grad_()
  )
  sm_scale = 0.5

  # reference implementation
  time_ref = time.time()
  ref_out = ref_attn(q, k, v, causal=causal, sm_scale=sm_scale)
  time_ref = time.time() - time_ref

  # triton implementation
  time_tri = time.time()
  tri_out = flash_attn_triton(q, k, v, causal=causal, sm_scale=sm_scale).half()
  time_tri = time.time() - time_tri

  # compare
  assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
  print(
    "causal = {} ref time: {:.4f} ms, tri time: {:.4f}".format(
      causal, time_ref * 1000, time_tri * 1000
    )
  )


def test_attention():
  BS, HEAD, SEQLEN, DIM = 1, 2, 1024, 64
  causal_test(BS, HEAD, SEQLEN, DIM, causal=False)
  causal_test(BS, HEAD, SEQLEN, DIM, causal=True)


if __name__ == "__main__":
  test_attention()
