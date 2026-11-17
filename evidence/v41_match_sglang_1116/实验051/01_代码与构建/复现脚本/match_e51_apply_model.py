from pathlib import Path
import hashlib,json
r=Path("/home/xiaopang/桌面/Albert分享/20261021从xllm开始/王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/03_代码/xllm-musa");records={}
def edit(name,changes):
 p=r/name;s=p.read_text();records[name]={'before':hashlib.sha256(p.read_bytes()).hexdigest()}
 for a,b,count in changes:
  assert s.count(a)==count,(name,a,s.count(a));s=s.replace(a,b)
 p.write_text(s)
p='xllm/core/layers/musa/deepseek_v41/'
edit(p+'text_model.h',[
 ('#include "core/layers/musa/deepseek_v41/decoder.h"','#include "core/layers/musa/deepseek_v41/decoder.h"\n#include "core/layers/musa/deepseek_v41/gather_bits.h"',1),
 ('bool owner_attention = false)','bool owner_attention = false,\n                       bool gather_bf16_bits = false)',1),
 ('    owner_attention_ = owner_attention;','    owner_attention_ = owner_attention;\n    gather_bf16_bits_ = gather_bf16_bits;',1),
 ('parallel_state::gather(hidden, attention_group_, /*dim=*/-1)','deepseek_v41_gather_bf16_bits(hidden, attention_group_, gather_bf16_bits_)',2),
 ('parallel_state::gather(local, attention_group_, /*dim=*/-1)','deepseek_v41_gather_bf16_bits(local, attention_group_, gather_bf16_bits_)',1),
 ('  bool owner_attention_ = false;','  bool owner_attention_ = false;\n  bool gather_bf16_bits_ = false;',1)])
edit(p+'draft_model.h',[
 ('#include "core/layers/musa/deepseek_v41/feed_forward.h"','#include "core/layers/musa/deepseek_v41/feed_forward.h"\n#include "core/layers/musa/deepseek_v41/gather_bits.h"',1),
 ('ProcessGroup* expert_group = nullptr)','ProcessGroup* expert_group = nullptr,\n                        bool gather_bf16_bits = false)',1),
 ('        attention_group_(attention_group) {','        attention_group_(attention_group),\n        gather_bf16_bits_(gather_bf16_bits) {',1),
 ('parallel_state::gather(hidden, attention_group_, /*dim=*/-1)','deepseek_v41_gather_bf16_bits(hidden, attention_group_, gather_bf16_bits_)',1),
 ('parallel_state::gather(local, attention_group_, /*dim=*/-1)','deepseek_v41_gather_bf16_bits(local, attention_group_, gather_bf16_bits_)',1),
 ('  ProcessGroup* attention_group_;','  ProcessGroup* attention_group_;\n  bool gather_bf16_bits_ = false;',1)])
p='xllm/models/llm/musa/'
init='''    gather_bf16_bits_ =
        util::get_bool_env("XLLM_DSV41_GATHER_BF16_BITS", false);
    CHECK(!gather_bf16_bits_ ||
          (tp_size() == 8 && musa_custom_allreduce_enabled(tp_group_)))
        << "BF16 bit gather requires the initialized TP8 custom communicator";
'''
edit(p+'deepseek_v41_native.h',[
 ('    native_dspark_ = util::get_bool_env("XLLM_DSV41_DSPARK_NATIVE", false);','    native_dspark_ = util::get_bool_env("XLLM_DSV41_DSPARK_NATIVE", false);\n'+init,1),
 ('        owner_attention_);','        owner_attention_,\n        gather_bf16_bits_);',1),
 ('(static_cast<int32_t>(fused_spec_snapshot_) << 3);','(static_cast<int32_t>(fused_spec_snapshot_) << 3) |\n        (static_cast<int32_t>(gather_bf16_bits_) << 4);',1),
 ('  bool owner_gather_outside_graph_ = false;','  bool owner_gather_outside_graph_ = false;\n  bool gather_bf16_bits_ = false;',1)])
edit(p+'deepseek_v41_dspark_native.h',[
 ('    static_draft_ = util::get_bool_env("XLLM_DSV41_STATIC_DRAFT", false);',init+'    static_draft_ = util::get_bool_env("XLLM_DSV41_STATIC_DRAFT", false);',1),
 ('        draft_expert_reduction_group());','        draft_expert_reduction_group(),\n        gather_bf16_bits_);',1),
 ('(static_cast<int32_t>(full_draft_graph_) << 2);','(static_cast<int32_t>(full_draft_graph_) << 2) |\n                           (static_cast<int32_t>(gather_bf16_bits_) << 3);',1),
 ('  bool markov_fp32_logits_ = false;','  bool markov_fp32_logits_ = false;\n  bool gather_bf16_bits_ = false;',1)])
p='xllm/core/layers/musa/deepseek_v41/gather_bits.h';assert not (r/p).exists();(r/p).write_bytes(Path('/tmp/v41_gather_bits.h').read_bytes());records[p]={'before':None}
Path('/tmp/match_e51_source_before.json').write_text(json.dumps(records,indent=2))
