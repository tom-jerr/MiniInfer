import torch
from torch.profiler import profile, ProfilerActivity

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams

llm = LLMEngine(model="Qwen/Qwen2-0.5B-Instruct", enforce_eager=False)

prompts = ["Hello"] * 8
params = [SamplingParams(max_tokens=64, temperature=0.7) for _ in prompts]

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    llm.generate(prompts, params, use_tqdm=False)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=80))
if torch.cuda.is_available():
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))

prof.export_chrome_trace("engine_profile.json")
print("saved: engine_profile.json")
