import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from .engine_config import Config
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .seqeunce import Sequence
from utils.sampling_params import SamplingParams
from models.qwen2 import Qwen2



class LLMEngine:
  def __init__(self, model, **kwargs):
    # 提取出 Config 类真正定义了的参数。防止传入无效参数导致报错。
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k,v in kwargs.items() if k in config_fields}
    config = Config(model, **config_kwargs)
    
    # for tp
    self.ps = []      # 存储子进程对象的列表
    self.events = []  # 存储同步信号量（Event）的列表
    ctx = mp.get_context("spawn") # 获取多进程上下文
    for i in range(1, config.tensor_parallel_size):
      event = ctx.Event()
      process = ctx.Process(target=ModelRunner, args=(config, i, event))
      process.start()
      self.ps.append(process)
      self.events.append(event)
    self.model_runner = ModelRunner(config, 0, self.events)
    self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    config.eos = self.tokenizer.eos_token_id
    self.scheduler = Scheduler(config)
    atexit.register(self.exit)


    def exit(self):
      self.model_runner.call("exit")
      del self.model_runner
      for p in self.ps:
        p.join()

    def add_request(self, prompt: str|list[int], sampling_params: SamplingParams):
      if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt) # translate to token seqs
      seq = Sequence(prompt, sampling_params)
      self.Scheduler.add(seq)
    
    def step(self):
      pass

    def is_finished(self):
      pass

    def generate(self, prompts: list[str]| list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool=True):
      if use_tqdm:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
      if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len[prompts]
      for prompt, param in zip(prompts, sampling_params):
        self.add_request(prompt, param)
      outputs = {}
      prefill_throughput = decode_throughput = .0
      while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        if use_tqdm:
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
        for seq_id, token_ids in output:
          outputs[seq_id] = token_ids
          if use_tqdm:
            pbar.update(1)

      outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
      outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
      if use_tqdm:
          pbar.close()
      return outputs
