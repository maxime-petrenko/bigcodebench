

from bigcodebench.provider.base import RefinementDecoderBase
import re
import os 
from tqdm import tqdm
from typing import List


from vllm import LLM, SamplingParams


class VllmDecoder(RefinementDecoderBase):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.model = LLM(model=model_name, 
                         max_gpu_memory_utilization=0.2,
                         tensor_parallel_size=4,
                         quantization="4bit",
                         max_seq_len=2048,
                         )
 

    

    def llm_call(self, messages, max_new_tokens=1024, **kwargs) -> str:
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )
        gen = self.model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        # gen is an iterator of Generation outputs; we take the first stream
        # and concatenate its .text chunks
        first = next(iter(gen))
        return "".join(chunk.text for chunk in first.chunks)
    

    def codegen(self, prompts, num_samples=1):
        results = []
        for p in tqdm(prompts):
            # if you want multiple samples, call llm_call repeatedly
            samples = [ self.llm_call([{"role":"user","content":p}]) 
                        for _ in range(num_samples) ]
            results.append(samples)
        return results

    
    # def codegen(self, prompts: List[str], do_sample: bool = True, num_samples: int = 1) -> List[str]:
       
    #    all_outputs = []
    #    for prompt in tqdm(prompts):
           

    #         ret = self.generate_one_completion(prompt)
    #         outputs = []
    #         for item in ret["choices"]:
    #             outputs.append(item["message"]["content"])
    #             #    outputs.append(item.message.content)
    #         all_outputs.append(outputs)

    #         print(f" \n -- nb of solved problems  {len(all_outputs)} \n results: {outputs} \n")

    #    return all_outputs
