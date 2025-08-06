

from bigcodebench.provider.base import RefinementDecoderBase

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from tqdm import tqdm
from typing import List



class AccDecoder(RefinementDecoderBase):


    def __init__(self, model_name):
        
        self.model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id,
        #     device_map="auto",               # spreads across GPUs
        #     torch_dtype=torch.float16,       # saves memory
        #     trust_remote_code=True
        # )
        # self.model.eval()



    def llm_call(self, input, max_new_tokens=1024, max_length=2048):

        if isinstance(input, list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in input):
            # Chat-style input
            flat_input = "\n\n".join([f"{m['role']}: {m['content']}" for m in input])
        elif isinstance(input, str):
            # Plain string input
            flat_input = input
        else:
            raise ValueError(f"Unsupported input type for llm_call: {type(input)}. Expected list of dicts or string.")


        inputs = self.tokenizer(flat_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024)

        # Remove prompt tokens from the output
        num_prompt_tokens = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][num_prompt_tokens:]

        output_str = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_str

    def codegen(self, prompts: List[str], do_sample: bool = True, num_samples: int = 1) -> List[str]:
        
        all_outputs = []
        # for prompt in tqdm(prompts):          
      
        #     output = self.llm_call(prompt)                
        #     all_outputs.append([output])
        #     print(f" \n -- nb of solved problems  {len(all_outputs)} \n results: {output} \n")

        return all_outputs


    # def codegen(self, prompts: List[str], do_sample: bool = True, num_samples: int = 1) -> List[str]:
       
    #    all_outputs = []
    #    for prompt in tqdm(prompts):
    #         print(f" --- IN CODEGEN ---- len  of prompts: {len(prompts)}")
    #         # ret = self.generate_one_completion(prompt)
    #         ret = self.llm_call(prompt)
    #         ret = {"choices": [{ "message": { "role": "assistant", "content": ret } }]}
    #         print(f" \n --- IN CODEGEN ---- ret: {ret} \n")
    #         outputs = []
    #         for item in ret["choices"]:
    #             outputs.append(item["message"]["content"])
    #             #    outputs.append(item.message.content)
    #         all_outputs.append(outputs)
    #         print(f" \n -- nb of solved problems  {len(all_outputs)} \n results: {outputs} \n")

    #    return all_outputs
  
  
    
  

