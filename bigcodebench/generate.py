import os
import json
import argparse
from typing import Optional, Tuple

# from bigcodebench.provider.refinement_unsloth import RefinementDecoder


from bigcodebench.provider.base import RefinementDecoderBase
# from bigcodebench.provider import make_model 

# from bigcodebench.provider import DecoderBase, make_model
from bigcodebench.data import get_bigcodebench, write_jsonl
from bigcodebench.sanitize import sanitize
from prompt_toolkit import prompt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)





from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm







def codegen(
    # model: DecoderBase,
    model: RefinementDecoderBase,
    target_path: str,
    split: str,
    subset: str,
    greedy: bool = False,
    strip_newlines: bool = False,
    n_samples: int = 1,
    id_range: Tuple[int, int] = None,
    resume: bool = True,
    batch_size: int = -1,
    tokenizer: Optional[AutoTokenizer] = None,
    local_model: Optional[AutoModelForCausalLM] = None,
):
    with Progress(
        TextColumn(f"BigCodeBench--{split.capitalize()} ({subset.capitalize()}) •" + "[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as p:
            
        dataset = get_bigcodebench(subset=subset)

        if model.is_direct_completion() and split == "instruct":
            raise Exception("Base model does not support direct completion for instruct tasks")
        
        # create target_path if it doesn't exist, e.g., a/b.jsonl

        print (f" -- Target path: {target_path}")
        dirname = os.path.dirname(target_path)
        if not os.path.exists(dirname) and dirname != "":
            os.makedirs(dirname)
            
        batch_prompts = []
        batch_task_ids = []
        batch_nsamples = []
        batch_entry_points = []
        
        # Read existing data once if resuming
        task2nexist = {}
        if resume and os.path.exists(target_path):
            with open(target_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    task2nexist[item["task_id"]] = task2nexist.get(item["task_id"], 0) + 1
        
        for id_num, (task_id, task) in enumerate(p.track(dataset.items())):
            if id_range is not None:
                low, high = id_range
                if id_num < low:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue
                if id_num >= id_range[1]:
                    break

            p_name = task_id.replace("/", "_")

            n_existing = task2nexist.get(task_id, 0)
            nsamples = n_samples - n_existing

            print(f"Processing task {id_num + 1}/{len(dataset)}: {task_id}  nsamples: {nsamples}, (existing: {n_existing})")
            
            try:
                prompt = task[f"{split}_prompt"]
            except:
                raise Exception(f"Invalid split {split} for bigcodebench-{subset}")
            if strip_newlines:
                prompt = prompt.strip("\n")
            
            if nsamples > 0:
                batch_prompts.append(prompt)
                batch_task_ids.append(task_id)
                batch_nsamples.append(nsamples)
                batch_entry_points.append(task["entry_point"])
                
                log = f"Codegen: {p_name} @ {model}"
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"
                p.console.print(log)
            
            if (batch_size and len(batch_prompts) == batch_size) or id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1):
                if not batch_prompts and (id_num == len(dataset) - 1 or (id_range and id_num == id_range[1] - 1)):
                    break



                # outputs = model.codegen(
                #     batch_prompts,
                #     do_sample=not greedy,
                #     num_samples=max(batch_nsamples),
                # )


                print(f" -- Running local model for {len(batch_prompts)} prompts")

                # prompt = batch_prompts


                outputs = []


                # for prompt in tqdm(batch_prompts):          
      
                #     output = llm_call(local_model, tokenizer, prompt)                
                #     outputs.append([output])
                #     print(f" \n -- nb of solved problems  {len(outputs)} \n results: {output} \n")



                #     # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048, padding_side = "left").to(local_model.device)
                #     # with torch.no_grad():
                #     #     outputs = local_model.generate(**inputs, max_new_tokens=1024)

                #     # print(tokenizer.decode(outputs[0], skip_special_tokens=True))


                # Process in batches instead of one by one
                for i in tqdm(range(0, len(batch_prompts), batch_size)):
                    batch = batch_prompts[i:i + batch_size]
                    
                    # Process the batch
                    batch_outputs = llm_call_batch(local_model, tokenizer, batch, batch_size=len(batch))

                    print(f" -- Codegen -- Batch {i // batch_size + 1} processed with {len(batch_outputs)} outputs")
                    
                    for output in batch_outputs:
                        outputs.append([output])  
                   





                assert outputs, "No outputs from model!"



                print(f"\n -- nb of solved problems {len(outputs)} \n")
                print( f" nsamples : {nsamples} \n")

                print(f" -- Batch {i // batch_size + 1} -- ")
                print(f"   len(batch_task_ids): {len(batch_task_ids)}")
                print(f"  batch_task_ids: {batch_task_ids}")
                print(f"   len(batch_prompts): {len(batch_prompts)}")
                print(f"   len(batch_entry_points): {len(batch_entry_points)}")
                print(f"   len(batch_nsamples): {len(batch_nsamples)}")
                print(f"   len(outputs): {len(outputs)}")
                
                samples = []
                for task_id, content, entry_point, nsamples, task_outputs in zip(batch_task_ids, batch_prompts, batch_entry_points, batch_nsamples, outputs):
                    if model.is_direct_completion():
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(content+completion, entry_point), raw_solution=content+completion)
                            for completion in task_outputs[:nsamples]
                        ])
                    else:
                        samples.extend([
                            dict(task_id=task_id, solution=sanitize(completion, entry_point), raw_solution=completion)
                            for completion in task_outputs[:nsamples]
                        ])

                print(f"Generated {len(samples)} samples")
                write_jsonl(target_path, samples, append=True)
            
                # Clear batches
                batch_prompts = []
                batch_task_ids = []
                batch_nsamples = []


def run_codegen(
    model: str,
    split: str,
    subset: str,
    root: str = "bcb_results",
    lora_path: str = None,
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 1280,
    # vllm
    max_model_len: int = 12800,
    greedy: bool = False,
    # openai
    reasoning_effort: str = "medium",
    # anthropic
    reasoning_budget: int = 0,
    reasoning_beta: str = "output-128k-2025-02-19",
    strip_newlines: bool = False,
    direct_completion: bool = False,
    resume: bool = True,
    id_range: str = None,
    backend: str = "vllm",
    base_url: str = None,
    tp: int = 1,
    instruction_prefix: str = "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
    response_prefix: str ="Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:",
    skip_prefill: bool = False,
    revision: str = "main",
    trust_remote_code: bool = False,
    tokenizer_name: str = None,
    tokenizer_legacy: bool = False,
):

    if greedy or (temperature == 0 and n_samples == 1):
        temperature = 0
        n_samples = 1
        greedy = True
        print("Greedy decoding ON (--greedy): setting n_samples=1, temperature=0")

    if id_range is not None:
        id_range = [int(i) for i in id_range.split("-")]
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    # Make project dir
    os.makedirs(root, exist_ok=True)
    
    # Make dir for codes generated by each model

    model_runner = make_model(model, backend)

    

    # if backend == "refinement":
    #     model_runner = RefinementDecoderBase(model_name=model)
    
    # model_runner = make_model(
    #     model=model,
    #     backend=backend,
    #     subset=subset,
    #     split=split,
    #     lora_path=lora_path,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    #     max_model_len=max_model_len,
    #     reasoning_effort=reasoning_effort,
    #     reasoning_budget=reasoning_budget,
    #     reasoning_beta=reasoning_beta,
    #     instruction_prefix=instruction_prefix,
    #     response_prefix=response_prefix,
    #     prefill=not skip_prefill,
    #     base_url=base_url,
    #     tp=tp,
    #     revision=revision,
    #     trust_remote_code=trust_remote_code,
    #     direct_completion=direct_completion,
    #     tokenizer_name=tokenizer_name,
    #     tokenizer_legacy=tokenizer_legacy
    # )
    
    extra = "-" + subset if subset != "full" else ""
    # if backend == "openai" and reasoning_effort and any(model.startswith(m) or model.endswith(m) for m in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]):
    #     model = model + f"--{reasoning_effort}"
    
    # if lora_path:
    #     model = model + f"--lora-{lora_path}"
    
    # if backend == "anthropic" and reasoning_budget and reasoning_beta:
    #     model = model + f"--{reasoning_budget}-{reasoning_beta}"
    
    if skip_prefill:
        identifier = model.replace("/", "--") + "--skip_prefill" + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    else:
        identifier = model.replace("/", "--") + f"--{revision}--bigcodebench{extra}-{split}--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    
    target_path = os.path.join(root, identifier)
    
    if not resume:
        os.remove(target_path)




    if torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    


    model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    local_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",               # spreads across GPUs
        torch_dtype=torch.float16,       # saves memory
        trust_remote_code=True
    )

    torch.distributed.barrier()

    local_model.eval()
    
    codegen(
        model=model_runner,
        target_path=target_path,
        split=split,
        subset=subset,
        greedy=greedy,
        strip_newlines=strip_newlines,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
        batch_size=bs,
        tokenizer=tokenizer,
        local_model=local_model
    )

    return target_path



def make_model(model: str, backend: str) -> RefinementDecoderBase:
    """
    Factory function to create a RefinementDecoder instance.
    This is a placeholder for the actual implementation.
    """
    # return RefinementDecoder()

    if backend == "refinement_vllm":
        from bigcodebench.provider.refinement_vllm import VllmDecoder
        return VllmDecoder(model)

    elif backend == "refinement_unsloth":
        from bigcodebench.provider.refinement_unsloth import UnslothDecoder
        return UnslothDecoder(model)
    
    elif backend == "refinement_acc":
        from bigcodebench.provider.refinement_acc import AccDecoder
        return AccDecoder(model)



def llm_call(local_model, local_tokenizer, input, max_new_tokens=1024, max_length=2048):

        if isinstance(input, list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in input):
            # Chat-style input
            flat_input = "\n\n".join([f"{m['role']}: {m['content']}" for m in input])
        elif isinstance(input, str):
            # Plain string input
            flat_input = input
        else:
            raise ValueError(f"Unsupported input type for llm_call: {type(input)}. Expected list of dicts or string.")


        inputs = local_tokenizer(flat_input, return_tensors="pt").to(local_model.device)
        with torch.no_grad():
            outputs = local_model.generate(**inputs, max_new_tokens=1024)

        # Remove prompt tokens from the output
        num_prompt_tokens = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][num_prompt_tokens:]

        output_str = local_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_str


# Method 3: Your original function modified for batching
def llm_call_batch(local_model, tokenizer, prompts, batch_size=4):
    """Modified version of your llm_call for batch processing"""
    all_outputs = []
    device = next(local_model.parameters()).device
    
    for i in range(0, len(prompts), batch_size):




        batch = prompts[i:i + batch_size]

        # assert inputs.input_ids.max() < tokenizer.vocab_size, "Token id exceeds vocab size"



        
        # Tokenize batch
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)


        # # Check for NaNs or out-of-bound values

        # assert not torch.isnan(inputs.input_ids).any(), "NaN found in input_ids"
        # assert inputs.input_ids.min() >= 0, "Negative token id found"
        # assert inputs.input_ids.max() < tokenizer.vocab_size, "Token id exceeds vocab size"
        # assert not torch.isnan(inputs.attention_mask).any()








        # Generate
        batch_decoded = []

        with torch.no_grad():
            try:
                outputs = local_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=2048,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                if "device-side assert triggered" in str(e):
                    print("⚠️ CUDA kernel failed. Skipping this sample.")
                    torch.cuda.empty_cache()
                    continue  # or `continue` if inside a loop
                else:
                    raise  # Re-raise unknown errors


        # # Generate
        # with torch.no_grad():
        #     outputs = local_model.generate(
        #         inputs.input_ids,
        #         attention_mask=inputs.attention_mask,
        #         max_length=2048,
        #         do_sample=True,
        #         top_p=0.9,
        #         temperature=0.7,
        #         pad_token_id=tokenizer.eos_token_id,
        #     )
        
        # Decode outputs
        batch_decoded = []
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            batch_decoded.append(decoded)
        
        all_outputs.extend(batch_decoded)
    
    return all_outputs


def main():
    from fire import Fire
    Fire(run_codegen)


if __name__ == "__main__":
    main()