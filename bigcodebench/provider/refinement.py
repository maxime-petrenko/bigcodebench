


from unsloth import FastLanguageModel
from transformers import TextStreamer
import re
import os 
from tqdm import tqdm
from typing import List

ref_prompts = {
    "gen_system_prompt": """ 
    You are an AI coding assistant generating solutions to coding problems. 
    You will receive a code problem, your goal is to generate the best solution that optimize the code simplicity and efficiency.

    You prefer code that:
    - Is simple and concise
    - Avoids overengineering or unnecessary abstraction
    - Uses fewer control structures and shallow nesting
    - Minimizes cyclomatic complexity

    Return the solution as plain code without any extra commentary.""",
    "system_prompt_agent1": """ You are an AI coding assistant generating solutions to coding problems. 
    You will receive a code problem, a proposed solution, and evaluation feedback.

    First of all, check if the verdict of feed back Acceptable. If yes, then just return the proposed solution.
    But if verdict says needs improvement, then update the code solution to address the issues raised while keeping align with you preference.

    You prefer code that:
    - Is simple and concise
    - Avoids overengineering or unnecessary abstraction
    - Uses fewer control structures and shallow nesting
    - Minimizes cyclomatic complexity

    Return the updated solution as plain code without any extra commentary. """,
    "agent2_verdict_prompt": """ 
    You are a senior code reviewer with a focus on clarity and correctness. 
    You will receive a code problem and a proposed solution. 
    Evaluate the solution based on its clarity and correctness. 
    Return your verdict : "Acceptable" or "Needs Improvement" 

    Do not include any extra commentary. """,
    "agent2_feedback_prompt": """ 
    You are a senior code reviewer with a focus on clarity and correctness. You are a senior code reviewer with a focus on clarity and correctness.
    You will receive a code problem, a proposed solution.
    Evaluate the solution and return your suggested feedback that would improve the clarity and correctness of the code.""",
    "check_consensus_1": """
    Is the verdict Acceptable ?
    Simply answer in one word : YES or NO """,

# gen_system_prompt -- Agent1: generation system prompt and preference(minimizing complexity)
# system_prompt_agent1 -- Agent1: Focus on minimizing complexity.
# agent2_verdict_prompt -- Agent2: reviewing with focus on clarity and correctness.

}


class RefinementDecoder():
    def __init__(self, model_path=None, model=None, tokenizer=None):
        default_model_path = "../../../models/checkpoints/lora_adapters_complexity"
        model_path = model_path or os.environ.get("REFINEMENT_MODEL_PATH", default_model_path)

        if model is None or tokenizer is None:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=True,
                device_map={"": 0},
            )
        else:
            self.model = model
            self.tokenizer = tokenizer



    def llm_call(self, input):
        # only llm output is returned

        FastLanguageModel.for_inference(self.model)
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        flat_input = "\n\n".join([f"{m['role']}: {m['content']}" for m in input])
        inputs = self.tokenizer(flat_input, return_tensors="pt").to("cuda")

        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            temperature=0.4,
            top_k=50,
            top_p=0.95,
            use_cache=False,
            streamer=text_streamer
        )

        input_len = inputs['input_ids'].shape[1]
        generated_tokens = output_tokens[:, input_len:]
        output_str = self.streamertokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return output_str

    def llm_call_short(self, input):
        FastLanguageModel.for_inference(self.model)
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        flat_input = "\n\n".join([f"{m['role']}: {m['content']}" for m in input])
        inputs = self.tokenizer(flat_input, return_tensors="pt").to("cuda")

        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=50,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            temperature=0.4,
            top_k=50,
            top_p=0.95,
            use_cache=False,
            streamer=text_streamer,
            past_key_values=None
        )

        input_len = inputs['input_ids'].shape[1]
        generated_tokens = output_tokens[:, input_len:]
        output_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return output_str

    def is_acceptable(self, llm_in_concensus: str) -> str:
        # Extract the first 10 words as a list (all in lowercase)
        words = llm_in_concensus.lower().split()[:10]
        decision = None

        # Iterate over the words with index to check sequentially
        for i, word in enumerate(words):
            # Check single-word acceptable keywords
            if word in ["yes", "acceptable", "approved", "ok"]:
                decision = "acceptable"
                break
            # Check for multi-word phrase "need improvement" or "need improvements"
            if word in ["need", "needs"] and i+1 < len(words) and words[i+1] in ["improvement", "improvements"]:
                decision = "needs improvement"
                break
        return decision

    def extract_python_code(self, text: str) -> str:
        # Recherche un bloc de code entouré par ```python ... ```
        match = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def format_messages(self, messages):
        return "\n\n".join([f"- {msg['role'].capitalize()}: {msg['content']}" for msg in messages])




    def codegen(self, prompts: List[str], do_sample: bool = True, num_samples: int = 200) -> List[str]:
       
       all_outputs = []
       for prompt in tqdm(prompts):
           

           ret = self.generate_one_completion(prompt)
           outputs = []
           for item in ret.choices:
               outputs.append(item.message.content)
           all_outputs.append(outputs)
       return all_outputs
    

    def generate_one_completion(self, task):        
        # Step 1: Initial solution from Agent1
        input_messages_main = [
            {"role": "system", "content": ref_prompts["gen_system_prompt"]},
            {"role": "user", "content": f" Code Problem: \n\n {task} \n\n Please provide a concise code solution."}
        ]
        llm_response = self.llm_call(input=input_messages_main)
        llm_code_solution = self.extract_python_code(llm_response)


        max_rounds = 2
        round_num = 0
        current_solution = llm_code_solution

        while round_num < max_rounds:
            # Step 2: Agent2 Verdict
            input_messages = [
                {"role": "system", "content": ref_prompts["agent2_verdict_prompt"]},
                {"role": "user", "content": f"Code Problem:\n {task} \n\n Proposed Solution: \n {current_solution}"}
            ]
            agent2_verdict_output = self.llm_call_short(input=input_messages)

            # Step 3: Consensus Check
            consensus_input = [
                {"role": "system", "content": ref_prompts["check_consensus_1"]},
                {"role": "user", "content": agent2_verdict_output}
            ]
            consensus_response = self.llm_call(input=consensus_input)
            decision = self.is_acceptable(consensus_response)


            if decision == "acceptable":
                break

            # Step 4: Feedback from Agent2
            feedback_input = [
                {"role": "system", "content": ref_prompts["agent2_feedback_prompt"]},
                {"role": "user", "content": f"Code Problem: \n {task} \n\n Proposed Solution: \n {current_solution}"}
            ]
            agent2_feedback_output = self.llm_call(input=feedback_input)

            # Step 5: Agent1 Refines Solution
            refinement_input = [
                {"role": "system", "content": ref_prompts["refinement_prompt_agent1"]},
                {"role": "user", "content": f"""Code Problem: {task} \n\n Current Proposed Solution: \n\n {current_solution} \n\n Agent2's Feedback:  \n\n {agent2_feedback_output} \n\n Based on the above feedback, please provide an updated solution that improves clarity and correctness while maintaining simplicity. """}
            ]
            llm_response = self.llm_call(input=refinement_input)
            current_solution = self.extract_python_code(llm_response)
            round_num += 1

        return current_solution
    

    # outputs = model.codegen(
    #                     batch_prompts,
    #                     do_sample=not greedy,
    #                     num_samples=max(batch_nsamples),
    #                 )

    # def codegen(
    #         self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    #     ) -> List[str]:






   














# # Agent2: reviewing with focus on clarity and correctness.
# eval_system_prompt = """
#  You are a senior code reviewer with a focus on clarity and correctness. 
#  You will receive a code problem, a proposed solution, and evaluation feedback.
#  Evaluate the solution based on its clarity and correctness.
#  Return your review in the exact JSON format below:
#  { "verdict": "Acceptable" or "Needs Improvement","rationale": "A short paragraph explaining your verdict based on clarity and correctness","suggested_feedback": "Specific feedback or 'None'"}

#  Do not include any extra commentary.
# """










