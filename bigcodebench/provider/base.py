from abc import ABC, abstractmethod
from typing import List
import re


from bigcodebench.provider.utility import EOS


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        subset: str,
        split: str,
        temperature: float = 0.8,
        max_new_tokens: int = 1280,
        revision: str = "main",
        dtype: str = "bfloat16",  # default
        direct_completion: bool = False,
        trust_remote_code: bool = False,
        tokenizer_name: str = None,
        tokenizer_legacy: bool = False,
        instruction_prefix: str = None,
        response_prefix: str = None,
        prefill: bool = True,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.subset = subset
        self.split = split
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.revision = revision
        self.direct_completion = direct_completion
        self.trust_remote_code = trust_remote_code
        self.tokenizer_name = tokenizer_name
        self.tokenizer_legacy = tokenizer_legacy
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix
        self.prefill = prefill

    @abstractmethod
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name




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

class RefinementDecoderBase():
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        
    @abstractmethod
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def llm_call(self, input, max_new_tokens=1024, max_length=2048):
        pass

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

    def generate_one_completion(self, task):        
        # Step 1: Initial solution from Agent1
        input_messages_main = [
            {"role": "system", "content": ref_prompts["gen_system_prompt"]},
            {"role": "user", "content": f" Code Problem: \n\n {task} \n\n Please provide a concise code solution."}
        ]
        llm_response = self.llm_call(input=input_messages_main)
        llm_code_solution = self.extract_python_code(llm_response)

        print(f"\n -- Initial solution from Agent1 \n INPUT:\n {input_messages_main}  \n ANSWER:\n {llm_code_solution}")


        max_rounds = 2
        round_num = 0
        current_solution = llm_code_solution

        while round_num < max_rounds:

            print(f"\n --  Round {round_num + 1} of refinement")
            # Step 2: Agent2 Verdict
            input_messages = [
                {"role": "system", "content": ref_prompts["agent2_verdict_prompt"]},
                {"role": "user", "content": f"Code Problem:\n {task} \n\n Proposed Solution: \n {current_solution}"}
            ]
            agent2_verdict_output = self.llm_call(input=input_messages, max_new_tokens=64)

            # Step 3: Consensus Check
            consensus_input = [
                {"role": "system", "content": ref_prompts["check_consensus_1"]},
                {"role": "user", "content": agent2_verdict_output}
            ]
            consensus_response = self.llm_call(input=consensus_input)
            decision = self.is_acceptable(consensus_response)
            if decision == "acceptable":
                break

            print(f"\n --Agent2 verdict : {consensus_response}")

            # Step 4: Feedback from Agent2
            feedback_input = [
                {"role": "system", "content": ref_prompts["agent2_feedback_prompt"]},
                {"role": "user", "content": f"Code Problem: \n {task} \n\n Proposed Solution: \n {current_solution}"}
            ]
            agent2_feedback_output = self.llm_call(input=feedback_input)

            print(f"\n --Agent2 feedback : \n INPUT:\n {feedback_input}  \n ANSWER:\n {agent2_feedback_output}")

            # Step 5: Agent1 Refines Solution
            refinement_input = [
                {"role": "system", "content": ref_prompts["system_prompt_agent1"]},
                {"role": "user", "content": f"""Code Problem: {task} \n\n Current Proposed Solution: \n\n {current_solution} \n\n Agent2's Feedback:  \n\n {agent2_feedback_output} \n\n Based on the above feedback, please provide an updated solution that improves clarity and correctness while maintaining simplicity. """}
            ]
            llm_response = self.llm_call(input=refinement_input)
            current_solution = self.extract_python_code(llm_response)
            round_num += 1
            print(f"\n --Refined input {refinement_input}")

        print(f"\n\n -- Final solution after {round_num} rounds of refinement")
        print(f"\n -- Solution: {current_solution}")

        # return current_solution
        return {"choices": [{ "message": { "role": "assistant", "content": current_solution } }]}


    # def __repr__(self) -> str:
    #     return self.name

    # def __str__(self) -> str:
    #     return self.name
        


    
    #     # Step 1: Initial solution from Agent1
    #     input_messages_main = [
    #         {"role": "system", "content": ref_prompts["gen_system_prompt"]},
    #         {"role": "user", "content": f" Code Problem: \n\n {task} \n\n Please provide a concise code solution."}
    #     ]
    #     llm_response = self.llm_call(input=input_messages_main)
    #     llm_code_solution = self.extract_python_code(llm_response)

    #     print(f"\n -- Initial solution from Agent1 \n INPUT:\n {input_messages_main}  \n ANSWER:\n {llm_code_solution}")


    #     max_rounds = 2
    #     round_num = 0
    #     current_solution = llm_code_solution

    #     while round_num < max_rounds:

    #         print(f"\n --  Round {round_num + 1} of refinement")
    #         # Step 2: Agent2 Verdict
    #         input_messages = [
    #             {"role": "system", "content": ref_prompts["agent2_verdict_prompt"]},
    #             {"role": "user", "content": f"Code Problem:\n {task} \n\n Proposed Solution: \n {current_solution}"}
    #         ]
    #         agent2_verdict_output = self.llm_call(input=input_messages, max_new_tokens=64)

    #         # Step 3: Consensus Check
    #         consensus_input = [
    #             {"role": "system", "content": ref_prompts["check_consensus_1"]},
    #             {"role": "user", "content": agent2_verdict_output}
    #         ]
    #         consensus_response = self.llm_call(input=consensus_input)
    #         decision = self.is_acceptable(consensus_response)


    #         if decision == "acceptable":
    #             break

    #         print(f"\n --Agent2 verdict : {consensus_response}")

    #         # Step 4: Feedback from Agent2
    #         feedback_input = [
    #             {"role": "system", "content": ref_prompts["agent2_feedback_prompt"]},
    #             {"role": "user", "content": f"Code Problem: \n {task} \n\n Proposed Solution: \n {current_solution}"}
    #         ]
    #         agent2_feedback_output = self.llm_call(input=feedback_input)

    #         print(f"\n --Agent2 feedback : \n INPUT:\n {feedback_input}  \n ANSWER:\n {agent2_feedback_output}")

    #         # Step 5: Agent1 Refines Solution
    #         refinement_input = [
    #             {"role": "system", "content": ref_prompts["system_prompt_agent1"]},
    #             {"role": "user", "content": f"""Code Problem: {task} \n\n Current Proposed Solution: \n\n {current_solution} \n\n Agent2's Feedback:  \n\n {agent2_feedback_output} \n\n Based on the above feedback, please provide an updated solution that improves clarity and correctness while maintaining simplicity. """}
    #         ]
    #         llm_response = self.llm_call(input=refinement_input)
    #         current_solution = self.extract_python_code(llm_response)
    #         round_num += 1
    #         print(f"\n --Refined input {refinement_input}")

    #     print(f"\n\n -- Final solution after {round_num} rounds of refinement")
    #     print(f"\n -- Solution: {current_solution}")

    #     # return current_solution
    #     return {"choices": [{ "message": { "role": "assistant", "content": current_solution } }]}

    def is_direct_completion(self) -> bool:
        """
        Check if the model is a direct completion model.
        This method should be overridden in subclasses.
        """
        return True



