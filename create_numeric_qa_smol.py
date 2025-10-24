from smolagents import (CodeAgent, 
                        OpenAIServerModel,
                        FinalAnswerTool,
                        tool
)
import json
import os
import time
import requests
import argparse
from typing import Dict, List, Optional
from smolagents import ChatMessage, Tool
from smolagents.models import ChatMessage
from tqdm import tqdm  # For progress tracking
from argo import ArgoModel, ARGO_MODELS
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
key = os.getenv("GEMINI_TEST_KEY")
api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"

MANAGED_MAX_STEPS=10
MANAGER_MAX_STEPS=30
def extract_tag_content(tag, text, occurrence=1):
    """Extract content between <tag> and </tag> for a specific occurrence.
    
    Args:
        tag: The tag name without angle brackets
        text: The text to search in
        occurrence: Which occurrence to extract (1-based indexing)
        
    Returns:
        The content between the nth occurrence of the tag, or None if not found
        For solution tags, attempts to extract numerical values from \boxed{} notation
    """
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    # Find the specified occurrence
    current_pos = 0
    for i in range(occurrence):
        start_pos = text.find(start_tag, current_pos)
        if start_pos == -1:
            return None  # Tag not found or fewer occurrences than requested
        current_pos = start_pos + 1
    
    # Extract content
    start_content = start_pos + len(start_tag)
    end_pos = text.find(end_tag, start_content)
    if end_pos == -1:
        return None  # Closing tag not found
    
    content = text[start_content:end_pos].strip()
    
    return content


class RateLimitedOAIServerModel(OpenAIServerModel):
    def __init__(self, *args, rate_limit=1.0/6.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit = 1.0 / rate_limit # seconds per request
        self.last_request_time = 0.0
def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        if self.rate_limit > 0:
            time_since_last_request = time.time() - self.last_request_time
            if time_since_last_request < self.rate_limit:
                time.sleep(self.rate_limit - time_since_last_request)
            self.last_request_time = time.time()
        response = self.client.chat.completions.create(**completion_kwargs)
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens

        message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        )
        message.raw = response
        if tools_to_call_from is not None:
            return parse_tool_args_if_needed(message)
        return message

def setup_model_interface(model_name):
    """
    Creates a multiagent system for generating and solving challenging scientific questions.
    
    Returns:
        function: A function that takes a scientific passage and returns the complete solution
    """
    maximum_steps = 35
    # Choose which model to use
    if model_name == 'gemini':
        model =  RateLimitedOAIServerModel(
            model_id="gemini-2.0-flash-thinking-exp-01-21",
            api_key=key,
            api_base=api_base,
            rate_limit= 1.0/6.0,
        )
    # Argo test
    elif 'argo' in model_name:

        argo_model = model_name.split(":")[-1]
        assert argo_model in ARGO_MODELS, f"{argo_model} not found in ARGO_MODELS; expected input format for argo is 'argo:<argo_model_name>'. Choose from {ARGO_MODELS}"

        model = ArgoModel(
            model_id = argo_model,
        )
    return model

def setup_question_generator(model, system_prompt=""):
    # Create specialized agents with detailed instructions
    question_generator = CodeAgent(
        tools=[FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="question_generator",
        description="""Generates an extremely challenging question with a well-defined numerical solution from scientific passages. 
        The question should be difficult enough to challenge a career researcher.
        Make sure the question:
        1. Is novel and extends beyond the direct content of the passage
        2. Requires sophisticated understanding of the scientific concepts
        3. Has a clear, computable numerical answer
        4. Most importantly: Would be challenging even for experts in the field
        5. Builds upon the concepts in the passage in non-trivial ways
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    question_generator.prompt_templates['system_prompt'] += system_prompt
    return question_generator
def setup_question_reviewer(model, system_prompt=""):
    question_review_agent = CodeAgent(
         tools = [FinalAnswerTool()],
            model = model,
            verbosity_level=1,
            name = "question_review_agent",
            description = """Reviews the question to ensure it meets the requirements.
            """,
            max_steps = MANAGED_MAX_STEPS,
    )
    question_review_agent.prompt_templates['system_prompt'] += system_prompt 
    return question_review_agent

def setup_solution_creation_agent(model, system_prompt=""):
    solution_agent = CodeAgent(
        tools=[FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="solution_agent",
        description="""Works through the logic and reasoning required to solve complex scientific questions.  
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    solution_agent.prompt_templates['system_prompt'] += system_prompt 
    return solution_agent

def setup_review_agent(model, system_prompt=""):
    review_agent = CodeAgent(
        tools=[FinalAnswerTool()],
        additional_authorized_imports=["numpy", "scipy", "sympy", "math", "matplotlib", "pandas"],
        model=model,
        verbosity_level=1,
        name="review_agent",
        description="""Reviews the solution to ensure they meet the requirements.
        You must verify calculations via python execution. 
        If any issues are found, clearly explain them and suggest improvements.
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    review_agent.prompt_templates['system_prompt'] += system_prompt 
    return review_agent
@tool
def qrs_answer_tool(query: str) -> str:
    """
    Convert an unformatted qrs triplet into a formatted output.
    Args:
        query (str): input string should include question, reasoning, and solution sections for extraction into formatted output
    Returns:
        model_request (str): the query that will be reformed.
    """

    return f"""Extract the following sections from the input string:
    <question> 'the question extracted from input' </question>
    <reasoning> 'the reasoning, denoted by <thinking>...</thinking> or equivalent tags' </reasoning>
    <solution> 'the final solution, denoted by <answer> ... </answer> or equivalent tags' </solution>
    Input string:
    {query}
    """
class NumericQAGenerator():
    def __init__(self, 
                    model_name: str,
                    ):

        self.model = setup_model_interface(model_name)
        question_generator = setup_question_generator(self.model)
        question_reviewer = setup_question_reviewer(self.model)
        solution_generator = setup_solution_creation_agent(self.model)
        review_agent = setup_review_agent(self.model)

        self.agent_system = CodeAgent(
                  tools=[FinalAnswerTool()],
        model=self.model,
        managed_agents=[question_generator, solution_generator, review_agent, question_reviewer],
        max_steps=MANAGER_MAX_STEPS,
        verbosity_level=1,
        additional_authorized_imports=["numpy", "scipy", "sympy", "math", "matplotlib", "pandas"],  
        )

    def __call__(self, query: str):
        prefix_prompt = """
    You are the manager of a complex system to create question-reasoning-solution triplets using a source passage as reference.  Your team consists of a question generation agent, "question_generator", a question review agent, "question_review_agent", a solution generation agent, "solution_agent", and a review agent, "reviw_agent".  
    WORKING WITH YOUR TEAM:
        question_generator: The question generator is tasked to create challenging questions based on the provided passage. It should be supplied the passage, and then accomplish the following tasks: 
            1. Requires at least 6 complex reasoning steps, logical relationships and factual recall to solve (where a reasoning step involves a separate logical inference, mathematical calculation, or application of a scientific principle)
            2. Has a single, unambiguous numerical answer (precise to at least 5 significant figures where appropriate)
            3. Is appropriate for a career researcher in this field
            4. Provides all necessary context within the problem statement itself (no direct reference to the excerpt)
            5. Presents a realistic scenario where this scientific knowledge would be applied
            6. Incorporates key concepts from the excerpt without directly stating formulas (students should know the relevant formulas)
            7. Requires deep understanding of the underlying scientific principles, not just plug-and-chug calculation
            8. If possible, include distractor information or extraneous details to make the question more challenging

            Avoid:
            - Problems solvable by plugging numbers into a given formula
            - Problems requiring numerical methods, such as numeric integration or differentiation.
            - Ambiguous wording or multiple possible interpretations
            - Mentioning the excerpt itself in the problem statement
            - Including formulas in the problem statement (these should be part of the expected knowledge)
            - Creating problems requiring knowledge outside the domain of the excerpt
            - Including Python code or providing the solution
            - Including steps required for solving the problem, or hints/tips for solving the problem
            - overt mention of third parties or subject: the authors, the passage, the disussion, the analysis, etc.
            The question_generator should only return the generated question, without other explanations or justifications.

        question_review_agent: 
            it is a critical evaluator of scientific questions. its task is to review the generated question to ensure it meets the requirements:
            1. CLARITY (1-10): Is the problem statement clearly written and unambiguous?
            - Clear context and setup: Are all needed variables defined?
            - Any extraneous or distracting information is intentionally included
            - Proper scientific terminology used correctly

            2. DIFFICULTY (1-10): How challenging is this for the target audience?
            - Score 1-3: Basic application of formulas, undergraduate level
            - Score 4-6: Multiple concepts, requires analysis, early graduate level  
            - Score 9-10: Complex integration of concepts, career researcher level
            - The target difficulty is 9-10 for this audience-The difficulty MUST be 9-10 for approval.

            3. SOLVABILITY (1-10):
            - Does the problem contain all necessary information to be solved?
            - Is there a single, unambiguous numerical answer?
            - Would a career researcher be able to solve this with the given information and enough time?

            4. DOMAIN RELEVANCE (1-10):
            - How well does the problem incorporate key concepts from the scientific excerpt?
            - Does it require deep understanding of the underlying principles?
            - Is it a realistic application of the scientific knowledge?
            - Do logical and reasoning steps used to generate the question follow from the excerpt?

            If any category is less than 9, the question_review_agent should provide feedback and ask the question_generator to try again.
            Your final recommendation should be one of the following:
            - Approve the question as is
            - Request minor revisions
            - Request major revisions

        solution_agent:
            Is a world-class solver of scientific problems with an encyclopedic knowledge of scientific principles and calculations.  Your job is to think step-by-step to solve a problem, then supply the actual numeric solution of the problem.  Your step-by-step reasoning should be offset by <thinking>...</thinking> tags.
            1. include logical proccesses and exploration of related ideas
            2. at least one instance of self-correction where the agent first attempts something incorrect and then self-corrects, e.g., in the next sentence.
            3. include a step-by-step breakdown of the problem statement before solution work begins
            4. if possible, include a self-correction in problem understanding as well
            5. include its final answer offset by <answer> ... </answer> tags
            6. In all steps, avoid third party mentions, e.g., the authors, the passage, etc.  Instead, include explicit references by restating important information directly.
        
        review_agent:
            The review agent is tasked with solving the reasoning of the solution agent in python code.  THE PYTHON CODE MUST EXECUTE AND MUST agree with the solution_agent's response to 4 significant figures.  In addition, the review agent must evaluate the Q-R-S triplet as follows:
                1. CORRECTNESS (1-10): Is the mathematical approach and final answer correct?
                - Are the appropriate scientific principles applied?
                - Do the code outputs agree with the reasoning and solutions provided?
                - Does reasoning correctly lead to the correct final answer?
                - Is the final answer numerically correct with appropriate precision?

                2. REASONING QUALITY (1-10):
                - Is each step logically justified and clearly explained?
                - Are all assumptions explicitly stated and reasonable?
                - Is the solution approach efficient and elegant?

                3. COMPLETENESS (1-10):
                - Does the solution address all aspects of the problem?
                - Are all variables properly defined and used?
                - Is the reasoning thorough with no missing steps?
            
            
            Based on the review, the review agent should take actions:
                a. If the solution is incorrect, provide feedback and ask the solution_agent to try again.

        CREATING A Q-R-S TRIPLET:
            1. First, instruct the question_generator to create a challenging question.
            2. Have the question_review_agent validate the created question, including for relevance compared to the original passage.
            3. If the question_review_agent recommends rewriting, send the passage, original geneerated question, and feedback to the question_generator for rewriting. Repeat until the question_review_agent is satisfied.
            4. Given the passage and finalized question, have the solution_agent write a detailed reasoning trace and answer. 
            5. Have the review_agent validate the reasoning and solution via python executed code. 
            6. If needed, send review_agent feedback (with original reasoning and solution) back to the solution_agent for rewrite.
            7. After review is passed, YOU MUST RETURN A DICTIONARY WITH THE FOLLOWING FIELDS:
            "question": [final question from the question generator]
            "reasoning": [final reasoning generated by the solution_agent]
            "solution": [the final single numerical response to the question]
        """

        input_query = f"{prefix_prompt}\nInput Passage: {query}"
        response = self.agent_system.run(input_query)
        return response
def process_cosmology_passages(input_file="../data/heuristic_filtered_cosmo_limited.jsonl", 
                               output_file="numeric-qa-cosmology-smol.jsonl"):
    """
    Reads scientific passages from a JSONL file, processes them through the multiagent system,
    and saves the generated question-reasoning-solution triplets to an output file.
    
    Args:
        input_file (str): Path to the input JSONL file containing scientific passages
        output_file (str): Path to the output JSONL file to save results
    """
    parser = argparse.ArgumentParser(description="Process scientific passages to generate challenging questions")
    parser.add_argument("--input_file", '-i', type=str, default=input_file, help="Path to the input JSONL file")
    parser.add_argument("--output_file", '-o', type=str, default=output_file, help="Path to the output JSONL file")
    parser.add_argument("--model_name", '-m', type=str, default='gemini', help="Name of the model to use")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    if '/' in output_file:
        # output file is in a subdirectory
        out_dir = output_file.split('/')[0]
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
    if size > 1:
        output_root = output_file.split('.')[0]
        output_file = f"{output_root}_rank{rank:02d}.jsonl"
    previous_comp_passages = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for l in f:
                previous_comp_passages.append(
                    json.loads(l)['passage'][:100]
                )
    # Set up our question solver
    scientific_question_solver = NumericQAGenerator(args.model_name)
    
    # Read passages from the input file
    passages = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Assuming each line has a 'passage' or 'text' field - adjust as needed
                    passage = data.get('passage', data.get('text', None))
                    if passage:
                        passages.append({
                            'original_data': data,
                            'passage': passage
                        })
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        print(f"Loaded {len(passages)} passages from {input_file}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Process each passage and save results
    results = []
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for i, passage_data in enumerate(tqdm(passages, desc="Processing passages")):
            passage = passage_data['passage']
            passage_id = passage_data['original_data'].get('id', f"passage_{i}")
            
            print(f"\nProcessing passage {i+1}/{len(passages)} (ID: {passage_id})")
            print(f"Passage: {passage[:150]}...")
            if passage[:100] in previous_comp_passages:
                print(f"[{rank}] Skipping previous passage {passage[:50]}...")
            if len(passage) < 200 \
                or "references" in passage[:100].lower() \
                or "acknowledgements" in passage[:100].lower():
                continue
            try:
                # Process the passage
                result = scientific_question_solver(passage)
                print(result)
                # pull out question, reasoning, and solution from the triplet
                try:
                    question = result['question']
                    reasoning = result['reasoning']
                    solution = result['solution']
                except:
                    question = extract_tag_content('question', json.dumps(result))
                    reasoning = extract_tag_content('reasoning', json.dumps(result))
                    solution = extract_tag_content('solution', json.dumps(result))
                # Save the result
                output_data = {
                    'passage_id': passage_id,
                    'passage': passage,
                    'question': question,
                    'reasoning': reasoning,
                    'solution': solution,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Write to file immediately (in case of later failures)
                out_file.write(json.dumps(output_data) + '\n')
                out_file.flush()  # Ensure writing to disk
                
                results.append(output_data)
                print(f"✓ Successfully processed passage {i+1}")
                

            except Exception as e:
                print(f"Error processing passage {i+1}: {e}")
                
                # Write error information to the output file
                error_data = {
                    'passage_id': passage_id,
                    'passage': passage,
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                out_file.write(json.dumps(error_data) + '\n')
                out_file.flush()
            # break # only doing one passage.
    print(f"\nProcessing complete. Generated {len(results)} QRS triplets.")
    print(f"Results saved to {output_file}")
    
    return results


def main():
    # Process the cosmology passages
    process_cosmology_passages()


if __name__ == "__main__":
    main()