#!/usr/bin/env python3
"""
Verifiable Code Generation System using Smolagents
Generates scientific computing problems with multi-language solutions and automated verification
"""

from smolagents import (
    CodeAgent, 
    OpenAIServerModel,
    FinalAnswerTool,
    tool,
    ChatMessage,
    Tool
)
import json
import os
import time
import subprocess
import tempfile
import argparse
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from argo import ArgoModel, ARGO_MODELS
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuration
MANAGED_MAX_STEPS = 10  # For individual agents
MANAGER_MAX_STEPS = 20  # For orchestrator
MAX_COMPILATION_ATTEMPTS = 5
MAX_TEST_ATTEMPTS = 5
SUPPORTED_LANGUAGES = ["python", "cpp", "fortran"]

# Model setup - argo.py handles ARGO_USER environment variable

# Problem domains and templates for variety
PROBLEM_DOMAINS = [
    {
        "domain": "physics",
        "topics": ["orbital mechanics", "thermodynamics", "wave propagation", "quantum mechanics"],
        "difficulty_range": (5, 10),
        "preferred_languages": ["python", "fortran", "cpp"]
    },
    {
        "domain": "mathematics", 
        "topics": ["number theory", "linear algebra", "numerical analysis", "optimization"],
        "difficulty_range": (6, 10),
        "preferred_languages": ["python", "cpp"]
    },
    {
        "domain": "algorithms",
        "topics": ["graph theory", "dynamic programming", "computational geometry", "sorting"],
        "difficulty_range": (7, 10),
        "preferred_languages": ["cpp", "python"]
    },
    {
        "domain": "data_science",
        "topics": ["time series", "statistics", "machine learning", "signal processing"],
        "difficulty_range": (5, 9),
        "preferred_languages": ["python"]
    }
]

@tool
def compile_code(code: str, language: str, debug: bool = False) -> Dict[str, Any]:
    """
    Compiles code and returns compilation status and errors
    
    Args:
        code: Source code to compile
        language: Programming language (python, cpp, fortran)
        debug: Whether to save debug files
        
    Returns:
        Dictionary with 'success', 'output', 'error', and optional 'executable_path'
    """
    result = {"success": False, "output": "", "error": "", "executable_path": None}
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix=_get_file_extension(language), delete=False) as f:
            f.write(code)
            f.flush()
            source_file = f.name
            
        if language == "python":
            # Python doesn't need compilation, just syntax check
            compile_result = subprocess.run(
                ["python", "-m", "py_compile", source_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            result["success"] = compile_result.returncode == 0
            result["error"] = compile_result.stderr if compile_result.returncode != 0 else ""
            result["executable_path"] = source_file
            
        elif language == "cpp":
            exe_name = source_file.replace('.cpp', '')
            compile_result = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-o", exe_name, source_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            result["success"] = compile_result.returncode == 0
            result["output"] = compile_result.stdout
            result["error"] = compile_result.stderr
            result["executable_path"] = exe_name if result["success"] else None
            
        elif language == "fortran":
            exe_name = source_file.replace('.f90', '')
            # Try modern Fortran compilation
            compile_result = subprocess.run(
                ["gfortran", "-std=f95", "-ffree-form", "-O2", "-o", exe_name, source_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                # Fallback to less strict compilation
                compile_result = subprocess.run(
                    ["gfortran", "-ffree-form", "-O2", "-o", exe_name, source_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            
            result["success"] = compile_result.returncode == 0
            result["output"] = compile_result.stdout
            result["error"] = compile_result.stderr
            result["executable_path"] = exe_name if result["success"] else None
            
        if debug and not result["success"]:
            debug_file = f"/tmp/debug_{language}_{time.time()}.{_get_file_extension(language)}"
            with open(debug_file, 'w') as df:
                df.write(code)
            result["debug_file"] = debug_file
            
        # Cleanup source file if not debugging
        if not debug:
            try:
                os.unlink(source_file)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        result["error"] = "Compilation timeout exceeded"
    except Exception as e:
        result["error"] = str(e)
        
    return result

@tool
def run_code_with_input(executable_path: str, test_input: str, language: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Runs compiled code with given input
    
    Args:
        executable_path: Path to executable or script
        test_input: Input to provide to the program
        language: Programming language
        timeout: Execution timeout in seconds
        
    Returns:
        Dictionary with 'success', 'output', 'error', 'execution_time_ms'
    """
    result = {"success": False, "output": "", "error": "", "execution_time_ms": 0}
    
    try:
        start_time = time.time()
        
        if language == "python":
            # Create modified script with mocked input
            with open(executable_path, 'r') as f:
                original_code = f.read()
            
            mock_input = f"""
import sys
from io import StringIO
import builtins

sys.stdin = StringIO('''{test_input}''')
_test_inputs = iter('''{test_input}'''.strip().split('\\n'))
def _mock_input(prompt=''):
    try:
        return next(_test_inputs)
    except StopIteration:
        return ''
builtins.input = _mock_input

"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(mock_input + original_code)
                f.flush()
                temp_script = f.name
            
            run_result = subprocess.run(
                ["python", temp_script],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            os.unlink(temp_script)
            
        else:
            # For compiled languages, provide input via stdin
            run_result = subprocess.run(
                [executable_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
        execution_time = (time.time() - start_time) * 1000
        
        result["success"] = run_result.returncode == 0
        result["output"] = run_result.stdout.strip()
        result["error"] = run_result.stderr.strip() if run_result.returncode != 0 else ""
        result["execution_time_ms"] = execution_time
        
    except subprocess.TimeoutExpired:
        result["error"] = f"Execution timeout exceeded ({timeout}s)"
        result["execution_time_ms"] = timeout * 1000
    except Exception as e:
        result["error"] = str(e)
        
    return result

@tool
def verify_output_match(expected: str, actual: str, tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Verifies if actual output matches expected output
    
    Args:
        expected: Expected output
        actual: Actual output from program
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        Dictionary with 'match' boolean and 'details' string
    """
    # Exact match
    if expected.strip() == actual.strip():
        return {"match": True, "details": "Exact match"}
    
    # Try numerical comparison
    try:
        expected_nums = [float(x) for x in expected.split()]
        actual_nums = [float(x) for x in actual.split()]
        
        if len(expected_nums) != len(actual_nums):
            return {"match": False, "details": f"Different number count: expected {len(expected_nums)}, got {len(actual_nums)}"}
        
        for i, (e, a) in enumerate(zip(expected_nums, actual_nums)):
            rel_error = abs(e - a) / max(abs(e), abs(a), 1)
            if rel_error > tolerance:
                return {"match": False, "details": f"Numerical mismatch at position {i}: expected {e}, got {a} (error: {rel_error})"}
        
        return {"match": True, "details": "Numerical match within tolerance"}
        
    except (ValueError, IndexError):
        # Not numerical, try string comparison
        if expected.strip().lower() == actual.strip().lower():
            return {"match": True, "details": "Case-insensitive match"}
        else:
            return {"match": False, "details": f"String mismatch: expected '{expected}', got '{actual}'"}

def _get_file_extension(language: str) -> str:
    """Get file extension for language"""
    return {
        "python": ".py",
        "cpp": ".cpp", 
        "fortran": ".f90"
    }.get(language, ".txt")

def setup_model_interface(model_name: str):
    """Set up model interface using argo models"""
    # Handle argo:model_name format or direct model name
    if 'argo:' in model_name:
        argo_model = model_name.split(":")[-1]
    else:
        argo_model = model_name
    
    # Validate model exists
    assert argo_model in ARGO_MODELS, f"{argo_model} not found in ARGO_MODELS. Choose from {list(ARGO_MODELS.keys())}"
    
    # Use ArgoModel as designed - it handles ARGO_USER internally
    model = ArgoModel(model_id=argo_model)
    return model

def setup_prompt_writer_agent(model, system_prompt=""):
    """Creates agent for generating problem statements"""
    agent = CodeAgent(
        tools=[FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="prompt_writer",
        description="""Generates challenging scientific computing problems.
        Creates problems that:
        1. Are based on real scientific concepts
        2. Have clear numerical or verifiable outputs
        3. Can be solved algorithmically in under 100 lines
        4. Test deep understanding of concepts
        5. Are appropriate for implementation in multiple programming languages
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    agent.prompt_templates['system_prompt'] += system_prompt
    return agent

def setup_code_solution_agent(model, system_prompt=""):
    """Creates agent for writing code solutions"""
    agent = CodeAgent(
        tools=[FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="code_solver",
        description="""Writes complete, working code solutions in specified programming languages.
        Expert in Python, C++, and Fortran for scientific computing.
        Ensures code is:
        1. Complete and runnable
        2. Efficient and well-structured
        3. Under 100 lines
        4. Free of infinite loops
        5. Properly handles input/output
        """,
        max_steps=MANAGED_MAX_STEPS,
        additional_authorized_imports=["numpy", "scipy", "math", "itertools", "collections"]
    )
    agent.prompt_templates['system_prompt'] += system_prompt
    return agent

def setup_compilation_checker_agent(model, system_prompt=""):
    """Creates agent for checking compilation and fixing errors"""
    agent = CodeAgent(
        tools=[compile_code, FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="compilation_checker",
        description="""Verifies code compiles correctly and fixes compilation errors.
        Expert in debugging:
        1. Syntax errors
        2. Type mismatches
        3. Missing headers/imports
        4. Language-specific compilation issues
        Can suggest fixes for compilation failures.
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    agent.prompt_templates['system_prompt'] += system_prompt
    return agent

def setup_test_writer_agent(model, system_prompt=""):
    """Creates agent for generating test cases"""
    agent = CodeAgent(
        tools=[FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="test_writer",
        description="""Generates comprehensive test cases for code solutions.
        Creates tests that:
        1. Cover edge cases and typical cases
        2. Test correctness thoroughly
        3. Include simple and complex inputs
        4. Have verifiable outputs
        5. Consider numerical precision issues
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    agent.prompt_templates['system_prompt'] += system_prompt
    return agent

def setup_test_runner_agent(model, system_prompt=""):
    """Creates agent for running tests and verifying results"""
    agent = CodeAgent(
        tools=[run_code_with_input, verify_output_match, FinalAnswerTool()],
        model=model,
        verbosity_level=1,
        name="test_runner",
        description="""Executes code with test inputs and verifies outputs.
        Handles:
        1. Running compiled/interpreted code
        2. Providing test inputs
        3. Capturing outputs
        4. Verifying correctness
        5. Reporting test results
        """,
        max_steps=MANAGED_MAX_STEPS,
    )
    agent.prompt_templates['system_prompt'] += system_prompt
    return agent

class VerifiableCodeGenerator:
    """Main orchestrator for verifiable code generation system"""
    
    def __init__(self, model_name: str):
        self.model = setup_model_interface(model_name)
        
        # Create specialized agents
        self.prompt_writer = setup_prompt_writer_agent(self.model)
        self.code_solver = setup_code_solution_agent(self.model)
        self.compilation_checker = setup_compilation_checker_agent(self.model)
        self.test_writer = setup_test_writer_agent(self.model)
        self.test_runner = setup_test_runner_agent(self.model)
        
        # Create orchestrator agent
        self.orchestrator = CodeAgent(
            tools=[FinalAnswerTool()],
            model=self.model,
            managed_agents=[
                self.prompt_writer,
                self.code_solver,
                self.compilation_checker,
                self.test_writer,
                self.test_runner
            ],
            max_steps=MANAGER_MAX_STEPS,
            verbosity_level=1,
            additional_authorized_imports=["numpy", "scipy", "math", "subprocess", "tempfile"]
        )
    
    def generate_from_domain(self, domain_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate problem and solutions from domain specification"""
        
        orchestration_prompt = f"""
You are the orchestrator managing a team to generate verifiable code problems and solutions.

TEAM:
- prompt_writer: Creates problem statements
- code_solver: Writes code solutions  
- compilation_checker: Checks compilation
- test_writer: Creates test cases
- test_runner: Runs tests

TASK: Generate 1 problem in {domain_spec['domain']} domain with Python solution.

WORKFLOW:
1. Ask prompt_writer to create a {domain_spec['domain']} problem (topic: {domain_spec['topics'][0]})
2. Ask code_solver to write Python code for the problem
3. Ask compilation_checker to verify the code compiles
4. Ask test_writer to create 3 test cases
5. Ask test_runner to execute the tests
6. Return results in JSON format

KEEP IT SIMPLE: Each step should be a clear delegation to one agent. Don't do the work yourself - delegate it.

Return format:
{{
  "problem": {{"statement": "...", "domain": "{domain_spec['domain']}" }},
  "solutions": [{{"language": "python", "code": "...", "tests_passed": 0, "tests_total": 0}}],
  "metadata": {{"successful_languages": []}}
}}
"""
        
        result = self.orchestrator.run(orchestration_prompt)
        return result
    
    def generate_from_paper(self, paper_text: str) -> Dict[str, Any]:
        """Generate problem and solutions from scientific paper text"""
        
        # Detect suitable languages based on paper content
        suitable_languages = self._detect_suitable_languages(paper_text)
        
        orchestration_prompt = f"""
You are the orchestrator of a verifiable code generation system. Your team consists of:
1. prompt_writer: Generates problem statements
2. code_solver: Writes code solutions  
3. compilation_checker: Verifies compilation and fixes errors
4. test_writer: Creates test cases
5. test_runner: Executes tests and verifies outputs

WORKFLOW WITH FEEDBACK LOOPS:

Phase 1 - Problem Generation from Paper:
- Use prompt_writer to create a problem inspired by this scientific paper excerpt:
  {paper_text}
- The problem should test understanding of concepts from the paper
- Should be computationally solvable with verifiable output

Phase 2 - Multi-Language Solution Generation:
For each language in {suitable_languages}:

  Step 1: Code Generation
  - Have code_solver write a solution in the target language
  - Solution must properly implement the scientific concepts
  
  Step 2: Compilation Check Loop (max {MAX_COMPILATION_ATTEMPTS} attempts)
  - Use compilation_checker to verify compilation
  - If fails: gather errors, feedback to code_solver, retry
  
  Step 3: Test Generation
  - Use test_writer for comprehensive test cases
  - Include tests that verify scientific correctness
  
  Step 4: Test Execution Loop (max {MAX_TEST_ATTEMPTS} attempts)  
  - Use test_runner to execute tests
  - If fails: diagnose issue, feedback to appropriate agent, retry

Phase 3 - Result Compilation:
Return comprehensive results in dictionary format

FEEDBACK LOOP DETAILS:
- Compilation errors → specific fixes for syntax, types, headers
- Wrong output → verify algorithm implementation
- Timeout → check for infinite loops, add iteration limits
- Numerical errors → adjust precision tolerance

RETURN FORMAT:
{{
  "problem": {{"statement": "...", "concepts": [...], "source": "paper"}},
  "solutions": [...],
  "metadata": {{...}}
}}
"""
        
        result = self.orchestrator.run(orchestration_prompt)
        return result
    
    def _detect_suitable_languages(self, paper_text: str) -> List[str]:
        """Detect suitable programming languages based on paper content"""
        text_lower = paper_text.lower()
        
        scores = {
            "python": sum(1 for term in ["python", "machine learning", "data analysis", "statistics", "pandas", "numpy"] 
                         if term in text_lower),
            "cpp": sum(1 for term in ["c++", "algorithm", "performance", "optimization", "parallel", "graph"]
                      if term in text_lower),
            "fortran": sum(1 for term in ["fortran", "numerical", "simulation", "finite", "monte carlo", "matrix"]
                          if term in text_lower)
        }
        
        # Always include Python
        languages = ["python"]
        
        # Add others if they score well
        if scores["cpp"] >= 2:
            languages.append("cpp")
        if scores["fortran"] >= 2:
            languages.append("fortran")
            
        # If only Python selected, add at least one more for diversity
        if len(languages) == 1:
            languages.append("cpp")
            
        return languages

def process_dataset(input_file: str, output_file: str, model_name: str, max_samples: int = 10):
    """Process dataset to generate verifiable code problems"""
    
    generator = VerifiableCodeGenerator(model_name)
    
    # Load input data if provided
    papers = []
    if input_file and os.path.exists(input_file):
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', data.get('passage', ''))
                    if text and len(text) > 500:
                        papers.append(text)
                except:
                    continue
        print(f"Loaded {len(papers)} papers from {input_file}")
    
    # Generate samples
    results = []
    with open(output_file, 'a') as out_f:
        for i in range(max_samples):
            print(f"\n[Rank {rank}] Generating sample {i+1}/{max_samples}")
            
            try:
                # Decide whether to use paper or domain template
                if papers and random.random() < 0.3:
                    # Generate from paper
                    paper = random.choice(papers)
                    result = generator.generate_from_paper(paper)
                else:
                    # Generate from domain
                    domain_spec = random.choice(PROBLEM_DOMAINS)
                    result = generator.generate_from_domain(domain_spec)
                
                # Add metadata
                result['sample_id'] = f"rank{rank}_sample{i}"
                result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Save immediately
                out_f.write(json.dumps(result) + '\n')
                out_f.flush()
                
                results.append(result)
                
                # Print summary
                successful = result['metadata'].get('successful_languages', [])
                print(f"  ✅ Generated sample with {len(successful)} successful languages: {successful}")
                
            except Exception as e:
                print(f"  ❌ Error generating sample {i+1}: {e}")
                error_data = {
                    'sample_id': f"rank{rank}_sample{i}",
                    'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                out_f.write(json.dumps(error_data) + '\n')
                out_f.flush()
    
    print(f"\nGenerated {len(results)} samples, saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate verifiable code dataset using agents")
    parser.add_argument("--input-file", "-i", type=str, default=None, 
                       help="Input JSONL file with papers (optional)")
    parser.add_argument("--output-file", "-o", type=str, 
                       default="verifiable_code_agents.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--model", "-m", type=str, default="gemini",
                       help="Model to use (gemini or argo:model_name)")
    parser.add_argument("--max-samples", "-n", type=int, default=10,
                       help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Adjust output file for MPI
    output_file = args.output_file
    if size > 1:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_rank{rank:02d}{ext}"
    
    print(f"Verifiable Code Generation with Agents")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Output: {output_file}")
    print("-" * 50)
    
    process_dataset(
        args.input_file,
        output_file,
        args.model,
        args.max_samples
    )

if __name__ == "__main__":
    main()