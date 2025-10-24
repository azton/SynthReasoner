#!/usr/bin/env python3
"""
Verifiable Code Reasoning Dataset Generator

Generates scientific coding problems with executable solutions and test cases.
Creates problems similar to Project Euler but focused on scientific computing.

Usage examples:
    # Generate 10 samples for testing
    python llm_verifiable_code.py --max-samples 10
    
    # Use different model and output file
    python llm_verifiable_code.py --model gpt4o --output code_problems.json --max-samples 20
    
    # Use papers as seed instead of generating problems
    python llm_verifiable_code.py --input-file papers.jsonl --max-samples 15
"""

import asyncio
import random
import argparse
import json
import time
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# MPI support
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MPI_AVAILABLE = True
except ImportError:
    rank = 0
    size = 1
    MPI_AVAILABLE = False

try:
    from .llm_interface import UnifiedLLMClient
except ImportError:
    from llm_interface import UnifiedLLMClient

class LLMInterface(ABC):
    """Abstract interface for LLM clients"""
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        pass

class ArgoLLMAdapter(LLMInterface):
    """Adapter to use UnifiedLLMClient with Argo server"""
    
    def __init__(self, model_name: str = "claudesonnet4", interface: str = "auto", **kwargs):
        self.client = UnifiedLLMClient(model_name=model_name, interface=interface, **kwargs)
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.client.generate(messages, temperature=temperature)

@dataclass
class CodeProblem:
    """Represents a coding problem"""
    problem_statement: str
    domain: str  # 'physics', 'math', 'algorithms', 'data_science', etc.
    difficulty: str  # 'easy', 'medium', 'hard', 'expert'
    concepts: List[str]  # Key concepts involved
    constraints: Dict[str, Any]  # Input constraints
    
@dataclass
class CodeSolution:
    """Represents a solution with tests"""
    code: str
    language: str  # 'python', 'cpp', 'fortran'
    test_cases: List[Dict[str, Any]]  # Input-output pairs
    execution_time_ms: float
    memory_usage_mb: float
    passed_tests: bool
    error_message: Optional[str]

class ScientificProblemGenerator:
    """Generates scientific coding problems"""
    
    PROBLEM_TEMPLATES = [
        # Physics problems - good for all languages
        {
            "domain": "physics",
            "template": "orbital_mechanics",
            "concepts": ["Newton's laws", "numerical integration", "orbital dynamics"],
            "languages": ["python", "cpp", "fortran"]
        },
        {
            "domain": "physics", 
            "template": "wave_propagation",
            "concepts": ["wave equation", "Fourier analysis", "signal processing"],
            "languages": ["python", "cpp", "fortran"]
        },
        {
            "domain": "physics",
            "template": "statistical_mechanics", 
            "concepts": ["Monte Carlo", "thermodynamics", "phase transitions"],
            "languages": ["python", "cpp", "fortran"]
        },
        {
            "domain": "physics",
            "template": "fluid_dynamics",
            "concepts": ["Navier-Stokes", "finite difference", "CFD"],
            "languages": ["cpp", "fortran"]
        },
        {
            "domain": "physics",
            "template": "quantum_mechanics",
            "concepts": ["Schrödinger equation", "eigenvalue problems", "linear algebra"],
            "languages": ["cpp", "fortran"]
        },
        # Mathematics problems
        {
            "domain": "mathematics",
            "template": "number_theory",
            "concepts": ["prime numbers", "modular arithmetic", "factorization"],
            "languages": ["python", "cpp"]
        },
        {
            "domain": "mathematics",
            "template": "combinatorics",
            "concepts": ["permutations", "combinations", "generating functions"],
            "languages": ["python", "cpp"]
        },
        {
            "domain": "mathematics",
            "template": "numerical_analysis",
            "concepts": ["root finding", "interpolation", "numerical derivatives"],
            "languages": ["python", "cpp", "fortran"]
        },
        {
            "domain": "mathematics",
            "template": "linear_algebra",
            "concepts": ["matrix operations", "eigenvalues", "decompositions"],
            "languages": ["cpp", "fortran"]
        },
        # Algorithm problems - better for C++
        {
            "domain": "algorithms",
            "template": "graph_theory",
            "concepts": ["shortest path", "network flow", "graph coloring"],
            "languages": ["python", "cpp"]
        },
        {
            "domain": "algorithms",
            "template": "dynamic_programming",
            "concepts": ["optimization", "memoization", "recursion"],
            "languages": ["python", "cpp"]
        },
        # Data science problems - mainly Python
        {
            "domain": "data_science",
            "template": "time_series",
            "concepts": ["autocorrelation", "forecasting", "signal analysis"],
            "languages": ["python"]
        },
        {
            "domain": "data_science",
            "template": "statistical_inference",
            "concepts": ["hypothesis testing", "confidence intervals", "distributions"],
            "languages": ["python"]
        }
    ]
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    async def generate_problem_from_template(self, template: Dict[str, Any]) -> CodeProblem:
        """Generate a problem based on a template"""
        
        prompt = f"""
Generate a challenging scientific coding problem in the {template['domain']} domain.

Template: {template['template']}
Key concepts to incorporate: {', '.join(template['concepts'])}
Will be implemented in: {', '.join(template.get('languages', ['python']))}

Requirements:
1. The problem should be solvable with code in under 100 lines
2. It should have a clear, verifiable numerical or structured output
3. Include specific input constraints and ranges
4. Make it challenging but implementable across multiple programming languages
5. The problem should be self-contained with all necessary context
6. Include realistic parameter values and scenarios
7. Focus on algorithmic/mathematical content rather than language-specific features

Format your response as:
PROBLEM STATEMENT:
[Clear problem description with background]

INPUT CONSTRAINTS:
[Specific constraints on inputs]

OUTPUT FORMAT:
[Expected output format]

EXAMPLE:
Input: [example input]
Output: [example output]

DIFFICULTY: [easy/medium/hard/expert]
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_problem(response, template)
    
    async def generate_problem_from_paper(self, paper_text: str) -> CodeProblem:
        """Generate a problem inspired by scientific paper content"""
        
        prompt = f"""
Based on this scientific paper excerpt, create a computational problem that tests understanding of the concepts:

Paper excerpt:
{paper_text[:2000]}

Create a problem that:
1. Is inspired by the scientific concepts in the paper
2. Requires implementing a computational solution
3. Has verifiable numerical outputs
4. Can be solved in under 100 lines of code in multiple programming languages
5. Tests deep understanding, not just formula application
6. Focuses on algorithmic/mathematical aspects that translate across languages

Format your response as:
PROBLEM STATEMENT:
[Clear problem description]

INPUT CONSTRAINTS:
[Specific constraints]

OUTPUT FORMAT:
[Expected output]

EXAMPLE:
Input: [example]
Output: [example]

DIFFICULTY: [easy/medium/hard/expert]

KEY CONCEPTS: [list of concepts tested]
"""
        
        response = await self.llm.generate(prompt, temperature=0.8)
        return self._parse_problem_from_paper(response)
    
    def _parse_problem(self, response: str, template: Dict[str, Any]) -> CodeProblem:
        """Parse LLM response into CodeProblem"""
        lines = response.strip().split('\n')
        
        problem_statement = ""
        constraints = {}
        difficulty = "medium"
        
        current_section = None
        for line in lines:
            if "PROBLEM STATEMENT:" in line:
                current_section = "problem"
            elif "INPUT CONSTRAINTS:" in line:
                current_section = "constraints"
            elif "OUTPUT FORMAT:" in line:
                current_section = "output"
            elif "DIFFICULTY:" in line:
                difficulty = line.split(":")[-1].strip().lower()
            elif current_section == "problem":
                problem_statement += line + "\n"
        
        return CodeProblem(
            problem_statement=problem_statement.strip(),
            domain=template['domain'],
            difficulty=difficulty,
            concepts=template['concepts'],
            constraints=constraints
        )
    
    def _parse_problem_from_paper(self, response: str) -> CodeProblem:
        """Parse problem generated from paper"""
        # Similar parsing logic but extracting concepts from response
        lines = response.strip().split('\n')
        
        problem_statement = ""
        constraints = {}
        difficulty = "medium"
        concepts = []
        
        current_section = None
        for line in lines:
            if "PROBLEM STATEMENT:" in line:
                current_section = "problem"
            elif "INPUT CONSTRAINTS:" in line:
                current_section = "constraints"
            elif "DIFFICULTY:" in line:
                difficulty = line.split(":")[-1].strip().lower()
            elif "KEY CONCEPTS:" in line:
                concepts = [c.strip() for c in line.split(":")[-1].split(",")]
            elif current_section == "problem":
                problem_statement += line + "\n"
        
        # Detect domain from concepts
        domain = self._detect_domain(concepts)
        
        return CodeProblem(
            problem_statement=problem_statement.strip(),
            domain=domain,
            difficulty=difficulty,
            concepts=concepts,
            constraints=constraints
        )
    
    def _detect_domain(self, concepts: List[str]) -> str:
        """Detect domain from concepts"""
        concept_text = ' '.join(concepts).lower()
        
        if any(word in concept_text for word in ['physics', 'mechanics', 'quantum', 'thermodynamics']):
            return 'physics'
        elif any(word in concept_text for word in ['prime', 'number theory', 'combinatorics']):
            return 'mathematics'
        elif any(word in concept_text for word in ['graph', 'algorithm', 'complexity']):
            return 'algorithms'
        elif any(word in concept_text for word in ['statistics', 'data', 'regression']):
            return 'data_science'
        else:
            return 'general'

class CodeGenerator:
    """Generates code solutions with test cases"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    async def generate_solution(self, problem: CodeProblem, language: str = "python") -> str:
        """Generate a solution for the problem"""
        
        if language == "python":
            return await self._generate_python_solution(problem)
        elif language == "cpp":
            return await self._generate_cpp_solution(problem)
        elif language == "fortran":
            return await self._generate_fortran_solution(problem)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    async def _generate_python_solution(self, problem: CodeProblem) -> str:
        """Generate Python solution"""
        
        prompt = f"""
Solve this scientific computing problem with clean, efficient Python code:

Problem: {problem.problem_statement}
Domain: {problem.domain}
Concepts: {', '.join(problem.concepts)}

Requirements:
1. Write a complete, working solution in Python
2. Use only standard library, numpy, and scipy if needed
3. Include a main function that handles input/output
4. Make the code efficient and well-structured
5. Keep it under 100 lines
6. AVOID infinite loops - ensure all loops have proper termination conditions
7. For iterative algorithms, limit iterations (e.g., max 10000 iterations)
8. Handle edge cases properly (empty input, zero values, etc.)

Format your response as:
```python
import sys

def solve(n):
    # Your solution here
    return result

if __name__ == "__main__":
    # Read input
    n = int(input())
    # Compute and print result
    result = solve(n)
    print(result)
```

IMPORTANT: 
- The code must complete execution within 30 seconds
- Avoid recursive solutions without proper base cases
- For numerical methods, use reasonable tolerances (e.g., 1e-6)
"""
        
        response = await self.llm.generate(prompt, temperature=0.6)
        return self._extract_code(response, "python")
    
    async def _generate_cpp_solution(self, problem: CodeProblem) -> str:
        """Generate C++ solution"""
        
        prompt = f"""
Solve this scientific computing problem with clean, efficient C++ code:

Problem: {problem.problem_statement}
Domain: {problem.domain}
Concepts: {', '.join(problem.concepts)}

Requirements:
1. Write a complete, working solution in C++
2. Use only standard library (iostream, vector, cmath, algorithm, etc.)
3. Handle input/output using cin/cout
4. Make the code efficient and well-structured
5. Keep it under 100 lines
6. AVOID infinite loops - ensure proper termination
7. For iterative algorithms, limit iterations

Format your response as:
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

int main() {{
    // Read input
    int n;
    cin >> n;
    
    // Solve problem
    double result = 0.0;
    // ... computation ...
    
    // Output result
    cout << result << endl;
    
    return 0;
}}
```

IMPORTANT: 
- The code must compile with: g++ -std=c++17 -O2
- Must complete execution within 30 seconds
- Use proper input/output with cin/cout
"""
        
        response = await self.llm.generate(prompt, temperature=0.6)
        return self._extract_code(response, "cpp")
    
    async def _generate_fortran_solution(self, problem: CodeProblem) -> str:
        """Generate Fortran solution"""
        
        prompt = f"""
Solve this scientific computing problem with clean, efficient Fortran code:

Problem: {problem.problem_statement}
Domain: {problem.domain}
Concepts: {', '.join(problem.concepts)}

Requirements:
1. Write a complete, working solution in Fortran 90/95
2. Use modern Fortran 90/95 syntax (NOT Fortran 77)
3. Handle input/output using READ and WRITE statements
4. Make the code efficient and well-structured
5. Keep it under 100 lines
6. Use proper Fortran conventions

IMPORTANT FORTRAN SYNTAX RULES:
- Use 'program' and 'end program' with matching names
- Always include 'implicit none' after program/subroutine/function declarations
- Declare all variables with proper types (integer, real, real(8), etc.)
- Use proper indentation (spaces, not tabs)
- For reading input: READ(*,*) variable
- For writing output: WRITE(*,*) variable
- Avoid using obsolete features like COMMON blocks or fixed format
- Use lowercase for keywords (modern style)

Format your response as a COMPLETE, COMPILABLE Fortran program:
```fortran
program main
    implicit none
    ! Variable declarations
    integer :: n, i
    real(8) :: result
    
    ! Read input
    read(*,*) n
    
    ! Compute result
    result = 0.0d0
    do i = 1, n
        result = result + real(i, 8)
    end do
    
    ! Write output
    write(*,*) result
end program main
```

NOTE: The code MUST compile with: gfortran -o program program.f90
"""
        
        response = await self.llm.generate(prompt, temperature=0.6)
        return self._extract_code(response, "fortran")
    
    async def generate_tests(self, problem: CodeProblem, solution_code: str, language: str = "python") -> List[Dict[str, Any]]:
        """Generate test cases for the solution"""
        
        lang_name = {"python": "Python", "cpp": "C++", "fortran": "Fortran"}[language]
        code_block = f"```{language}\n{solution_code}\n```"
        
        prompt = f"""
Generate comprehensive test cases for this problem and solution:

Problem: {problem.problem_statement}
Solution code ({lang_name}):
{code_block}

Generate 5-10 test cases that:
1. Cover edge cases and typical cases
2. Test the correctness thoroughly
3. Include both simple and complex inputs
4. Have verifiable outputs
5. Consider the specific characteristics of {lang_name} (e.g., numerical precision, performance)

Format each test case as:
TEST 1:
Input: [exact input format]
Expected Output: [exact expected output]
Description: [what this tests]

TEST 2:
...
"""
        
        response = await self.llm.generate(prompt, temperature=0.5)
        return self._parse_tests(response)
    
    def _extract_code(self, response: str, language: str = "python") -> str:
        """Extract code from LLM response"""
        
        # Try to extract from code blocks first
        if language == "python" and "```python" in response:
            start = response.index("```python") + 9
            end = response.index("```", start)
            return response[start:end].strip()
        elif language == "cpp" and "```cpp" in response:
            start = response.index("```cpp") + 6
            end = response.index("```", start)
            return response[start:end].strip()
        elif language == "fortran" and "```fortran" in response:
            start = response.index("```fortran") + 11
            end = response.index("```", start)
            return response[start:end].strip()
        elif "```" in response:
            # Generic code block
            start = response.index("```")
            start = response.index("\n", start) + 1
            end = response.index("```", start)
            return response[start:end].strip()
        else:
            # Try to extract code heuristically
            return self._heuristic_code_extraction(response, language)
    
    def _heuristic_code_extraction(self, response: str, language: str) -> str:
        """Heuristically extract code based on language patterns"""
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        if language == "python":
            for line in lines:
                if line.strip().startswith('def ') or line.strip().startswith('import ') or line.strip().startswith('if __name__'):
                    in_code = True
                if in_code:
                    code_lines.append(line)
        elif language == "cpp":
            for line in lines:
                if line.strip().startswith('#include') or line.strip().startswith('using namespace') or 'int main(' in line:
                    in_code = True
                if in_code:
                    code_lines.append(line)
        elif language == "fortran":
            for line in lines:
                if line.strip().lower().startswith('program ') or line.strip().lower().startswith('subroutine '):
                    in_code = True
                if in_code:
                    code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _parse_tests(self, response: str) -> List[Dict[str, Any]]:
        """Parse test cases from response"""
        tests = []
        lines = response.strip().split('\n')
        
        current_test = {}
        for line in lines:
            if line.startswith('TEST'):
                if current_test:
                    tests.append(current_test)
                current_test = {}
            elif 'Input:' in line:
                current_test['input'] = line.split('Input:')[-1].strip()
            elif 'Expected Output:' in line or 'Output:' in line:
                current_test['expected_output'] = line.split(':')[-1].strip()
            elif 'Description:' in line:
                current_test['description'] = line.split('Description:')[-1].strip()
        
        if current_test:
            tests.append(current_test)
        
        return tests

class CodeExecutor:
    """Executes code and runs tests"""
    
    @staticmethod
    def execute_with_timeout(code: str, test_input: str, language: str = "python", timeout: int = 30) -> Tuple[bool, str, float, float]:
        """Execute code with input and timeout"""
        
        if language == "python":
            return CodeExecutor._execute_python(code, test_input, timeout)
        elif language == "cpp":
            return CodeExecutor._execute_cpp(code, test_input, timeout)
        elif language == "fortran":
            return CodeExecutor._execute_fortran(code, test_input, timeout)
        else:
            return False, f"Unsupported language: {language}", 0.0, 0.0
    
    @staticmethod
    def _execute_python(code: str, test_input: str, timeout: int) -> Tuple[bool, str, float, float]:
        """Execute Python code"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Modify code to accept input
            modified_code = CodeExecutor._modify_python_for_testing(code, test_input)
            f.write(modified_code)
            f.flush()
            
            start_time = time.time()
            try:
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                execution_time = (time.time() - start_time) * 1000  # ms
                
                if result.returncode == 0:
                    return True, result.stdout.strip(), execution_time, 0.0
                else:
                    return False, result.stderr, execution_time, 0.0
                    
            except subprocess.TimeoutExpired:
                return False, "Timeout exceeded", timeout * 1000, 0.0
            except Exception as e:
                return False, str(e), 0.0, 0.0
            finally:
                os.unlink(f.name)
    
    @staticmethod
    def _execute_cpp(code: str, test_input: str, timeout: int) -> Tuple[bool, str, float, float]:
        """Execute C++ code"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            f.flush()
            
            # Compile
            exe_name = f.name.replace('.cpp', '')
            compile_result = subprocess.run(
                ['g++', '-std=c++17', '-O2', '-o', exe_name, f.name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                os.unlink(f.name)
                return False, f"Compilation error: {compile_result.stderr}", 0.0, 0.0
            
            # Execute
            start_time = time.time()
            try:
                result = subprocess.run(
                    [exe_name],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                execution_time = (time.time() - start_time) * 1000  # ms
                
                success = result.returncode == 0
                output = result.stdout.strip() if success else result.stderr
                return success, output, execution_time, 0.0
                
            except subprocess.TimeoutExpired:
                return False, "Timeout exceeded", timeout * 1000, 0.0
            except Exception as e:
                return False, str(e), 0.0, 0.0
            finally:
                # Cleanup
                try:
                    os.unlink(f.name)
                    os.unlink(exe_name)
                except:
                    pass
    
    @staticmethod
    def _execute_fortran(code: str, test_input: str, timeout: int) -> Tuple[bool, str, float, float]:
        """Execute Fortran code"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=False) as f:
            # Clean up the code before writing
            cleaned_code = CodeExecutor._clean_fortran_code(code)
            f.write(cleaned_code)
            f.flush()
            
            # Compile with more lenient flags
            exe_name = f.name.replace('.f90', '')
            compile_result = subprocess.run(
                ['gfortran', '-std=f95', '-ffree-form', '-O2', '-o', exe_name, f.name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if compile_result.returncode != 0:
                # Try with even more lenient flags
                compile_result = subprocess.run(
                    ['gfortran', '-ffree-form', '-O2', '-o', exe_name, f.name],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
            if compile_result.returncode != 0:
                os.unlink(f.name)
                return False, f"Compilation error: {compile_result.stderr}", 0.0, 0.0
            
            # Execute
            start_time = time.time()
            try:
                result = subprocess.run(
                    [exe_name],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                execution_time = (time.time() - start_time) * 1000  # ms
                
                success = result.returncode == 0
                output = result.stdout.strip() if success else result.stderr
                return success, output, execution_time, 0.0
                
            except subprocess.TimeoutExpired:
                return False, "Timeout exceeded", timeout * 1000, 0.0
            except Exception as e:
                return False, str(e), 0.0, 0.0
            finally:
                # Cleanup
                try:
                    os.unlink(f.name)
                    os.unlink(exe_name)
                except:
                    pass
    
    @staticmethod
    def _modify_python_for_testing(code: str, test_input: str) -> str:
        """Modify Python code to use test input instead of stdin"""
        # More robust approach to handle various input methods
        
        # Check if the code reads from stdin in any way
        reads_input = any(pattern in code for pattern in [
            'input(', 'sys.stdin', 'stdin.read', 'raw_input('
        ])
        
        if reads_input or test_input.strip():
            # Create comprehensive mock input that handles multiple input methods
            mock_input = f"""
import sys
from io import StringIO
import builtins

# Mock stdin for all input methods
sys.stdin = StringIO('''{test_input}''')

# Also mock the input() function directly
_test_inputs = iter('''{test_input}'''.strip().split('\\n'))
def _mock_input(prompt=''):
    try:
        return next(_test_inputs)
    except StopIteration:
        return ''

builtins.input = _mock_input
"""
            return mock_input + '\n' + code
        else:
            return code
    
    @staticmethod
    async def verify_solution(solution_code: str, test_cases: List[Dict[str, Any]], language: str = "python") -> CodeSolution:
        """Verify solution against test cases"""
        
        passed_count = 0
        failed_tests = []
        total_time = 0.0
        
        # Debug: Print code being tested
        if language == "fortran":
            print(f"      Testing Fortran code ({len(solution_code.split(chr(10)))} lines)...")
        
        for i, test in enumerate(test_cases):
            try:
                success, output, exec_time, _ = CodeExecutor.execute_with_timeout(
                    solution_code,
                    test.get('input', ''),
                    language,
                    timeout=30  # Increase timeout for verification
                )
                
                total_time += exec_time
                
                if success:
                    # Check if output matches expected
                    expected = test.get('expected_output', '').strip()
                    actual = output.strip()
                    
                    if CodeExecutor._outputs_match(expected, actual):
                        passed_count += 1
                        print(f"        Test {i+1}: PASS")
                    else:
                        failed_tests.append(f"Test {i+1}: Expected '{expected}', got '{actual}'")
                        print(f"        Test {i+1}: FAIL - Output mismatch")
                else:
                    failed_tests.append(f"Test {i+1}: Execution error - {output[:200]}")
                    print(f"        Test {i+1}: FAIL - {output[:100]}")
                    
            except Exception as e:
                failed_tests.append(f"Test {i+1}: Exception - {str(e)}")
                print(f"        Test {i+1}: EXCEPTION - {str(e)[:100]}")
        
        passed = passed_count == len(test_cases)
        error_msg = None if passed else "; ".join(failed_tests)
        
        return CodeSolution(
            code=solution_code,
            language=language,
            test_cases=test_cases,
            execution_time_ms=total_time / len(test_cases) if test_cases else 0,
            memory_usage_mb=0.0,  # Not measured in this simple version
            passed_tests=passed,
            error_message=error_msg
        )
    
    @staticmethod
    def _clean_fortran_code(code: str) -> str:
        """Clean Fortran code to fix common issues"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove tabs and replace with spaces
            line = line.replace('\t', '    ')
            
            # Ensure proper spacing around operators
            if not line.strip().startswith('!'):
                # Don't modify comment lines
                line = line
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _outputs_match(expected: str, actual: str) -> bool:
        """Check if outputs match, handling numerical tolerance"""
        # Try exact match first
        if expected == actual:
            return True
        
        # Try numerical comparison with tolerance
        try:
            expected_nums = [float(x) for x in expected.split()]
            actual_nums = [float(x) for x in actual.split()]
            
            if len(expected_nums) != len(actual_nums):
                return False
            
            for e, a in zip(expected_nums, actual_nums):
                if abs(e - a) > 1e-6 * max(abs(e), abs(a), 1):
                    return False
            return True
            
        except:
            # Not numerical, do string comparison
            return expected.strip().lower() == actual.strip().lower()

class VerifiableReasoningGenerator:
    """Main generator for verifiable code reasoning dataset"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.problem_generator = ScientificProblemGenerator(llm)
        self.code_generator = CodeGenerator(llm)
        self.executor = CodeExecutor()
    
    async def generate_from_template(self) -> Dict[str, Any]:
        """Generate a complete problem with solutions in all supported languages"""
        
        # Select random template
        template = random.choice(ScientificProblemGenerator.PROBLEM_TEMPLATES)
        supported_languages = template.get('languages', ['python'])
        
        print(f"  Generating problem from template: {template['template']}")
        print(f"  Will generate solutions in: {', '.join(supported_languages)}")
        
        # Generate problem (language-agnostic)
        problem = await self.problem_generator.generate_problem_from_template(template)
        
        print(f"  Generated problem in domain: {problem.domain}, difficulty: {problem.difficulty}")
        
        # Generate solutions in all supported languages
        solutions = []
        for lang in supported_languages:
            print(f"    Generating {lang} solution...")
            try:
                # Generate solution
                solution_code = await self.code_generator.generate_solution(problem, lang)
                
                print(f"    Generated {lang} solution ({len(solution_code.split(chr(10)))} lines)")
                
                # Generate test cases
                test_cases = await self.code_generator.generate_tests(problem, solution_code, lang)
                
                print(f"    Generated {len(test_cases)} test cases for {lang}")
                
                # Debug: Save code for inspection if it's Fortran
                if lang == "fortran" and len(solution_code) > 0:
                    debug_file = f"/tmp/debug_fortran_{lang}_{time.time()}.f90"
                    with open(debug_file, 'w') as df:
                        df.write(solution_code)
                    print(f"      Debug: Fortran code saved to {debug_file}")
                
                # Verify solution
                solution = await self.executor.verify_solution(solution_code, test_cases, lang)
                
                if solution.passed_tests:
                    print(f"    ✓ {lang} tests passed!")
                else:
                    print(f"    ✗ {lang} tests failed: {solution.error_message[:200]}")
                
                solutions.append({
                    'language': solution.language,
                    'code': solution.code,
                    'lines_of_code': len(solution.code.split('\n')),
                    'execution_time_ms': solution.execution_time_ms,
                    'passed_all_tests': solution.passed_tests,
                    'error': solution.error_message,
                    'tests': test_cases
                })
                
            except Exception as e:
                print(f"    ❌ Error generating {lang} solution: {e}")
                solutions.append({
                    'language': lang,
                    'code': None,
                    'lines_of_code': 0,
                    'execution_time_ms': 0,
                    'passed_all_tests': False,
                    'error': str(e),
                    'tests': []
                })
        
        # Check if at least one solution passed
        passed_solutions = [s for s in solutions if s['passed_all_tests']]
        
        return {
            'problem': {
                'statement': problem.problem_statement,
                'domain': problem.domain,
                'difficulty': problem.difficulty,
                'concepts': problem.concepts
            },
            'solutions': solutions,
            'metadata': {
                'generator': 'template',
                'template_name': template['template'],
                'supported_languages': supported_languages,
                'successful_languages': [s['language'] for s in passed_solutions],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    async def generate_from_paper(self, paper_text: str) -> Dict[str, Any]:
        """Generate from scientific paper text with solutions in all appropriate languages"""
        
        # Detect which languages are most appropriate for this paper
        preferred_languages = self._detect_suitable_languages(paper_text)
        
        print(f"  Generating problem from paper excerpt")
        print(f"  Will generate solutions in: {', '.join(preferred_languages)}")
        
        # Generate problem from paper (language-agnostic)
        problem = await self.problem_generator.generate_problem_from_paper(paper_text)
        
        print(f"  Generated problem in domain: {problem.domain}")
        
        # Generate solutions in all appropriate languages
        solutions = []
        for lang in preferred_languages:
            print(f"    Generating {lang} solution...")
            try:
                # Generate solution
                solution_code = await self.code_generator.generate_solution(problem, lang)
                
                print(f"    Generated {lang} solution ({len(solution_code.split(chr(10)))} lines)")
                
                # Generate test cases
                test_cases = await self.code_generator.generate_tests(problem, solution_code, lang)
                
                print(f"    Generated {len(test_cases)} test cases for {lang}")
                
                # Debug: Save code for inspection if it's Fortran
                if lang == "fortran" and len(solution_code) > 0:
                    debug_file = f"/tmp/debug_fortran_{lang}_{time.time()}.f90"
                    with open(debug_file, 'w') as df:
                        df.write(solution_code)
                    print(f"      Debug: Fortran code saved to {debug_file}")
                
                # Verify solution
                solution = await self.executor.verify_solution(solution_code, test_cases, lang)
                
                if solution.passed_tests:
                    print(f"    ✓ {lang} tests passed!")
                else:
                    print(f"    ✗ {lang} tests failed: {solution.error_message[:200]}")
                
                solutions.append({
                    'language': solution.language,
                    'code': solution.code,
                    'lines_of_code': len(solution.code.split('\n')),
                    'execution_time_ms': solution.execution_time_ms,
                    'passed_all_tests': solution.passed_tests,
                    'error': solution.error_message,
                    'tests': test_cases
                })
                
            except Exception as e:
                print(f"    ❌ Error generating {lang} solution: {e}")
                solutions.append({
                    'language': lang,
                    'code': None,
                    'lines_of_code': 0,
                    'execution_time_ms': 0,
                    'passed_all_tests': False,
                    'error': str(e),
                    'tests': []
                })
        
        # Check if at least one solution passed
        passed_solutions = [s for s in solutions if s['passed_all_tests']]
        
        return {
            'problem': {
                'statement': problem.problem_statement,
                'domain': problem.domain,
                'difficulty': problem.difficulty,
                'concepts': problem.concepts,
                'source': 'paper'
            },
            'solutions': solutions,
            'metadata': {
                'generator': 'paper',
                'preferred_languages': preferred_languages,
                'successful_languages': [s['language'] for s in passed_solutions],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def _detect_suitable_languages(self, paper_text: str) -> List[str]:
        """Detect suitable languages based on paper content"""
        text_lower = paper_text.lower()
        
        # Language indicators
        fortran_indicators = ['fortran', 'numerical simulation', 'finite difference', 'monte carlo', 
                             'computational fluid dynamics', 'lattice', 'molecular dynamics', 
                             'matrix computation', 'linear algebra']
        
        cpp_indicators = ['c++', 'algorithm', 'optimization', 'performance', 'parallel computing', 
                         'graph theory', 'data structure', 'computational geometry']
        
        python_indicators = ['python', 'data analysis', 'machine learning', 'statistics', 
                           'visualization', 'pandas', 'numpy', 'scipy']
        
        fortran_score = sum(1 for indicator in fortran_indicators if indicator in text_lower)
        cpp_score = sum(1 for indicator in cpp_indicators if indicator in text_lower)
        python_score = sum(1 for indicator in python_indicators if indicator in text_lower)
        
        # Determine suitable languages based on scores
        suitable_languages = []
        
        # Always include Python as it's most versatile
        suitable_languages.append('python')
        
        # Add other languages if they have good indicators
        if fortran_score >= 2 or 'numerical' in text_lower or 'simulation' in text_lower:
            suitable_languages.append('fortran')
        
        if cpp_score >= 2 or 'algorithm' in text_lower or 'optimization' in text_lower:
            suitable_languages.append('cpp')
        
        # If no specific indicators, use all languages for maximum coverage
        if len(suitable_languages) == 1:
            suitable_languages = ['python', 'cpp', 'fortran']
        
        return suitable_languages

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate verifiable code reasoning dataset")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="claudesonnet4",
        help="LLM model to use (default: claudesonnet4)"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional: Input file with paper texts for seed generation"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to generate (default: 10)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="verifiable_code_dataset.json",
        help="Output file for generated dataset (default: verifiable_code_dataset.json)"
    )
    
    parser.add_argument(
        "--paper-ratio",
        type=float,
        default=0.3,
        help="Ratio of samples to generate from papers if input file provided (default: 0.3)"
    )
    
    parser.add_argument(
        "--require-all-pass",
        action="store_true",
        help="Require all language solutions to pass tests (default: accept if at least one passes)"
    )
    
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_arguments()
    
    if rank == 0:
        print(f"Verifiable Code Reasoning Dataset Generator")
        print(f"Model: {args.model}")
        print(f"Max samples: {args.max_samples}")
        print(f"Multi-language generation: Python + C++ + Fortran")
        print(f"Acceptance criteria: {'All languages must pass' if args.require_all_pass else 'At least one language must pass'}")
        if args.input_file:
            print(f"Input file: {args.input_file}")
            print(f"Paper ratio: {args.paper_ratio}")
        print("-" * 50)
    
    # Initialize LLM
    llm = ArgoLLMAdapter(args.model)
    generator = VerifiableReasoningGenerator(llm)
    
    # Load papers if provided
    papers = []
    if args.input_file and os.path.exists(args.input_file):
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', data.get('passage', ''))
                    if text and len(text) > 500:
                        papers.append(text)
                except:
                    continue
        
        if rank == 0:
            print(f"Loaded {len(papers)} papers from {args.input_file}")
    
    # Generate samples
    samples = []
    max_attempts_per_sample = 3
    
    samples_per_rank = args.max_samples // max(size, 1)
    if rank < args.max_samples % max(size, 1):
        samples_per_rank += 1
    
    for i in range(samples_per_rank):
        sample_idx = rank * (args.max_samples // max(size, 1)) + min(rank, args.max_samples % max(size, 1)) + i
        print(f"[Rank {rank}] Generating sample {sample_idx + 1}/{args.max_samples}")
        
        success = False
        for attempt in range(max_attempts_per_sample):
            try:
                # Decide whether to use paper or template
                use_paper = papers and random.random() < args.paper_ratio
                
                if use_paper:
                    paper = random.choice(papers)
                    sample = await generator.generate_from_paper(paper)
                else:
                    sample = await generator.generate_from_template()
                
                # Check acceptance criteria
                passed_solutions = [s for s in sample['solutions'] if s['passed_all_tests']]
                
                if args.require_all_pass:
                    # Require ALL solutions to pass
                    accept_sample = len(passed_solutions) == len(sample['solutions'])
                    criteria = "all languages pass"
                else:
                    # Accept if at least one solution passes
                    accept_sample = len(passed_solutions) > 0
                    criteria = "at least one language passes"
                
                if accept_sample:
                    samples.append(sample)
                    success = True
                    lang_summary = ", ".join([f"{s['language']}:{'✓' if s['passed_all_tests'] else '✗'}" for s in sample['solutions']])
                    print(f"  ✅ Sample {sample_idx + 1} generated successfully ({lang_summary})")
                    break
                else:
                    print(f"  ⚠️ Attempt {attempt + 1} failed criteria ({criteria}), retrying...")
                    
            except Exception as e:
                print(f"  ❌ Error in attempt {attempt + 1}: {e}")
        
        if not success:
            print(f"  ❌ Failed to generate valid sample after {max_attempts_per_sample} attempts")
    
    # Save results
    output_file = args.output
    if size > 1:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}_rank{rank}{ext}"
    
    output_data = {
        'metadata': {
            'generator': 'llm_verifiable_code',
            'model': args.model,
            'total_samples': len(samples),
            'require_all_pass': args.require_all_pass,
            'paper_ratio': args.paper_ratio if args.input_file else 0.0,
            'supported_languages': ['python', 'cpp', 'fortran'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'samples': samples
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[Rank {rank}] Generated {len(samples)} samples, saved to {output_file}")
    
    if MPI_AVAILABLE and size > 1:
        comm.Barrier()
        if rank == 0:
            print(f"✅ All {size} MPI processes completed")

if __name__ == "__main__":
    asyncio.run(main())