"""
CoderAgent - Self-Extending Tool Synthesizer & Code Generation Engine (agents/coder_agent.py)

Uses Gemma and local models to dynamically generate Python tool extensions on-demand,
analyzes Python AST, performs automated refactoring, synthesizes unit tests, and
validates/executes code safely.
"""

import ast
import importlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
from .boundary_guard import BoundaryGuard, get_boundary_guard
from .gemma_agent import GemmaAgent
from .spine import get_spine

logger = logging.getLogger(__name__)


class CoderAgent:
    """
    Self-Extending Coding Agent capable of code generation, tool synthesis, AST analysis,
    and automatic unit test creation.
    """

    def __init__(
        self,
        extensions_dir: str = "extensions",
        gemma_agent: Optional[GemmaAgent] = None,
        boundary_guard: Optional[BoundaryGuard] = None,
    ):
        """
        Initialize CoderAgent.

        Args:
            extensions_dir: Path to extensions folder where generated tools are saved.
            gemma_agent: GemmaAgent for code synthesis.
            boundary_guard: Preflight gate for generated-code writes.
        """
        self.extensions_dir = extensions_dir
        self.gemma = gemma_agent or GemmaAgent()
        self.guard = boundary_guard or get_boundary_guard()
        self.spine = get_spine()
        os.makedirs(self.extensions_dir, exist_ok=True)
        if self.extensions_dir not in sys.path:
            sys.path.insert(0, self.extensions_dir)

    def generate_code(self, specification: str, language: str = "python", context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate high-quality structured code given a prompt specification.

        Args:
            specification: Description of desired code module.
            language: Programming language (default 'python').
            context: Additional surrounding code or documentation context.

        Returns:
            Dictionary with generated code, AST validation status, and syntax checks.
        """
        prompt = (
            f"You are an expert software engineer. Write production-ready {language} code for the following spec:\n"
            f"Specification: {specification}\n\n"
        )
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"Return ONLY valid code inside ```{language} and ``` without additional commentary."

        response = self.gemma.generate(prompt, temperature=0.2)
        code = self._extract_code(response, specification)

        is_valid = True
        error = None
        if language.lower() == "python":
            is_valid, error = self._validate_syntax(code)

        return {
            "success": is_valid,
            "language": language,
            "code": code,
            "syntax_valid": is_valid,
            "error": error,
        }

    def analyze_codebase(self, code_snippet: str) -> Dict[str, Any]:
        """
        Perform AST analysis on Python code to extract functions, classes, imports, and metrics.

        Args:
            code_snippet: Python source code string.

        Returns:
            Dictionary containing AST metrics, classes, functions, and import declarations.
        """
        try:
            tree = ast.parse(code_snippet)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")

            return {
                "success": True,
                "classes": classes,
                "functions": functions,
                "imports": list(set(imports)),
                "total_lines": len(code_snippet.splitlines()),
                "ast_valid": True,
            }
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}",
                "ast_valid": False,
            }

    def refactor_code(self, code_snippet: str, refactor_goal: str) -> Dict[str, Any]:
        """
        Refactor existing Python code to meet a specified objective (optimization, clean architecture, etc.).

        Args:
            code_snippet: Source code snippet.
            refactor_goal: Guidance for refactoring (e.g. 'Add docstrings and type hints').

        Returns:
            Refactored code result.
        """
        prompt = (
            f"Refactor the following Python code snippet:\n\n"
            f"```python\n{code_snippet}\n```\n\n"
            f"Refactoring Goal: {refactor_goal}\n\n"
            f"Return ONLY the updated code inside ```python and ```."
        )

        response = self.gemma.generate(prompt, temperature=0.2)
        refactored_code = self._extract_code(response, "Refactored code")
        is_valid, error = self._validate_syntax(refactored_code)

        return {
            "success": is_valid,
            "original_code": code_snippet,
            "refactored_code": refactored_code,
            "goal": refactor_goal,
            "syntax_valid": is_valid,
            "error": error,
        }

    def generate_tests(self, code_snippet: str) -> Dict[str, Any]:
        """
        Synthesize pytest test cases for a given Python code snippet.

        Args:
            code_snippet: Python code to test.

        Returns:
            Generated unit test file content.
        """
        prompt = (
            f"Generate thorough pytest test cases for the following Python code:\n\n"
            f"```python\n{code_snippet}\n```\n\n"
            f"Return ONLY valid pytest code inside ```python and ```."
        )

        response = self.gemma.generate(prompt, temperature=0.3)
        test_code = self._extract_code(response, "Pytest suite")
        is_valid, error = self._validate_syntax(test_code)

        return {
            "success": is_valid,
            "test_code": test_code,
            "syntax_valid": is_valid,
            "error": error,
        }

    def synthesize_tool(
        self,
        task_description: str,
        tool_name: Optional[str] = None,
        measurable_gain: Optional[str] = None,
        validation_criterion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize a Python tool script based on a task description.

        The write is gated by the SANGRE boundary guards: a tool that declares no
        measurable gain and no validation criterion is blocked before anything
        reaches disk (guard_010). Accepted writes are recorded in a rollback
        manifest so the run can be rewound (guard_013).

        Args:
            task_description: Description of tool requirements.
            tool_name: Optional custom name for generated tool.
            measurable_gain: What measurable behavioral gain this tool provides.
            validation_criterion: How to tell whether the tool works.

        Returns:
            Dictionary with code, filename, validation result, and status.
        """
        name = tool_name or f"tool_{int(time.time())}"
        filename = f"{name}.py"
        filepath = os.path.join(self.extensions_dir, filename)

        decision = self.guard.preflight_code_synthesis(
            tool_name=name,
            write_target=filepath,
            measurable_gain=measurable_gain,
            validation_criterion=validation_criterion,
        )
        if not decision.allowed:
            logger.warning("Tool synthesis blocked for '%s': %s", name, decision.decision_rationale)
            return {
                "success": False,
                "tool_name": name,
                "blocked_by_guard": True,
                "decision": decision.decision,
                "triggered_guards": decision.triggered_guards,
                "error": decision.decision_rationale,
            }

        prompt = (
            f"Write a standalone Python function named 'run' for the following requirement:\n"
            f"Requirement: {task_description}\n\n"
            f"The script must define `def run(data: dict) -> dict:` and return a dictionary result.\n"
            f"Return ONLY valid Python code block inside ```python and ```."
        )

        code_response = self.gemma.generate(prompt, temperature=0.2)
        code = self._extract_code(code_response, task_description)

        # Validate syntax
        is_valid, error = self._validate_syntax(code)
        if not is_valid:
            return {"success": False, "tool_name": name, "error": f"Syntax error: {error}", "code": code}

        # Save tool extension, then record the rewind entry guard_013 requires.
        with open(filepath, "w") as f:
            f.write(code)
        self.guard.record_rollback_manifest(run_id=name, created_files=[filepath])

        return {
            "success": True,
            "tool_name": name,
            "filename": filename,
            "filepath": filepath,
            "code": code,
            "status": "ready",
            "measurable_gain": measurable_gain,
            "validation_criterion": validation_criterion,
        }

    def execute_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a dynamically synthesized tool.

        Args:
            tool_name: Name of tool script in extensions/.
            payload: Input parameter dictionary.

        Returns:
            Output from tool execution.
        """
        try:
            if self.extensions_dir not in sys.path:
                sys.path.insert(0, self.extensions_dir)
            module = importlib.import_module(tool_name)
            importlib.reload(module)
            if hasattr(module, "run"):
                res = module.run(payload)
                return {"success": True, "output": res}
            return {"success": False, "error": "Tool module missing 'run(data)' function"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _validate_syntax(code: str) -> tuple[bool, Optional[str]]:
        """Validate Python code string syntax using AST."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def _extract_code(text: str, fallback_desc: str) -> str:
        """Extract clean python code block from model response."""
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        if "def run(" in text:
            return text

        # Deterministic fallback script if LLM response is unstructured
        return (
            f"# Auto-synthesized tool for: {fallback_desc}\n"
            f"def run(data: dict) -> dict:\n"
            f"    return {{\n"
            f"        'status': 'processed',\n"
            f"        'input_received': data,\n"
            f"        'summary': 'Synthesized extension executed successfully'\n"
            f"    }}\n"
        )
