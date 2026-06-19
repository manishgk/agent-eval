"""agent-eval: statistical reliability testing for LLM agent tool-calling.

Runs each eval case many times against an LLM agent and reports *reliability*
(success rate), *flake rate*, Wilson confidence intervals, and pass^k — bringing
flaky-test discipline from software QA to non-deterministic AI systems.
"""

__version__ = "0.1.0"
