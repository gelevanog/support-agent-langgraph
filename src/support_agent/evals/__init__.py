"""Evaluation suite: labelled tickets, deterministic metrics and an LLM-as-judge for reply quality.

Classification, decisions, actions and retrieval have ground truth, so they are scored exactly.
Reply quality has none, so a judge model scores groundedness, tone and policy compliance against
the context the agent was allowed to use. `python -m support_agent.evals` runs everything.
"""
