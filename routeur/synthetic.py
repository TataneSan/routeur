from __future__ import annotations

import random
from typing import Any

LEVEL_TEMPLATES = {
    1: [
        "Classify this message as positive or negative: {topic}",
        "Extract the email address from this text: {topic}",
        "Translate this short phrase to English: {topic}",
        "Return a JSON category for this support ticket: {topic}",
    ],
    2: [
        "Write a concise follow-up email about {topic}",
        "Rewrite this paragraph in a friendlier tone: {topic}",
        "Summarize these meeting notes in five bullets: {topic}",
        "Draft a customer support answer for {topic}",
    ],
    3: [
        "Compare the tradeoffs of three implementation options for {topic}",
        "Analyze this product feedback and propose priorities: {topic}",
        "Create a migration checklist for {topic}",
        "Explain the likely root causes and next steps for {topic}",
    ],
    4: [
        "Debug this Python traceback and propose a patch: {topic}",
        "Solve this probability problem step by step: {topic}",
        "Review this SQL query for correctness and performance: {topic}",
        "Design an algorithm and discuss edge cases for {topic}",
    ],
    5: [
        "Review this production security vulnerability and assess exploit risk: {topic}",
        "Analyze this legal contract clause and identify risky obligations: {topic}",
        "Audit this medical triage policy for dangerous failure modes: {topic}",
        "Evaluate a high-stakes incident response plan for {topic}",
    ],
}

TASK_TEMPLATES = {
    "general": [
        "Answer this straightforward question clearly and briefly: {topic}.",
        "Extract the requested fields from this short support message about {topic}.",
    ],
    "frontend": [
        "Build a responsive React dashboard with accessible keyboard navigation and loading states for {topic}.",
        "Turn this screenshot into a polished Next.js page with responsive CSS and reusable components for {topic}.",
        "Review this frontend performance regression and propose a fix that preserves the visual design: {topic}.",
    ],
    "backend": [
        "Design a production REST API and PostgreSQL schema for {topic}, including idempotency and failure handling.",
        "Debug this distributed backend incident involving timeouts, retries, and duplicate writes: {topic}.",
        "Plan a safe migration of a multi-tenant backend for {topic}, including rollback and observability.",
    ],
    "coding": [
        "Implement a robust function and tests for this requirement, including edge cases: {topic}.",
        "Refactor this code for correctness and maintainability, then explain the tradeoffs: {topic}.",
    ],
    "reasoning": [
        "Solve this non-trivial probability problem and verify every assumption: {topic}.",
        "Prove or disprove the following statement, including edge cases and a rigorous argument: {topic}.",
        "Derive the result step by step and explain why the tempting shortcut fails: {topic}.",
    ],
    "research": [
        "Investigate {topic}, compare credible sources, and present a cited recommendation with uncertainty.",
        "Perform a deep literature and market review of {topic}; separate established facts from speculation.",
    ],
    "writing": [
        "Rewrite this customer communication in a warm, concise tone while preserving the facts: {topic}.",
        "Draft a clear one-page brief for non-technical readers about {topic}.",
    ],
    "data": [
        "Write and optimize a SQL query for {topic}; explain indexes, correctness, and null behavior.",
        "Analyze this data pipeline and propose a reliable metric definition for {topic}.",
    ],
    "safety": [
        "Audit this production change for security, legal, and operational failure modes before deployment: {topic}.",
        "Assess the risks of this medical, financial, or compliance decision about {topic}; be conservative.",
    ],
    "vision": [
        "Analyze this image, extract the visible facts, and explain uncertain details: {topic}.",
        "Read this diagram and answer the visual question without inventing hidden information: {topic}.",
    ],
    "agentic": [
        "Plan and execute a multi-step tool-using workflow for {topic}, including verification and recovery.",
        "Build an autonomous agent for {topic} with tool selection, state, retries, and safe stopping rules.",
    ],
}

TOPICS = [
    "subscription cancellation",
    "payment failure",
    "database timeout",
    "OAuth login regression",
    "invoice reconciliation",
    "multi-tenant permissions",
    "GPU inference cost",
    "customer churn",
    "data retention",
    "API rate limiting",
]

DETAILS = [
    "with strict latency SLOs and graceful degradation",
    "for a multi-tenant SaaS product with audit logs",
    "with French and English users and locale-specific edge cases",
    "under a moderate traffic spike after a release",
    "with incomplete input and adversarial users",
    "where backwards compatibility is mandatory",
    "with a small team and a limited operational budget",
    "including tests, monitoring, and a rollback plan",
    "where correctness matters more than raw throughput",
    "with a hard requirement for keyboard and screen-reader access",
    "using an existing PostgreSQL and Redis stack",
    "where the output must be valid JSON and schema-checked",
    "with a 100k-row dataset and missing values",
    "for a regulated production environment",
    "with three competing implementation options",
    "where the request is ambiguous and assumptions must be explicit",
    "including examples of common failure modes",
    "with a security review before deployment",
    "where the answer will be used by a non-technical stakeholder",
    "with a migration from a legacy monolith",
]


def generate_seed_examples(count: int, *, seed: int = 42) -> list[dict[str, Any]]:
    if count < 5:
        raise ValueError("count must be at least 5")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        level = (idx % 5) + 1
        template = rng.choice(LEVEL_TEMPLATES[level])
        topic = f"{rng.choice(TOPICS)} {rng.choice(DETAILS)}"
        rows.append(
            {
                "prompt": template.format(topic=topic),
                "level": level,
                "task": "general",
                "source": "synthetic_seed",
                "metadata": {"seed": seed},
            }
        )
    rng.shuffle(rows)
    return rows


def generate_specialty_examples(count: int, *, seed: int = 42) -> list[dict[str, Any]]:
    """Generate balanced task examples with deliberately mixed difficulty."""
    if count < len(TASK_TEMPLATES):
        raise ValueError(f"count must be at least {len(TASK_TEMPLATES)}")
    rng = random.Random(seed)
    tasks = list(TASK_TEMPLATES)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        task = tasks[idx % len(tasks)]
        level = 1 + ((idx // len(tasks)) % 5)
        if task in {"safety", "reasoning"}:
            level = max(level, 4)
        if task in {"frontend", "backend", "coding", "data", "vision", "agentic"}:
            level = max(level, 3)
        template = rng.choice(TASK_TEMPLATES[task])
        topic = f"{rng.choice(TOPICS)} {rng.choice(DETAILS)}"
        rows.append(
            {
                "prompt": template.format(topic=topic),
                "level": level,
                "task": task,
                "source": "synthetic_specialty",
                "metadata": {"seed": seed},
            }
        )
    rng.shuffle(rows)
    return rows
