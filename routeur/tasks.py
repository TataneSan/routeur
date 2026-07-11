from __future__ import annotations

import re

TASKS = (
    "general",
    "frontend",
    "backend",
    "coding",
    "reasoning",
    "research",
    "writing",
    "data",
    "safety",
    "vision",
    "agentic",
)

TASK_ALIASES = {
    "creation": "writing",
    "brainstorming": "writing",
    "rewriting": "writing",
    "summarization": "writing",
    "programming": "coding",
    "classification": "general",
    "extraction": "data",
    "closed_questions": "general",
    "open_questions": "general",
}

_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("safety", ("security", "vulnerability", "exploit", "authentication bypass", "legal", "contract", "medical", "diagnosis", "production incident", "compliance", "rgpd", "gdpr")),
    ("agentic", ("agentic", "ai agent", "multi-agent", "autonomous agent", "autonomously", "tool calling", "tool use", "browser tools", "mcp server", "browser automation", "execute commands", "computer use")),
    ("vision", ("analyze this image", "analyze the image", "describe this image", "image understanding", "read this photo", "ocr", "video analysis", "visual question", "diagram interpretation")),
    ("frontend", ("frontend", "front-end", "ui", "ux", "user interface", "web design", "website", "landing page", "react", "next.js", "vue", "css", "tailwind", "html", "browser", "responsive", "accessibility", "figma", "screenshot")),
    ("backend", ("backend", "back-end", "api", "rest api", "graphql", "microservice", "database", "postgres", "mysql", "redis", "kafka", "queue", "deployment", "docker", "kubernetes", "observability", "distributed system")),
    ("reasoning", ("prove", "proof", "theorem", "equation", "integral", "probability", "algebra", "geometry", "mathematics", "math", "physics", "chemistry", "derive", "reason step by step", "logic puzzle")),
    ("research", ("research", "investigate", "compare sources", "literature review", "find evidence", "browse", "web search", "citation", "fact-check", "market analysis")),
    ("data", ("sql", "dataframe", "pandas", "spreadsheet", "excel", "csv", "analytics", "metric", "etl", "query", "dashboard", "statistics")),
    ("coding", ("code", "coding", "program", "python", "javascript", "typescript", "java", "rust", "go ", "c++", "c#", "debug", "traceback", "refactor", "function", "class ", "algorithm", "repository", "pull request", "git")),
    ("writing", ("write", "rewrite", "translate", "email", "summarize", "summary", "blog", "copy", "creative", "poem", "tone", "draft", "caption")),
)


def normalize_task(value: str | None) -> str:
    if not value:
        return "general"
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    key = TASK_ALIASES.get(key, key)
    return key if key in TASKS else "general"


def infer_task(prompt: str, *, category: str | None = None) -> str:
    category_task = normalize_task(category)
    if category and category_task != "general":
        # Programming examples still benefit from the frontend/backend split.
        if category_task == "coding":
            prompt_task = infer_task(prompt)
            return prompt_task if prompt_task in {"frontend", "backend", "data", "safety"} else "coding"
        return category_task

    text = re.sub(r"\s+", " ", prompt.lower()).strip()
    for task, needles in _PATTERNS:
        if any(needle in text for needle in needles):
            return task
    return "general"


def difficulty_to_level(value: int | float | str | None) -> int:
    """Map the 1-7 HF difficulty scale to the router's conservative 1-5 scale."""
    try:
        difficulty = float(value)
    except (TypeError, ValueError):
        return 3
    if difficulty <= 1.5:
        return 1
    if difficulty <= 2.5:
        return 2
    if difficulty <= 4.0:
        return 3
    if difficulty <= 5.0:
        return 4
    return 5
