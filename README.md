# Swarm

A multi-agent CLI framework for autonomous task execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER TASK                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│  - Decomposes complex tasks into atomic subtasks            │
│  - Manages dependencies between subtasks                    │
│  - Coordinates parallel execution                           │
│  - Aggregates results                                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    GRUNT 1    │ │    GRUNT 2    │ │    GRUNT N    │
│  (parallel)   │ │  (parallel)   │ │  (parallel)   │
│               │ │               │ │               │
│  Tools:       │ │  Tools:       │ │  Tools:       │
│  - bash       │ │  - bash       │ │  - bash       │
│  - read       │ │  - read       │ │  - read       │
│  - write      │ │  - write      │ │  - write      │
│  - search     │ │  - search     │ │  - search     │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   QA AGENT    │ │   QA AGENT    │ │   QA AGENT    │
│  - Validates  │ │  - Validates  │ │  - Validates  │
│  - Feedback   │ │  - Feedback   │ │  - Feedback   │
│  - Retry?     │ │  - Retry?     │ │  - Retry?     │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL ROUTER                           │
│  - Task complexity analysis                                 │
│  - Cost/capability optimization                             │
│  - Auto-escalation on failure                               │
│                                                             │
│  Anthropic: haiku → sonnet → opus                           │
│  OpenAI:    gpt-4o-mini → gpt-4o → o1                       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/noah-ing/Swarm.git
cd Swarm
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Usage

```bash
# Single task
./swarm.py "create a python script that scrapes HN front page"

# Single grunt mode (no orchestration)
./swarm.py "fix the typo in README.md" --single

# Cheap mode (haiku/gpt-4o-mini only)
./swarm.py "list all TODO comments" --cheap

# Interactive mode
./swarm.py -i

# Resume session
./swarm.py --session abc123

# Force provider/model
./swarm.py "build auth system" --prefer anthropic --model opus
```

### CLI Options

| Flag | Description |
|------|-------------|
| `-i, --interactive` | Interactive REPL mode |
| `-s, --single` | Single grunt, no orchestration |
| `-p, --prefer` | Prefer provider (anthropic/openai) |
| `-m, --model` | Force specific model |
| `--cheap` | Use cheapest models |
| `--no-qa` | Skip QA validation |
| `--no-parallel` | Disable parallel execution |
| `-w, --working-dir` | Set working directory |
| `--session` | Resume previous session |
| `-v, --verbose` | Verbose output |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/history` | Show task history |
| `/stats` | Show usage statistics |
| `/clear` | Clear screen |
| `/help` | Show help |
| `quit` | Exit and save session |

## Components

### Orchestrator
Breaks complex tasks into dependency-ordered subtasks. Identifies parallelizable work and coordinates execution waves.

### Grunt Agent
Focused executor with minimal context. Tools:
- `bash` - Shell commands
- `read` - File reading
- `write` - File creation/modification
- `search` - Glob/grep patterns

### QA Agent
Validates grunt output against task requirements. Provides feedback for retry loops. Configurable strictness.

### Model Router
Analyzes task complexity via keyword matching, length, and file references. Routes to appropriate model tier. Auto-escalates on failure.

**Complexity scoring:**
- High: refactor, architect, design, implement, security, performance
- Low: fix, typo, rename, format, lint, simple

## Roadmap

### Memory System
- [ ] Vector store for semantic search over past tasks/solutions
- [ ] Episodic memory: what worked, what failed, in what context
- [ ] Skill library: reusable solution patterns indexed by problem type
- [ ] Cross-session knowledge persistence

### Self-Improvement
- [ ] Prompt mutation based on QA feedback patterns
- [ ] A/B testing of prompt variants with success rate tracking
- [ ] Automatic tool creation from repeated bash patterns
- [ ] Meta-learning: which decomposition strategies work for which task types

### Goal Persistence
- [ ] Long-term objective tracking across sessions
- [ ] Subgoal generation and progress monitoring
- [ ] Automatic task resumption on failure
- [ ] Priority queue with dynamic reordering

### World Model
- [ ] Codebase graph: dependencies, call trees, data flow
- [ ] Effect prediction: what will this change break?
- [ ] Rollback planning: undo strategies before execution
- [ ] Simulation: dry-run changes in sandbox

## Cost

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| haiku | $0.25 | $1.25 |
| sonnet | $3.00 | $15.00 |
| opus | $15.00 | $75.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |

## License

MIT
