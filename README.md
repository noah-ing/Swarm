# Swarm

A multi-agent CLI framework for autonomous task execution with memory, adaptive routing, and intelligent orchestration.

```
   ____
  / ___|_      ____ _ _ __ _ __ ___
  \___ \ \ /\ / / _` | '__| '_ ` _ \
   ___) \ V  V / (_| | |  | | | | | |
  |____/ \_/\_/ \__,_|_|  |_| |_| |_|
```

## Features

- **Smart Orchestration** - Automatically decides between direct execution (single agent) and task decomposition
- **Memory System** - SQLite-backed memory with semantic search for past solutions
- **Adaptive Model Routing** - Learns which models work best for which task types
- **Parallel Execution** - Run independent subtasks concurrently with wave-based scheduling
- **Streaming Output** - Real-time visibility into agent thoughts and tool calls
- **QA Validation** - Automatic quality checks with retry escalation
- **Session Persistence** - Resume work across sessions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER TASK                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                             │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Analyze   │ →  │   Direct?    │ →  │  Decompose   │   │
│  │    Task     │    │  (1 grunt)   │    │ (N grunts)   │   │
│  └─────────────┘    └──────────────┘    └──────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  MEMORY SYSTEM                       │   │
│  │  • Past solutions (semantic search)                  │   │
│  │  • Model performance tracking                        │   │
│  │  • Skill patterns                                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    GRUNT 1    │ │    GRUNT 2    │ │    GRUNT N    │
│  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │
│  │ bash    │  │ │  │ bash    │  │ │  │ bash    │  │
│  │ read    │  │ │  │ read    │  │ │  │ read    │  │
│  │ write   │  │ │  │ write   │  │ │  │ write   │  │
│  │ search  │  │ │  │ search  │  │ │  │ search  │  │
│  └─────────┘  │ │  └─────────┘  │ │  └─────────┘  │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   QA AGENT    │ │   QA AGENT    │ │   QA AGENT    │
│  ✓ approve    │ │  ✓ approve    │ │  ✗ retry      │
│  ✗ retry      │ │  ✗ retry      │ │  → escalate   │
└───────────────┘ └───────────────┘ └───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     MODEL ROUTER                            │
│                                                             │
│  Complexity Analysis → Model Selection → Auto-Escalation    │
│                                                             │
│  Anthropic: haiku ──→ sonnet ──→ opus                       │
│  OpenAI:    gpt-4o-mini ──→ gpt-4o ──→ o1                   │
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
# Simple task (auto-decides: direct or decompose)
./swarm.py "count lines of code in all python files"

# Force single grunt (no orchestration)
./swarm.py --single "fix the typo in README.md"

# Streaming output (see agent thinking in real-time)
./swarm.py --stream "explain how the router works"

# Cheap mode (haiku/gpt-4o-mini only)
./swarm.py --cheap "list all TODO comments"

# Interactive mode
./swarm.py -i

# Force specific model
./swarm.py --model opus "architect a new auth system"

# Resume previous session
./swarm.py --session abc123

# View stats
./swarm.py stats
./swarm.py history
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--stream` | Real-time streaming output |
| `-s, --single` | Single grunt, no orchestration |
| `-p, --prefer` | Prefer provider (anthropic/openai) |
| `-m, --model` | Force specific model |
| `--cheap` | Use cheapest models |
| `--no-qa` | Skip QA validation |
| `--no-parallel` | Disable parallel execution |
| `-w, --working-dir` | Set working directory |
| `--session` | Resume previous session |
| `-i, --interactive` | Interactive REPL mode |
| `-v, --verbose` | Verbose output |

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/history` | Show task history |
| `/stats` | Show usage statistics |
| `/memory` | Show memory and model performance |
| `/clear` | Clear screen |
| `/help` | Show help |
| `quit` | Exit and save session |

## Smart Orchestration

The orchestrator analyzes each task and decides the best execution strategy:

**Direct Execution** (single grunt) for:
- Simple queries and commands
- Single file operations
- Tasks doable in 1-3 tool calls

**Decomposition** (multiple grunts) for:
- Multi-phase tasks
- Coordinated file changes
- Tasks with dependencies

This prevents over-engineering simple tasks while properly handling complex ones.

## Memory System

Swarm remembers past solutions and learns from experience:

```
~/.swarm/memory.db
├── memories      # Past task solutions (semantic searchable)
├── model_perf    # Which models work for which tasks
└── skills        # Reusable solution patterns
```

When you run a similar task, Swarm retrieves relevant past solutions as context.

## Model Routing

Automatic complexity-based model selection:

| Complexity | Anthropic | OpenAI |
|------------|-----------|--------|
| Low | haiku | gpt-4o-mini |
| Medium | sonnet | gpt-4o |
| High | opus | o1 |

On failure, automatically escalates to a more capable model.

## Cost Reference

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| haiku | $0.25 | $1.25 |
| sonnet | $3.00 | $15.00 |
| opus | $15.00 | $75.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |

## Roadmap

### Memory Enhancements
- [ ] Real embedding model integration (OpenAI/Cohere)
- [ ] Cross-project knowledge sharing
- [ ] Automatic skill extraction from successful patterns

### Self-Improvement
- [ ] Prompt mutation based on success rates
- [ ] A/B testing of prompt variants
- [ ] Meta-learning for decomposition strategies

### Goal Persistence
- [ ] Long-term objective tracking
- [ ] Automatic task resumption
- [ ] Priority queue with dynamic reordering

### World Model
- [ ] Codebase dependency graph
- [ ] Effect prediction before changes
- [ ] Rollback planning

## License

MIT
