# Swarm

A self-improving multi-agent CLI framework with cognitive architecture, prompt evolution, and autonomous learning.

```
   ____
  / ___|_      ____ _ _ __ _ __ ___
  \___ \ \ /\ / / _` | '__| '_ ` _ \
   ___) \ V  V / (_| | |  | | | | | |
  |____/ \_/\_/ \__,_|_|  |_| |_| |_|

  "The swarm that learns"
```

## What Makes Swarm Different

Most AI agent frameworks execute tasks. Swarm **learns from every execution**:

- **Reflects** on outcomes to extract insights
- **Evolves** prompts through mutation and A/B testing
- **Thinks** before acting (meta-cognition)
- **Remembers** solutions for similar future tasks
- **Understands** the codebase it's working on

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SUPERVISOR                              │
│        "CEO" - thinks before acting, learns after            │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ THINKER │  │   BRAIN   │  │EVOLUTION │  │  CODEBASE   │   │
│  │ meta-   │  │ reflect & │  │ mutate   │  │  understand │   │
│  │cognition│  │ learn     │  │ prompts  │  │  context    │   │
│  └─────────┘  └───────────┘  └──────────┘  └─────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                     ORCHESTRATOR                             │
│              smart decomposition + parallel                  │
├──────────────────────────────────────────────────────────────┤
│              GRUNTS ─────────► QA                            │
│            execute tasks      verify                         │
├──────────────────────────────────────────────────────────────┤
│                     MODEL ROUTER                             │
│         haiku → sonnet → opus (auto-escalation)              │
└──────────────────────────────────────────────────────────────┘
```

## Cognitive Systems

### Brain (`brain.py`)
- **Reflection**: Analyzes every task outcome for insights
- **Strategy Selection**: Chooses execution approach based on patterns
- **Uncertainty Quantification**: Knows when confidence is low
- **Skill Extraction**: Auto-creates reusable patterns from successes

### Evolution (`evolution.py`)
- **Prompt Variants**: Tracks multiple versions per agent
- **A/B Testing**: Routes tasks to variants based on fitness
- **Mutation**: Creates new variants (simplify, add_constraint, etc.)
- **Survival**: Retires underperformers, promotes winners

### Thinker (`agents/thinker.py`)
- **Meta-cognition**: Thinks about HOW to approach before acting
- **Risk Assessment**: Identifies what could go wrong
- **Confidence Scoring**: Knows when to ask for help

### Codebase Analyzer (`codebase.py`)
- **Semantic Understanding**: AST analysis of project structure
- **Pattern Detection**: Identifies architectural patterns
- **Context Retrieval**: Provides relevant files for tasks

### Knowledge Transfer (`knowledge.py`)
- **Project Tracking**: Identifies and tracks different codebases
- **Cross-Project Search**: Finds relevant solutions from other projects
- **Universal Insights**: Extracts patterns that work everywhere
- **Transferability Assessment**: Determines which solutions generalize

### Effect Prediction (`effects.py`)
- **Impact Analysis**: Predicts which files will be affected by changes
- **Risk Scoring**: Calculates risk level (low/medium/high/critical)
- **Breaking Change Detection**: Identifies potential breaking changes
- **Safety Suggestions**: Recommends tests to run and precautions

### Rollback System (`rollback.py`)
- **File Snapshots**: Captures file states before changes
- **Automatic Planning**: Creates rollback plans before execution
- **Safe Restoration**: Restores files to previous state on failure
- **Git Integration**: Uses git for rollback when available

### Multi-Agent Negotiation (`negotiation.py`, `agents/critic.py`, `agents/negotiator.py`)
- **Multiple Proposers**: Generate 2-3 solutions from different models
- **Structured Critique**: CriticAgent reviews each solution for issues
- **Consensus Building**: NegotiatorAgent selects or synthesizes best approach
- **Risk-Based Activation**: Auto-triggers on HIGH/CRITICAL risk tasks

### Hierarchical Goal Decomposition (`goals.py`)
- **Recursive Planning**: Breaks complex goals into nested sub-goals
- **Tree Structure**: Goals form a tree, executed bottom-up (leaves first)
- **Depth Limits**: Configurable max depth (default 3) prevents over-decomposition
- **Parallel Execution**: Independent sibling goals run concurrently
- **Result Propagation**: Child results aggregate up to parent goals

### Background Learning (`background.py`)
- **Autonomous Daemon**: Runs in background to continuously learn
- **Skill Practice**: Identifies and practices weak skills
- **Knowledge Consolidation**: Merges and compresses learned patterns
- **Self-Improvement**: Periodically improves Swarm's own code
- **Exploration**: Discovers new patterns from codebases

### Meta-Learning (`metalearner.py`)
- **Learning Efficiency Analysis**: Tracks success rates, token usage, transfer rates
- **Hypothesis Generation**: Identifies improvement opportunities automatically
- **A/B Experiments**: Tests configuration changes with control/treatment
- **Hyperparameter Tuning**: Optimizes mutation rates, thresholds, weights
- **Self-Optimization**: The system that optimizes the optimization

### Agent Factory (`agentfactory.py`)
- **Dynamic Agent Creation**: Spawns new agent types from templates
- **Self-Modifying Architecture**: System creates its own agents
- **Agent Evolution**: Mutates and evolves agents (enhance, specialize, simplify)
- **Performance Tracking**: Monitors and retires underperforming agents
- **Generation Tracking**: Tracks lineage across agent evolution

### Dashboard (`dashboard/`)
- **Real-Time Visualization**: WebSocket-powered live event stream
- **Agent Dialogue View**: See inter-agent messages and negotiations
- **Thought Stream**: Monitor agent reasoning in real-time
- **Task Execution**: Run tasks directly from the browser
- **Stats Dashboard**: Track learning progress and success rates

## Installation

```bash
git clone https://github.com/noah-ing/Swarm.git
cd Swarm
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Usage

### Basic Tasks

```bash
# Simple task (auto-decides strategy)
./swarm.py "count lines of code in all python files"

# Force single grunt (no orchestration)
./swarm.py --single "fix the typo in README.md"

# Streaming output (see agent thinking)
./swarm.py --stream "explain how the router works"

# Cheap mode (haiku/gpt-4o-mini only)
./swarm.py --cheap "list all TODO comments"
```

### Training & Learning

```bash
# Run benchmark tasks to train the system
python3 train.py benchmark --rounds 5 --model haiku

# Let Swarm improve its own code
python3 train.py self-improve --iterations 10

# Autonomous self-directed learning
python3 train.py autonomous --duration 60

# Background learning daemon (continuous improvement)
python3 train.py daemon --duration 60

# Meta-learning optimization (tunes its own parameters)
python3 train.py optimize

# Spawn new dynamic agent types
python3 train.py spawn --type analyzer

# Interactive training mode
python3 train.py interactive

# Check learning progress
python3 train.py stats
```

### Single Task with Cognitive Display

```bash
# See the full cognitive process
python3 run_task.py "your task here" sonnet
```

### Web Dashboard

```bash
# Start the real-time dashboard
python3 -m uvicorn dashboard.server:app --port 8420

# Open http://localhost:8420 in your browser
```

The dashboard provides:
- **Live Activity Stream**: See agent thoughts, messages, and task progress in real-time
- **Agent Dialogue**: Watch inter-agent communication and negotiations
- **Thought Stream**: Monitor agent reasoning and meta-cognition
- **Stats Dashboard**: Track learning progress, success rates, and mutations
- **Task Execution**: Run tasks directly from the browser

## Learning Progress

After training, check what Swarm has learned:

```bash
python3 train.py stats
```

```
============================================================
 SWARM LEARNING STATS
============================================================
 Brain:
   Reflections: 60 (58 successful)
   Skills learned: 0
   Failure patterns tracked: 0
   Avg confidence: 69.7%

 Evolution:
   Total mutations: 3
   grunt: 4 active variants, 73.9% success

 Memory:
   Solutions stored: 11
   Successful solutions: 11
============================================================
```

## Data Persistence

All learning is stored in SQLite databases:

```
~/.swarm/
├── brain.db           # Reflections, insights, skills
├── evolution.db       # Prompt variants, mutations, fitness
├── memory.db          # Past solutions, model performance
├── codebase.db        # Project understanding cache
├── knowledge.db       # Cross-project knowledge transfer
├── embeddings_cache.db # Cached embeddings (saves API calls)
├── rollback.db        # File snapshots for rollback
├── negotiation.db     # Multi-agent debate history
├── goals.db           # Hierarchical goal trees
├── background.db      # Background learning sessions
├── metalearner.db     # Meta-learning experiments & config
└── agentfactory.db    # Dynamic agent blueprints & evolution
```

Learning persists across sessions and accumulates over time.

## CLI Options

| Flag | Description |
|------|-------------|
| `--stream` | Real-time streaming output |
| `-s, --single` | Single grunt, no orchestration |
| `-p, --prefer` | Prefer provider (anthropic/openai) |
| `-m, --model` | Force specific model |
| `--cheap` | Use cheapest models |
| `--no-qa` | Skip QA validation |
| `--no-parallel` | Disable parallel execution |
| `-i, --interactive` | Interactive REPL mode |

## Model Routing

Automatic complexity-based model selection with auto-escalation on failure:

| Complexity | Anthropic | OpenAI |
|------------|-----------|--------|
| Low | haiku | gpt-4o-mini |
| Medium | sonnet | gpt-4o |
| High | opus | o1 |

## Cost Reference

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| haiku | $0.25 | $1.25 |
| sonnet | $3.00 | $15.00 |
| opus | $15.00 | $75.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |

## How Evolution Works

1. **Base prompts** registered for each agent type
2. **Tasks executed** with probabilistic variant selection
3. **Outcomes recorded** (success, tokens, duration)
4. **Fitness calculated** (success rate + efficiency)
5. **Mutations created** from high performers
6. **Poor performers retired** after enough data

Over time, prompts naturally evolve to be more effective.

## Roadmap

### Completed
- [x] Real embedding model for semantic search (OpenAI text-embedding-3-small)
- [x] Cross-project knowledge transfer (universal insights, transferability assessment)
- [x] Effect prediction before changes (risk scoring, impact analysis)
- [x] Rollback planning (file snapshots, safe restoration, git integration)
- [x] Multi-agent negotiation (critique, consensus, risk-based activation)
- [x] Hierarchical goal decomposition (recursive planning, tree execution)
- [x] Continuous background learning (daemon mode, skill practice, consolidation)
- [x] Meta-learning (self-optimization, hypothesis testing, hyperparameter tuning)
- [x] Self-modifying architecture (dynamic agent creation, evolution, retirement)
- [x] Real-time dashboard (WebSocket events, agent dialogue, thought streams)

### Research Directions
- [ ] Multi-modal reasoning (code + diagrams + docs)
- [ ] Federated learning across Swarm instances
- [ ] Neural architecture search for agent topologies

## License

MIT
