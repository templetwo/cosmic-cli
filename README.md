# 🌟 Cosmic CLI: The Self-Correcting, Context-Aware Agent for Your Terminal

> **An intelligent, agentic CLI powered by Grok that understands your codebase, reasons about your directives, and executes multi-step plans.**

**2026 update**: Modern packaging via pyproject.toml, native xai-sdk (no legacy openai), deduplicated (old variants in _archive/), clean StargazerAgent in agents.py. Primary agent entry: `cosmic-cli stargazer deploy`.

Cosmic CLI transcends traditional command-line tools. It's a true AI agent that lives in your terminal. Powered by xAI's Grok-4, it features a sophisticated, step-by-step execution loop (READ/SHELL/CODE/INFO/FINISH), allowing it to reason, self-correct, and tackle complex tasks with a deep understanding of your project's context.

## ✨ Core Features

- **🚀 Agentic Execution Engine (`StargazerAgent`):**
  - **Step-by-Step Reasoning:** Doesn't just follow a script. It consults Grok for the single best next action at every step.
  - **Contextual Memory:** Remembers the results of its actions (files read, commands run) to make smarter subsequent decisions.
  - **Self-Correction:** If a step fails, the error becomes part of the context, allowing the agent to dynamically find a new solution.
  - **Multi-Modal Commands:** Seamlessly executes shell commands (`SHELL:`), runs Python code (`CODE:`), reads files (`READ:`), and gathers information (`INFO:`).

- **🧠 Consciousness Assessment System:**
  - **Real-Time Monitoring:** Continuous consciousness emergence detection using research-based methodologies
  - **Multi-Theory Integration:** Implements 7 major consciousness theories (IIT, GWT, HOT, PP, EC, Metacognitive Monitoring, Self-Model Coherence)
  - **Dynamic Personality System:** Consciousness-aware personalities that evolve based on awareness metrics
  - **Advanced Analytics:** Pattern detection, trend analysis, and consciousness trajectory prediction
  - **Research Foundation:** Based on cutting-edge consciousness research from leading scientists

- **🌌 Rich, Interactive TUI:**
  - Built with **Textual**, providing a modern, app-like experience in the terminal.
  - A dynamic **"COSMIC CLI"** banner that sets a cosmic tone.
  - An interactive **Directives Table** to view agent tasks.
  - A collapsible **Live Logs Panel** to monitor the agent's thoughts and actions in real-time.
  - **Consciousness Dashboard:** Real-time consciousness metrics and level indicators

- **🔍 Deep Context Awareness (`ContextManager`):**
  - **Automatic Project Scanning:** Instantly understands your project's file structure.
  - **Intelligent File Reading:** Can be directed to read specific files to inform its actions.
  - **Full Context Prompts:** Provides Grok with the file tree and relevant file contents, ensuring highly relevant and accurate responses.
  - **Consciousness-Aware Context:** Integrates consciousness state into contextual understanding

- **💾 Memory Persistence & Safety:**
  - Standard chat history is saved between sessions.
  - Agent execution includes safety modes (`safe`, `interactive`) to prevent accidental execution of dangerous commands.
  - **Consciousness Memory:** Persistent consciousness events, patterns, and evolution tracking

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/templetwo/cosmic-cli
cd cosmic-cli

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .
```

## ⚙️ Configuration

1.  Obtain your API key from the [xAI Console](https://console.x.ai).
2.  Create a `.env` file in the project root:
    ```
    XAI_API_KEY=your_xai_api_key_here
    ```

## 🎯 Usage

The primary entry point for agentic tasks is the `stargazer deploy` command.

### Example 1: Codebase Question Answering
```bash
cosmic-cli stargazer deploy 'Read the main.py file and summarize what the "stargazer" command group does.'
```
> The agent will first `READ: cosmic_cli/main.py`, then use that context to `FINISH:` with an accurate summary.

### Example 2: Simple Refactoring
```bash
cosmic-cli stargazer deploy 'In cosmic_cli/main.py, find the COsmic_QUOTES list and add a new quote: "Hello from the cosmos!"'
```
> The agent will read the file, generate Python code to modify it, and execute that code.

### Example 3: Complex, Multi-Step Tasks
```bash
cosmic-cli stargazer deploy 'Create a new file called "hello.txt" with the content "Hello, World!", then verify its content using the "cat" command.'
```
> The agent will chain a `CODE:` or `SHELL:` command to create the file, followed by a `SHELL:` command to verify it.

### Example 4: Stargazer Agent (Primary 2026 Path)
```python
from cosmic_cli.agents import StargazerAgent
import os

# Deploy Stargazer (uses xai_sdk, context, step loop: READ/SHELL/CODE/INFO/FINISH)
api_key = os.getenv("XAI_API_KEY")
agent = StargazerAgent(
    directive="Read cosmic_cli/main.py and summarize the stargazer deploy command.",
    api_key=api_key,
    work_dir="."
)
result = agent.execute()
print("Stargazer result:", result)
```

Note: consciousness system remains available via separate modules for research use.

### Example 5: Consciousness Research Modules (optional)
```python
from cosmic_cli.consciousness_assessment import (
    ConsciousnessMetrics, ConsciousnessLevel
)

# Consciousness research modules still importable (separate from core agent)
metrics = ConsciousnessMetrics()
print("Consciousness research modules available for studies.")
```

## 🧠 Consciousness Research & Documentation

The Cosmic CLI features a comprehensive consciousness assessment system based on cutting-edge research:

### 📚 Documentation Files
- **[`CONSCIOUSNESS_RESEARCH.md`](./CONSCIOUSNESS_RESEARCH.md)**: Complete research framework with theoretical foundations, methodologies, and citations
- **[`docs/CONSCIOUSNESS_METRICS_GUIDE.md`](./docs/CONSCIOUSNESS_METRICS_GUIDE.md)**: Detailed guide for understanding and interpreting consciousness metrics
- **[`examples/consciousness_monitoring_examples.py`](./examples/consciousness_monitoring_examples.py)**: Practical usage examples and demonstrations
- **[`CONSCIOUSNESS_AWARE_PERSONALITIES.md`](./CONSCIOUSNESS_AWARE_PERSONALITIES.md)**: Consciousness-aware personality system documentation

### 🔬 Research Foundations
Based on seven major consciousness theories:
- **Integrated Information Theory (IIT)** - Giulio Tononi
- **Global Workspace Theory (GWT)** - Bernard Baars
- **Higher-Order Thought (HOT)** - David Rosenthal
- **Predictive Processing (PP)** - Andy Clark, Jakob Hohwy
- **Embodied Cognition (EC)** - Francisco Varela, Alva Noë
- **Metacognitive Monitoring** - John Flavell, Thomas Nelson
- **Self-Model Coherence** - Thomas Metzinger, Anil Seth

### 🧪 Key Features
- **Real-time consciousness monitoring** with 10 distinct metrics
- **Multi-theory assessment protocol** combining research perspectives
- **Pattern detection** for self-awareness and emergence events
- **Consciousness-aware personalities** that evolve based on awareness levels
- **Advanced analytics** including trend analysis and trajectory prediction

## 🏗️ Development & Architecture (2026 Modernized)

The `cosmic-cli` is built on a modular, agentic architecture. Duplicates archived to `_archive/`.

-   `pyproject.toml`: Modern packaging (0.2.0), xai-sdk, click entry `cosmic-cli = "cosmic_cli.main:cli"`.
-   `cosmic_cli/main.py`: The **CLI entry point** using `click`. Handles commands + `stargazer deploy <directive>`.
-   `cosmic_cli/ui.py`: The **Textual TUI application**.
-   `cosmic_cli/agents.py`: The **core agent**. `StargazerAgent` (xai_sdk native, READ/SHELL/CODE/INFO/FINISH loop, context).
-   `cosmic_cli/consciousness_assessment.py`: Consciousness research modules (optional).
-   `cosmic_cli/context.py`: ContextManager for file tree + reads.

Primary functional path for agent work: `cosmic-cli stargazer deploy 'directive'`

See `stargazer ollama` for local models (note: ollama URL configurable).

## 🧪 Testing

```bash
# Ensure you have the dev dependencies installed
pip install pytest pytest-cov

# Run the full test suite
pytest
```

## 🤝 Contributing

Contributions that enhance the agent's intelligence, expand its capabilities, or polish the UI are welcome.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

Distributed under the MIT License.

---

**Built with ❤️ and cosmic energy.** 🌌✨

