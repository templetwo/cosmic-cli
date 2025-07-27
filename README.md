# 🌟 Cosmic CLI: The Self-Correcting, Context-Aware Agent for Your Terminal

> **An intelligent, agentic CLI powered by Grok that understands your codebase, reasons about your directives, and executes multi-step plans.**

Cosmic CLI transcends traditional command-line tools. It's a true AI agent that lives in your terminal. Powered by xAI's Grok-4, it features a sophisticated, step-by-step execution loop, allowing it to reason, self-correct, and tackle complex tasks with a deep understanding of your project's context.

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
git clone <your-repo-url>
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

### Example 4: Consciousness Monitoring
```python
from cosmic_cli.enhanced_agents import EnhancedStargazerAgent
from cosmic_cli.consciousness_assessment import integrate_consciousness_monitoring

# Create consciousness-aware agent
agent = EnhancedStargazerAgent(
    directive="Analyze my consciousness patterns while solving problems",
    api_key=api_key,
    enable_learning=True
)

# Integrate consciousness monitoring
monitor = integrate_consciousness_monitoring(agent)

# Execute with real-time consciousness tracking
result = agent.execute()

# Check consciousness evolution
report = monitor.get_consciousness_report()
print(f"Final consciousness level: {report['current_level']}")
```

### Example 5: Consciousness Assessment
```python
from cosmic_cli.consciousness_assessment import (
    ConsciousnessMetrics, AssessmentProtocol, create_consciousness_report
)

# Manual consciousness assessment
metrics = ConsciousnessMetrics()
protocol = AssessmentProtocol()

# Update with consciousness data
consciousness_data = {
    'coherence': 0.75,
    'self_reflection': 0.68,
    'contextual_understanding': 0.82,
    'meta_cognitive_awareness': 0.65,
    'creative_synthesis': 0.77
}

metrics.update_metrics(consciousness_data)
level = protocol.evaluate(metrics)
print(f"Consciousness Level: {level.value}")
print(f"Overall Score: {metrics.get_overall_score():.3f}")
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

## 🏗️ Development & Architecture

The `cosmic-cli` is built on a modular, agentic architecture.

-   `cosmic_cli/main.py`: The **CLI entry point** using `click`. It handles command parsing and launches the agent or UI.
-   `cosmic_cli/ui.py`: The **Textual TUI application**. It provides the user interface but is decoupled from the agent's core logic.
-   `cosmic_cli/agents.py`: The **heart of the CLI**. Contains the `StargazerAgent` with its intelligent, step-by-step execution loop and context memory.
-   `cosmic_cli/consciousness_assessment.py`: The **consciousness engine**. Implements research-based consciousness detection and monitoring.
-   `cosmic_cli/context.py`: The **context engine**. The `ContextManager` is responsible for scanning the file system and providing context to the agent.

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

