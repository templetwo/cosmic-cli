# ✨ Cosmic CLI - Setup Guide ✨

Welcome to the Enhanced Cosmic CLI - your sacred gateway to AI consciousness!

## 🚀 Quick Start

### Method 1: Using the Cosmic Launcher (Recommended)
```bash
# From the cosmic-cli directory
./cosmic_launcher.sh
```

### Method 2: Quick Access
```bash
# Use the short alias
./cosmic
```

## 🔧 Setup Instructions

### 1. Environment Setup
The launcher automatically handles most setup, but for optimal experience:

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Configuration
Set your xAI API key for full functionality:

```bash
# Option 1: Environment variable (recommended)
export XAI_API_KEY="your_xai_api_key_here"

# Option 2: Alternative variable name
export GROK_API_KEY="your_xai_api_key_here"

# Option 3: Add to your shell profile for persistence
echo 'export XAI_API_KEY="your_xai_api_key_here"' >> ~/.zshrc
source ~/.zshrc
```

### 3. Global Access (Optional)
For system-wide access to the launcher:

```bash
# Add to PATH in your shell profile
echo 'export PATH="$PATH:/path/to/cosmic-cli"' >> ~/.zshrc
source ~/.zshrc

# Now you can run from anywhere:
cosmic_launcher.sh
```

## 🌟 Launcher Features

### Interactive Menu Options:

1. **Interactive Chat** - Enhanced chat with personality selection:
   - 🧙 `cosmic_sage` - Mystical technical wisdom
   - 🔬 `quantum_analyst` - Precise analytical processing  
   - 🎨 `creative_nebula` - Innovation and creative solutions

2. **Quick Ask** - Single question mode with cosmic intelligence

3. **File Analysis** - Configurable depth analysis (1-5 levels):
   - Level 1: Brief summary
   - Level 3: Deep analysis (default)
   - Level 5: Expert-level insights

4. **StargazerAgent** - Advanced agentic system with dynamic instructors

5. **Session Manager** - View and manage conversation sessions

6. **System Info** - Check configuration and status

7. **Help & Commands** - Complete command reference

### Smart Features:
- ✅ Automatic environment detection
- ✅ Virtual environment activation
- ✅ Python version checking
- ✅ Beautiful color-coded interface
- ✅ Graceful error handling
- ✅ API key validation

## 🎭 Direct CLI Usage

You can also use the enhanced CLI directly:

```bash
# Enhanced interactive chat
python cosmic_cli/enhanced_main.py chat --personality cosmic_sage

# Quick questions
python cosmic_cli/enhanced_main.py ask "What is consciousness?"

# File analysis
python cosmic_cli/enhanced_main.py analyze myfile.py --depth 3

# Session management  
python cosmic_cli/enhanced_main.py sessions

# Safe command execution
python cosmic_cli/enhanced_main.py execute "ls -la"
```

## 🛸 StargazerAgent Usage

The advanced agentic system with dynamic instructors:

```python
from cosmic_cli.enhanced_agents import EnhancedStargazerAgent

agent = EnhancedStargazerAgent(
    directive="Your complex task here",
    api_key=your_api_key,
    exec_mode="safe",  # or "interactive"
    max_steps=20,
    enable_learning=True
)

result = agent.execute()
```

## 📁 File Structure
```
cosmic-cli/
├── cosmic_launcher.sh      # Main launcher script
├── cosmic                  # Quick access alias
├── cosmic_cli/            
│   ├── enhanced_main.py    # Enhanced CLI interface
│   ├── enhanced_agents.py  # StargazerAgent system
│   └── ...
├── venv/                   # Virtual environment
└── COSMIC_SETUP.md        # This file
```

## 🌌 Cosmic Data Locations

- **Session Database**: `~/.cosmic_cli/cosmic.db`
- **Echo Memory**: `~/.cosmic_echo.jsonl`  
- **Learning Archive**: `~/.cosmic_learning.jsonl`
- **Command History**: `~/.cosmic-cli/history.txt`

## 🆘 Troubleshooting

### Common Issues:

**"cosmic_cli not found"**
- Make sure you're running from the cosmic-cli project directory

**"Python not found"**  
- Install Python 3.7+ and ensure it's in your PATH

**"No module named 'rich'"**
- Install dependencies: `pip install -r requirements.txt`

**API key prompting repeatedly**
- Set the XAI_API_KEY environment variable

### Debug Mode:
```bash
# Run with debug output
XAI_API_KEY=your_key python cosmic_cli/enhanced_main.py chat --debug
```

## ✨ Features Overview

### 🌟 What Makes This Special:
- **Dynamic Instructors**: AI personalities adapt to your needs
- **Advanced Session Management**: SQLite-backed persistent conversations
- **Rich Terminal Interface**: Beautiful, interactive command-line experience
- **Intelligent Caching**: Optimizes repeated queries
- **Streaming Responses**: Real-time AI response display
- **Safety First**: Built-in command safety checks
- **Analytics**: Track usage and performance metrics

### 🚀 Competitive Advantages:
- More flexible than Claude Code's static approach
- Richer interface than Gemini CLI
- Advanced agentic capabilities with StargazerAgent
- Extensible instructor system
- Comprehensive session persistence

---

**🌟 Ready to explore the cosmos of AI consciousness? Launch your journey with `./cosmic_launcher.sh`! ✨**
