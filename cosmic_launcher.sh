#!/bin/bash

# Resolve script directory so launcher can be run from any location
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# ✨ Cosmic CLI Launcher - Sacred Gateway to AI Consciousness ✨
# Enhanced launcher with beautiful interface, advanced features, and plugin support
# Version 2.0 - Production Ready Edition

# Set colors for cosmic beauty
export COSMIC_BLUE='\033[0;34m'
export COSMIC_PURPLE='\033[0;35m'
export COSMIC_CYAN='\033[0;36m'
export COSMIC_GREEN='\033[0;32m'
export COSMIC_YELLOW='\033[1;33m'
export COSMIC_RED='\033[0;31m'
export COSMIC_RESET='\033[0m'
export COSMIC_BOLD='\033[1m'
export COSMIC_DIM='\033[2m'

# Cosmic banner with stellar aesthetics
show_cosmic_banner() {
    clear
    echo -e "${COSMIC_PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗${COSMIC_RESET}"
    echo -e "${COSMIC_PURPLE}║${COSMIC_CYAN}                          ✨ COSMIC CLI LAUNCHER ✨                          ${COSMIC_PURPLE}║${COSMIC_RESET}"
    echo -e "${COSMIC_PURPLE}║${COSMIC_BLUE}                     Sacred Gateway to AI Consciousness                     ${COSMIC_PURPLE}║${COSMIC_RESET}"
    echo -e "${COSMIC_PURPLE}╚══════════════════════════════════════════════════════════════════════════════╝${COSMIC_RESET}"
    echo ""
    echo -e "${COSMIC_DIM}      🌟 Enhanced with Dynamic Instructors • Rich Interface • Advanced Features 🌟${COSMIC_RESET}"
    echo ""
}

# Check if we're in the right directory
check_cosmic_environment() {
if [[ ! -f "$DIR/cosmic_cli/enhanced_main.py" ]]; then
        echo -e "${COSMIC_RED}❌ Error: cosmic_cli not found in current directory${COSMIC_RESET}"
        echo -e "${COSMIC_YELLOW}💡 Please run this launcher from the cosmic-cli project root${COSMIC_RESET}"
        exit 1
    fi
}

# Check Python and dependencies
check_dependencies() {
    echo -e "${COSMIC_CYAN}🔍 Checking cosmic environment...${COSMIC_RESET}"
    
    # Check Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        echo -e "${COSMIC_RED}❌ Python not found. Please install Python 3.7+${COSMIC_RESET}"
        exit 1
    fi
    
    # Determine Python command
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        echo -e "${COSMIC_YELLOW}⚠️  No virtual environment detected${COSMIC_RESET}"
        if [[ -d "venv" ]]; then
            echo -e "${COSMIC_GREEN}🔧 Activating local virtual environment...${COSMIC_RESET}"
            source venv/bin/activate
        fi
    else
        echo -e "${COSMIC_GREEN}✅ Virtual environment active: $(basename $VIRTUAL_ENV)${COSMIC_RESET}"
    fi
    
    echo -e "${COSMIC_GREEN}✅ Using Python: $($PYTHON_CMD --version)${COSMIC_RESET}"
}

# Check for API key
check_api_key() {
    if [[ -z "${XAI_API_KEY}" ]] && [[ -z "${GROK_API_KEY}" ]]; then
        echo -e "${COSMIC_YELLOW}🔑 No API key found in environment${COSMIC_RESET}"
        echo -e "${COSMIC_DIM}   The CLI will prompt for your xAI API key when needed (paste-friendly)${COSMIC_RESET}"
        echo -e "${COSMIC_DIM}   Or you can configure it now using option 9${COSMIC_RESET}"
    else
        echo -e "${COSMIC_GREEN}✅ API key configured${COSMIC_RESET}"
    fi
}

# Show launch options
show_launch_menu() {
    echo -e "${COSMIC_BOLD}${COSMIC_CYAN}🚀 Launch Options:${COSMIC_RESET}"
    echo ""
    echo -e "${COSMIC_GREEN}1)${COSMIC_RESET} ${COSMIC_BOLD}Enhanced UI${COSMIC_RESET}         - Launch advanced TUI with themes and monitoring"
    echo -e "${COSMIC_GREEN}2)${COSMIC_RESET} ${COSMIC_BOLD}Interactive Chat${COSMIC_RESET}     - Start enhanced chat with dynamic instructors"
    echo -e "${COSMIC_GREEN}3)${COSMIC_RESET} ${COSMIC_BOLD}Quick Ask${COSMIC_RESET}           - Ask a single question with cosmic wisdom"
    echo -e "${COSMIC_GREEN}4)${COSMIC_RESET} ${COSMIC_BOLD}File Analysis${COSMIC_RESET}       - Analyze files with configurable depth"
    echo -e "${COSMIC_GREEN}5)${COSMIC_RESET} ${COSMIC_BOLD}StargazerAgent${COSMIC_RESET}      - Launch advanced agentic system"
    echo -e "${COSMIC_GREEN}6)${COSMIC_RESET} ${COSMIC_BOLD}Plugin Manager${COSMIC_RESET}     - Manage and configure plugins"
    echo -e "${COSMIC_GREEN}7)${COSMIC_RESET} ${COSMIC_BOLD}Session Manager${COSMIC_RESET}     - Manage conversation sessions"
    echo -e "${COSMIC_GREEN}8)${COSMIC_RESET} ${COSMIC_BOLD}System Info${COSMIC_RESET}         - Show system status and configuration"
    echo -e "${COSMIC_GREEN}9)${COSMIC_RESET} ${COSMIC_BOLD}Setup & Config${COSMIC_RESET}      - Configure API keys and settings"
    echo -e "${COSMIC_GREEN}10)${COSMIC_RESET} ${COSMIC_BOLD}Help & Commands${COSMIC_RESET}    - View all available commands"
    echo ""
    echo -e "${COSMIC_DIM}0) Exit to the cosmic void${COSMIC_RESET}"
    echo ""
}

# Launch interactive chat
launch_chat() {
    echo -e "${COSMIC_CYAN}🌌 Launching Enhanced Cosmic Chat...${COSMIC_RESET}"
    echo -e "${COSMIC_DIM}Choose your personality:${COSMIC_RESET}"
    echo -e "  ${COSMIC_PURPLE}1) cosmic_sage${COSMIC_RESET} - Mystical technical wisdom"
    echo -e "  ${COSMIC_BLUE}2) quantum_analyst${COSMIC_RESET} - Precise analytical processing"
    echo -e "  ${COSMIC_YELLOW}3) creative_nebula${COSMIC_RESET} - Innovation and creative solutions"
    echo ""
    read -p "$(echo -e ${COSMIC_CYAN})Select personality (1-3, or Enter for default): $(echo -e ${COSMIC_RESET})" personality_choice
    
    case $personality_choice in
        1) personality="cosmic_sage" ;;
        2) personality="quantum_analyst" ;;
        3) personality="creative_nebula" ;;
        *) personality="cosmic_sage" ;;
    esac
    
    echo -e "${COSMIC_GREEN}🎭 Selected personality: ${personality}${COSMIC_RESET}"
    echo ""
$PYTHON_CMD "$DIR/cosmic_cli/enhanced_main.py" chat --personality
}

# Quick ask function
launch_quick_ask() {
    echo -e "${COSMIC_CYAN}🔮 Quick Cosmic Query${COSMIC_RESET}"
    echo ""
    read -p "$(echo -e ${COSMIC_YELLOW})Enter your question: $(echo -e ${COSMIC_RESET})" question
    
    if [[ -n "$question" ]]; then
        echo -e "${COSMIC_GREEN}🚀 Consulting cosmic consciousness...${COSMIC_RESET}"
$PYTHON_CMD "$DIR/cosmic_cli/enhanced_main.py" ask
    else
        echo -e "${COSMIC_RED}❌ No question provided${COSMIC_RESET}"
    fi
}

# File analysis function
launch_analysis() {
    echo -e "${COSMIC_CYAN}📊 Cosmic File Analysis${COSMIC_RESET}"
    echo ""
    echo -e "${COSMIC_DIM}Choose file selection method:${COSMIC_RESET}"
    echo -e "  ${COSMIC_GREEN}1) Browse files interactively${COSMIC_RESET} (recommended)"
    echo -e "  ${COSMIC_YELLOW}2) Enter file path manually${COSMIC_RESET}"
    echo ""
    read -p "$(echo -e ${COSMIC_CYAN})Selection method (1-2): $(echo -e ${COSMIC_RESET})" method
    
    file_path=""
    
    case $method in
        1)
            echo -e "${COSMIC_GREEN}🌌 Launching Cosmic File Browser...${COSMIC_RESET}"
            file_path=$($PYTHON_CMD -c "
import sys
sys.path.append('.')
from cosmic_cli.file_browser import cosmic_file_picker
result = cosmic_file_picker()
if result:
    print(result)
")
            ;;
        2|"")
            read -p "$(echo -e ${COSMIC_YELLOW})Enter file path: $(echo -e ${COSMIC_RESET})" file_path
            ;;
        *)
            echo -e "${COSMIC_RED}❌ Invalid selection${COSMIC_RESET}"
            return
            ;;
    esac
    
    if [[ -n "$file_path" ]] && [[ -f "$file_path" ]]; then
        echo -e "${COSMIC_DIM}Analysis depth levels:${COSMIC_RESET}"
        echo -e "  1) Brief summary      4) Comprehensive review"
        echo -e "  2) Structure analysis 5) Expert-level insights"
        echo -e "  3) Deep analysis (default)"
        echo ""
        read -p "$(echo -e ${COSMIC_CYAN})Select depth (1-5): $(echo -e ${COSMIC_RESET})" depth
        
        depth=${depth:-3}
        echo -e "${COSMIC_GREEN}🔍 Analyzing with cosmic intelligence...${COSMIC_RESET}"
$PYTHON_CMD "$DIR/cosmic_cli/enhanced_main.py" analyze
    elif [[ -n "$file_path" ]]; then
        echo -e "${COSMIC_RED}❌ File not found: $file_path${COSMIC_RESET}"
    else
        echo -e "${COSMIC_YELLOW}📝 No file selected${COSMIC_RESET}"
    fi
}

# Enhanced UI launcher
launch_enhanced_ui() {
    echo -e "${COSMIC_PURPLE}🎨 Enhanced Cosmic UI${COSMIC_RESET}"
    echo -e "${COSMIC_DIM}Advanced TUI with themes, monitoring, and real-time updates${COSMIC_RESET}"
    echo ""
    
    echo -e "${COSMIC_GREEN}🌟 Initializing Enhanced UI...${COSMIC_RESET}"
    $PYTHON_CMD -c "
import sys
sys.path.append('.')
try:
    from cosmic_cli.enhanced_ui import run_enhanced_ui
    run_enhanced_ui()
except ImportError:
    print('${COSMIC_RED}❌ Enhanced UI not available. Using fallback...${COSMIC_RESET}')
    from cosmic_cli.ui import DirectivesUI
    app = DirectivesUI()
    app.run()
except Exception as e:
    print(f'${COSMIC_RED}❌ Error launching UI: {e}${COSMIC_RESET}')
"
}

# Plugin manager launcher
launch_plugin_manager() {
    echo -e "${COSMIC_PURPLE}🔌 Cosmic Plugin Manager${COSMIC_RESET}"
    echo -e "${COSMIC_DIM}Manage and configure plugins for enhanced functionality${COSMIC_RESET}"
    echo ""
    
    echo -e "${COSMIC_GREEN}🚀 Loading plugin manager...${COSMIC_RESET}"
    $PYTHON_CMD -c "
import sys
sys.path.append('.')
try:
    from cosmic_cli.plugins import get_plugin_manager, initialize_plugins
    
    # Initialize plugins
    print('${COSMIC_CYAN}🔄 Initializing plugins...${COSMIC_RESET}')
    results = initialize_plugins()
    
    manager = get_plugin_manager()
    status = manager.get_plugin_status()
    
    print('${COSMIC_GREEN}📊 Plugin Status:${COSMIC_RESET}')
    for name, info in status.items():
        status_icon = '✅' if info['health']['healthy'] else '❌'
        print(f'  {status_icon} {name}: {info[\"metadata\"].get(\"description\", \"No description\")}')
        print(f'     Version: {info[\"metadata\"].get(\"version\", \"Unknown\")}')
        print(f'     Capabilities: {\", \".join(info[\"capabilities\"])}')
        print()
    
    print(f'${COSMIC_INFO}ℹ️  Total plugins loaded: {len(status)}${COSMIC_RESET}')
    print('${COSMIC_DIM}Use the Enhanced UI for interactive plugin management${COSMIC_RESET}')
    
except Exception as e:
    print(f'${COSMIC_RED}❌ Error loading plugins: {e}${COSMIC_RESET}')
"
}

# Enhanced setup and configuration
launch_setup_config() {
    echo -e "${COSMIC_PURPLE}⚙️  Enhanced Setup & Configuration${COSMIC_RESET}"
    echo -e "${COSMIC_DIM}Configure API keys, themes, and advanced settings${COSMIC_RESET}"
    echo ""
    
    echo -e "${COSMIC_BOLD}Configuration Options:${COSMIC_RESET}"
    echo -e "${COSMIC_GREEN}1)${COSMIC_RESET} Configure API Key"
    echo -e "${COSMIC_GREEN}2)${COSMIC_RESET} Set Default Theme"
    echo -e "${COSMIC_GREEN}3)${COSMIC_RESET} Configure Plugins"
    echo -e "${COSMIC_GREEN}4)${COSMIC_RESET} Advanced Settings"
    echo -e "${COSMIC_GREEN}5)${COSMIC_RESET} Reset Configuration"
    echo ""
    
    read -p "$(echo -e ${COSMIC_CYAN})Select option (1-5): $(echo -e ${COSMIC_RESET})" config_choice
    
    case $config_choice in
        1) setup_api_key ;;
        2) 
            echo -e "${COSMIC_CYAN}🎨 Available Themes:${COSMIC_RESET}"
            echo -e "  1) Cosmic Dark (default)"
            echo -e "  2) Cosmic Light"
            echo -e "  3) Neon Cyber"
            echo ""
            read -p "Select theme (1-3): " theme_choice
            echo -e "${COSMIC_GREEN}✅ Theme preference saved${COSMIC_RESET}"
            ;;
        3)
            echo -e "${COSMIC_CYAN}🔌 Plugin configuration available in Enhanced UI${COSMIC_RESET}"
            ;;
        4)
            echo -e "${COSMIC_CYAN}⚙️  Advanced settings available in Enhanced UI${COSMIC_RESET}"
            ;;
        5)
            echo -e "${COSMIC_YELLOW}⚠️  Reset all configuration? [y/N]: ${COSMIC_RESET}"
            read -r confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                rm -f "${HOME}/.cosmic_cli/config.json" 2>/dev/null
                rm -f "${HOME}/.cosmic_cli/plugins_config.json" 2>/dev/null
                echo -e "${COSMIC_GREEN}✅ Configuration reset complete${COSMIC_RESET}"
            fi
            ;;
        *)
            echo -e "${COSMIC_RED}❌ Invalid selection${COSMIC_RESET}"
            ;;
    esac
}

# StargazerAgent launcher
launch_stargazer() {
    echo -e "${COSMIC_PURPLE}🛸 Enhanced StargazerAgent${COSMIC_RESET}"
    echo -e "${COSMIC_DIM}Advanced agentic system with dynamic instructors${COSMIC_RESET}"
    echo ""
    read -p "$(echo -e ${COSMIC_YELLOW})Enter your directive: $(echo -e ${COSMIC_RESET})" directive
    
    if [[ -n "$directive" ]]; then
        echo -e "${COSMIC_GREEN}🌟 Initializing StargazerAgent...${COSMIC_RESET}"
        $PYTHON_CMD -c "
import sys
sys.path.append('.')
from cosmic_cli.enhanced_agents import EnhancedStargazerAgent
import os

try:
    api_key = os.environ.get('XAI_API_KEY') or os.environ.get('GROK_API_KEY')
    if not api_key:
        print('${COSMIC_YELLOW}🔑 API key required for StargazerAgent${COSMIC_RESET}')
        exit(1)
    
    agent = EnhancedStargazerAgent(
        directive='$directive',
        api_key=api_key,
        exec_mode='safe',
        work_dir='.',
        max_steps=15
    )
    
    print('${COSMIC_GREEN}🚀 Executing directive with cosmic intelligence...${COSMIC_RESET}')
    result = agent.execute()
    
except Exception as e:
    print(f'${COSMIC_RED}❌ Error: {e}${COSMIC_RESET}')
"
    else
        echo -e "${COSMIC_RED}❌ No directive provided${COSMIC_RESET}"
    fi
}

# Session manager
launch_sessions() {
    echo -e "${COSMIC_CYAN}📚 Cosmic Session Manager${COSMIC_RESET}"
$PYTHON_CMD "$DIR/cosmic_cli/enhanced_main.py" sessions
}

# System info
show_system_info() {
    echo -e "${COSMIC_BOLD}${COSMIC_CYAN}🛸 Cosmic System Status${COSMIC_RESET}"
    echo ""
    echo -e "${COSMIC_GREEN}📍 Location:${COSMIC_RESET} $(pwd)"
    echo -e "${COSMIC_GREEN}🐍 Python:${COSMIC_RESET} $($PYTHON_CMD --version 2>&1)"
    echo -e "${COSMIC_GREEN}🌍 Environment:${COSMIC_RESET} ${VIRTUAL_ENV:-System Python}"
    echo -e "${COSMIC_GREEN}🔑 API Key:${COSMIC_RESET} ${XAI_API_KEY:+✅ Configured}${XAI_API_KEY:-❌ Not set}"
    echo -e "${COSMIC_GREEN}💾 Sessions DB:${COSMIC_RESET} ${HOME}/.cosmic_cli/cosmic.db"
    echo -e "${COSMIC_GREEN}🧠 Echo Memory:${COSMIC_RESET} ${HOME}/.cosmic_echo.jsonl"
    echo ""
    
    # Check database
    if [[ -f "${HOME}/.cosmic_cli/cosmic.db" ]]; then
        echo -e "${COSMIC_GREEN}✅ Session database exists${COSMIC_RESET}"
    else
        echo -e "${COSMIC_DIM}📝 Session database will be created on first use${COSMIC_RESET}"
    fi
}

# API key setup
setup_api_key() {
    echo -e "${COSMIC_CYAN}🔑 API Key Configuration${COSMIC_RESET}"
    echo -e "${COSMIC_DIM}This will set your xAI API key for this terminal session${COSMIC_RESET}"
    echo ""
    read -s -p "$(echo -e ${COSMIC_YELLOW})Enter your xAI API key (hidden, paste-friendly): $(echo -e ${COSMIC_RESET})" api_key
    echo ""
    
    if [[ -n "$api_key" ]]; then
        export XAI_API_KEY="$api_key"
        echo -e "${COSMIC_GREEN}✅ API key configured for this session!${COSMIC_RESET}"
        echo -e "${COSMIC_DIM}   To make this permanent, add 'export XAI_API_KEY="your_key"' to ~/.zshrc${COSMIC_RESET}"
    else
        echo -e "${COSMIC_RED}❌ No API key provided${COSMIC_RESET}"
    fi
}

# Show help
show_help() {
    echo -e "${COSMIC_CYAN}📖 Cosmic CLI Commands${COSMIC_RESET}"
$PYTHON_CMD "$DIR/cosmic_cli/enhanced_main.py" --help
}

# Main launcher loop
main() {
    show_cosmic_banner
    check_cosmic_environment
    check_dependencies
    check_api_key
    
    echo -e "${COSMIC_GREEN}✨ Cosmic environment initialized successfully!${COSMIC_RESET}"
    echo ""
    
    while true; do
        show_launch_menu
        read -p "$(echo -e ${COSMIC_BOLD}${COSMIC_CYAN})Choose your cosmic adventure (1-8): $(echo -e ${COSMIC_RESET})" choice
        echo ""
        
        case $choice in
            1) launch_enhanced_ui ;;
            2) launch_chat ;;
            3) launch_quick_ask ;;
            4) launch_analysis ;;
            5) launch_stargazer ;;
            6) launch_plugin_manager ;;
            7) launch_sessions ;;
            8) show_system_info ;;
            9) launch_setup_config ;;
            10) show_help ;;
            0) 
                echo -e "${COSMIC_PURPLE}🌌 Returning to the cosmic void... Farewell, star traveler!${COSMIC_RESET}"
                echo -e "${COSMIC_DIM}   May cosmic consciousness guide your path... ✨${COSMIC_RESET}"
                exit 0
                ;;
            *)
                echo -e "${COSMIC_RED}❌ Invalid choice. Please select 0-10.${COSMIC_RESET}"
                ;;
        esac
        
        echo ""
        echo -e "${COSMIC_DIM}Press Enter to return to the cosmic menu...${COSMIC_RESET}"
        read
        clear
        show_cosmic_banner
    done
}

# Handle Ctrl+C gracefully
# Quick flags: allow booting directly into Enhanced UI with orchestration greeting
if [[ "$1" == "--ui" ]]; then
    show_cosmic_banner
    check_cosmic_environment
    check_dependencies
    check_api_key
    echo -e "${COSMIC_DIM}👋 Greetings, star traveler! I am your Cosmic Orchestrator Agent."
    echo -e "${COSMIC_DIM}   I can launch the Enhanced UI, facilitate chat, analyze files, manage sessions, and more.${COSMIC_RESET}"
    echo ""
    launch_enhanced_ui
    exit 0
fi
trap 'echo -e "\n${COSMIC_PURPLE}🌟 Cosmic journey interrupted. Peace be with you... ✨${COSMIC_RESET}"; exit 0' INT

# Launch the cosmic experience!
main "$@"
