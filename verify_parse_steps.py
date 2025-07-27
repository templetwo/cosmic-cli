import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to allow importing cosmic_cli
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmic_cli.agents import StargazerAgent

grok_response = """
Here's your cosmic plan:
1. SHELL: ls -l
2. CODE: Create a hello world script
3. INFO: Check system status
"""

try:
    # Call _parse_steps as a static method
    steps = StargazerAgent._parse_steps(grok_response)
    print(f"Successfully parsed steps: {steps}")
    assert len(steps) == 3
    assert "SHELL: ls -l" in steps[0]
    assert "CODE: Create a hello world script" in steps[1]
    assert "INFO: Check system status" in steps[2]
    print("Verification successful: _parse_steps works as expected.")
except AttributeError:
    print("AttributeError: _parse_steps method not found on StargazerAgent.")
except AssertionError:
    print("AssertionError: _parse_steps did not return expected results.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
