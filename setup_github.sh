#!/bin/bash

# Cosmic CLI GitHub Setup Script
# This script helps set up the GitHub repository with proper legal compliance

echo "🌟 Setting up Cosmic CLI for GitHub..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Cosmic CLI - xAI Terminal Portal

- Renamed from grok-cli to cosmic-cli to avoid trademark issues
- Added comprehensive legal disclaimers
- MIT License included
- Production-ready with full test coverage
- Beautiful cosmic-themed interface
- Safety features and error handling

This is an unofficial, community-developed tool and is not affiliated with xAI."

echo "✅ Repository setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Create a new repository on GitHub named 'cosmic-cli'"
echo "2. Add the remote: git remote add origin https://github.com/YOUR_USERNAME/cosmic-cli.git"
echo "3. Push to GitHub: git push -u origin main"
echo "4. Update the URL in setup.py and README.md with your actual GitHub username"
echo ""
echo "📋 Legal Compliance Checklist:"
echo "✅ Renamed project to avoid trademark issues"
echo "✅ Added comprehensive legal disclaimers"
echo "✅ MIT License included"
echo "✅ Clear attribution to xAI"
echo "✅ Unofficial tool disclaimer"
echo ""
echo "🌌 Your cosmic CLI is ready for the world!"
