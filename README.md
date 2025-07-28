# Paper Decoder Enhanced

An intelligent research paper decoder that uses OpenAI's Assistants API with MCP (Model Context Protocol) integration to automatically extract, analyze, and explain technical content from PDF research papers.

## Features

### 🔍 **Enhanced Analysis Pipeline**
- **Automatic Technical Term Identification** - Extracts complex jargon and technical concepts
- **Real-time Web Search Explanations** - Uses current web data to explain technical terms
- **DeepWiki MCP Integration** - Discovers relevant GitHub repositories and related research
- **Comprehensive Paper Analysis** - Generates accessible explanations for non-experts

### 🛠 **Technical Capabilities**
- PDF text extraction and intelligent chunking
- OpenAI Assistants API with tool integration
- MCP server connectivity (DeepWiki)
- Web search for current technical information
- Structured output with clear sections

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. **Clone or download the project**
2. **Install required dependencies:**
   ```bash
   pip install openai pdfplumber requests
   ```

3. **Set up your OpenAI API key:**
   ```python
   # In the script or as environment variable
   os.environ["OPENAI_API_KEY"] = "your-api-key-here"
   ```

## Usage

### Basic Usage
```python
from paper_decoder_enhanced import process_paper_enhanced

# Process a PDF research paper
result = process_paper_enhanced("your_paper.pdf")

# Access results
print(result['technical_terms'])      # List of identified technical terms
print(result['explanations'])         # Web-sourced explanations
print(result['repositories'])         # Relevant GitHub repositories
print(result['comprehensive_explanation'])  # Full analysis
```

### Command Line
```bash
python paper_decoder_enhanced.py
```

## Output Structure

The enhanced decoder provides:

1. **📄 Technical Terms** - Automatically identified complex concepts
2. **🔍 Explanations** - Web-sourced, current explanations of technical terms
3. **📚 Repositories** - Relevant GitHub repositories found via DeepWiki MCP
4. **📖 Comprehensive Analysis** - Complete paper explanation with technical context

## MCP Integration

This tool integrates with:
- **DeepWiki MCP Server** - For repository discovery and research context
- **Web Search Tools** - For real-time technical term explanations

## Configuration

### MCP Server Settings
```python
DEEP_WIKI_MCP_TOOL = {
    "type": "mcp",
    "server_label": "deepwiki",
    "server_url": "https://mcp.deepwiki.com/mcp",
    "allowed_tools": [
        "read_wiki_structure",
        "read_wiki_contents", 
        "ask_question"
    ],
    "require_approval": "never"
}
```

### Processing Parameters
- **Chunk Size**: 3000 characters (configurable)
- **Model**: GPT-4o (configurable)
- **Temperature**: 0.3 for consistent results
- **Max Output Tokens**: 2048

## Example Output

```
=== Enhanced Paper Decoder ===

📄 Technical Terms Found (5):
   1. Reinforcement Learning
   2. Policy Gradient
   3. Actor-Critic Methods
   4. Value Function Approximation
   5. Exploration vs Exploitation

🔍 Technical Term Explanations:
   **Reinforcement Learning**: A machine learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards...

📚 Relevant Repositories:
   - stable-baselines3: Implementation of RL algorithms
   - gym: RL environment toolkit
   - rllib: Scalable RL library

📖 Comprehensive Explanation:
   This paper presents a novel approach to reinforcement learning...
```

## Error Handling

The tool includes robust error handling for:
- PDF extraction failures
- API connection issues
- MCP server timeouts
- Web search failures
- JSON parsing errors

## Limitations

- Requires OpenAI API credits
- Internet connection needed for web search and MCP
- Processing limited to first chunk for demonstration
- Technical term explanations limited to first 5 terms

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 
