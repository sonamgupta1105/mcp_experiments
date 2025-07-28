import os
import openai
import json
import requests
from typing import List, Dict, Any, Optional

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = openai.OpenAI()

import pdfplumber

# Tool definitions
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

# Web search tool for technical term explanations
WEB_SEARCH_TOOL = {
    "type": "web_search"
}

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, max_length=3000):
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(' '.join(chunk)) > max_length:
            chunks.append(' '.join(chunk))
            chunk = []
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def call_openai_with_tools(prompt, model="gpt-4o", use_deepwiki=False, use_web_search=False):
    """
    Call OpenAI API with DeepWiki MCP and/or web search tools
    """
    input_messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    params = {
        "model": model,
        "input": input_messages,
        "text": {"format": {"type": "text"}},
        "reasoning": {},
        "temperature": 0.3,
        "max_output_tokens": 2048,
        "top_p": 1,
        "store": True
    }
    
    tools = []
    if use_deepwiki:
        tools.append(DEEP_WIKI_MCP_TOOL)
    if use_web_search:
        tools.append(WEB_SEARCH_TOOL)
    
    if tools:
        params["tools"] = tools
    
    print('DEBUG: About to call OpenAI responses API')
    response = client.responses.create(**params)
    print('DEBUG: Finished OpenAI responses API call')
    
    # Extract the text from the output field
    try:
        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'content') and output_item.content:
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text') and content_item.text:
                            return content_item.text
        return "[No output from model]"
    except Exception as e:
        return f"Response error: {str(e)}"

def identify_technical_terms(chunk):
    """Identify technical terms that need explanation using web search"""
    prompt = f"""
    Extract unique technical terms, concepts, and jargon from this research paper text that a non-expert might not understand.
    
    IMPORTANT GUIDELINES:
    1. Group similar concepts together (e.g., "agentic systems", "agentic AI systems", "agentic artificial intelligence" should be consolidated as "agentic systems")
    2. Avoid JSON formatting artifacts like "```json" or "```"
    3. Focus on distinct, meaningful technical concepts
    4. Return ONLY a JSON array of strings with the technical terms, nothing else.
    
    Example format: ["term1", "term2", "term3"]
    
    Text: {chunk}
    """
    response = call_openai_with_tools(prompt, use_web_search=True)
    
    # Parse JSON response
    try:
        response_str = response.strip()
        if response_str.startswith('[') and response_str.endswith(']'):
            terms = json.loads(response_str)
            if isinstance(terms, list):
                # Filter out JSON artifacts and clean terms
                cleaned_terms = []
                for term in terms:
                    if term and isinstance(term, str):
                        # Remove JSON formatting artifacts
                        clean_term = term.replace('```json', '').replace('```', '').strip()
                        if clean_term and len(clean_term) > 2 and not clean_term.startswith('```'):
                            cleaned_terms.append(clean_term)
                
                # Deduplicate similar terms
                deduplicated_terms = []
                for term in cleaned_terms:
                    # Check if this term is too similar to existing ones
                    is_duplicate = False
                    for existing_term in deduplicated_terms:
                        # Simple similarity check - if one term contains the other, keep the more specific one
                        if term.lower() in existing_term.lower() or existing_term.lower() in term.lower():
                            # Keep the longer/more specific term
                            if len(term) > len(existing_term):
                                deduplicated_terms.remove(existing_term)
                                deduplicated_terms.append(term)
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        deduplicated_terms.append(term)
                
                return deduplicated_terms[:8]  # Limit to 8 most important terms
    except Exception:
        pass
    
    # Fallback: extract terms from text
    import re
    lines = response.split('\n')
    terms = [line.strip().strip('-*•').strip() for line in lines if line.strip() and len(line.strip()) > 2]
    return terms[:8]

def explain_technical_term_with_web_search(term):
    """Explain a technical term using web search"""
    prompt = f"""
    Search the web for information about '{term}' and provide a clear, concise explanation suitable for a non-expert audience.
    Focus on:
    1. A simple definition
    2. Why it's important in the context
    3. A practical example if applicable
    
    Keep the explanation under 100 words.
    """
    return call_openai_with_tools(prompt, use_web_search=True)

def find_relevant_repositories(chunk):
    """Find relevant repositories using DeepWiki based on the research paper content"""
    prompt = f"""
    Based on this research paper text, identify relevant GitHub repositories that might be related to the topics, 
    technologies, or methods discussed. Use DeepWiki to search for repositories that could be useful for:
    1. Implementation of the methods described
    2. Related research or similar approaches
    3. Tools and libraries mentioned
    4. Datasets or benchmarks referenced
    
    Research paper text: {chunk[:1000]}  # Use first 1000 chars for repository search
    
    Provide a list of relevant repositories with brief descriptions of why they're relevant.
    """
    return call_openai_with_tools(prompt, use_deepwiki=True)

def explain_paper_with_enhanced_tools(chunk):
    """
    Explain a research paper chunk using web search for technical terms and DeepWiki for repositories
    """
    prompt = f"""
    Explain this research paper text in a clear, accessible way. When you encounter technical terms, 
    jargon, or complex concepts, use web search to provide accurate, detailed explanations.
    
    Focus on:
    1. Summarizing the main points clearly
    2. Explaining technical terms using web search when needed
    3. Making the content accessible to non-experts
    4. Highlighting the significance and impact
    
    Research paper text:
    {chunk}
    
    Remember to use web search for any technical terms, scientific concepts, or jargon 
    that would benefit from detailed explanation.
    """
    return call_openai_with_tools(prompt, use_web_search=True)

def process_paper_enhanced(file_path):
    """
    Enhanced workflow: Extract text, identify terms, explain with web search, find repositories with DeepWiki
    """
    # Extract and chunk text
    full_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(full_text)
    
    # Process first chunk for demonstration
    chunk = chunks[0]
    
    # Step 1: Identify technical terms
    technical_terms = identify_technical_terms(chunk)
    
    # Step 2: Explain technical terms using web search
    explanations = {}
    for term in technical_terms[:5]:  # Limit to first 5 terms
        explanation = explain_technical_term_with_web_search(term)
        explanations[term] = explanation
    
    # Step 3: Find relevant repositories using DeepWiki
    repositories = find_relevant_repositories(chunk)
    
    # Step 4: Create comprehensive explanation
    comprehensive_explanation = explain_paper_with_enhanced_tools(chunk)
    
    return {
        "technical_terms": technical_terms,
        "explanations": explanations,
        "repositories": repositories,
        "comprehensive_explanation": comprehensive_explanation,
        "chunk_processed": chunk[:200] + "..." if len(chunk) > 200 else chunk
    }

def main():
    """Main function to run the enhanced paper decoder"""
    file_path = "SLMs.pdf"
    
    try:
        result = process_paper_enhanced(file_path)
        
        print("\n" + "="*50)
        print("ENHANCED PAPER DECODER RESULTS")
        print("="*50)
        
        print(f"\n📄 Technical Terms Found ({len(result['technical_terms'])}):")
        for i, term in enumerate(result['technical_terms'], 1):
            print(f"   {i}. {term}")
        
        print(f"\n🔍 Technical Term Explanations:")
        for term, explanation in result['explanations'].items():
            print(f"\n   **{term}**:")
            print(f"   {explanation}")
        
        print(f"\n📚 Relevant Repositories:")
        print(result['repositories'])
        
        print(f"\n📖 Comprehensive Explanation:")
        print(result['comprehensive_explanation'])
        
    except Exception as e:
        print(f"Error processing paper: {str(e)}")

if __name__ == "__main__":
    main() 
