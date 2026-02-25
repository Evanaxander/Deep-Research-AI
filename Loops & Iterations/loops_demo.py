#!/usr/bin/env python3
"""
Task 6: Loops & Iterations Demo
Shows how to create iterative refinement loops in LangGraph
"""

import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

print("\n🔄 LOOPS & ITERATIONS DEMO")
print("=" * 40)

# Initialize LLM with environment variables
llm = ChatOllama(
    model="llama3",
    temperature=0,
)

# Define state structure
class ResearchState(TypedDict):
    query: str
    search_results: str
    quality_score: float
    iteration: int
    max_iterations: int
    final_answer: str

def search_node(state: ResearchState):
    """Simulate search with progressive improvement"""
    print(f"\n🔍 Search Node (Iteration {state['iteration'] + 1})")
    
    query = state["query"]
    iteration = state["iteration"]
    
    # Simulate search getting better with iterations
    if iteration == 0:
        results = f"Basic results for: {query}"
        print(f"   → Found basic information")
    elif iteration == 1:
        results = f"Detailed results for: {query} with more context"
        print(f"   → Found detailed information")
    else:
        results = f"Comprehensive results for: {query} with citations and examples"
        print(f"   → Found comprehensive information")
    
    return {
        "search_results": results,
        "iteration": iteration + 1
    }

def evaluate_node(state: ResearchState):
    """Evaluate search quality"""
    print(f"\n📊 Evaluate Node")
    
    results = state["search_results"]
    iteration = state["iteration"]
    
    # Quality improves with iterations
    if "comprehensive" in results.lower():
        score = 0.9
        print(f"   → Quality: Excellent (score: {score})")
    elif "detailed" in results.lower():
        score = 0.6
        print(f"   → Quality: Good (score: {score})")
    else:
        score = 0.3
        print(f"   → Quality: Needs improvement (score: {score})")
    
    return {"quality_score": score}

def summarize_node(state: ResearchState):
    """Generate final answer from accumulated results"""
    print(f"\n✨ Summarize Node")
    
    # Use LLM to generate summary
    prompt = f"""Based on these search results, provide a concise answer:
    Query: {state['query']}
    Results: {state['search_results']}
    
    Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content
    
    print(f"   → Generated final answer")
    
    return {"final_answer": answer}

def should_continue(state: ResearchState) -> Literal["search", "summarize"]:
    """Router function: decides whether to loop or finish"""
    print(f"\n🚦 Router Decision:")
    
    # Check if we've reached max iterations
    if state["iteration"] >= state["max_iterations"]:
        print(f"   → Max iterations ({state['max_iterations']}) reached - exiting loop")
        return "summarize"
    
    # Check if quality is sufficient
    if state["quality_score"] >= 0.8:
        print(f"   → Quality sufficient ({state['quality_score']:.1f}) - exiting loop")
        return "summarize"
    
    print(f"   → Quality insufficient ({state['quality_score']:.1f}) - continuing loop")
    return "search"

# Build the workflow
print("\n🏗️ Building workflow with loop...")
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("search", search_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("summarize", summarize_node)

# Define the flow
workflow.add_edge(START, "search")
workflow.add_edge("search", "evaluate")

# Add conditional edge for the loop
workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "search": "search",      # Loop back to search
        "summarize": "summarize"  # Exit loop to summarize
    }
)

workflow.add_edge("summarize", END)

# Compile the graph
app = workflow.compile()

# Test the loop with different queries
print("\n" + "=" * 40)
print("🚀 RUNNING LOOP TESTS")
print("=" * 40)

test_queries = [
    "What is LangGraph?",
    "How do loops work in workflows?"
]

for query in test_queries:
    print(f"\n📝 Testing query: '{query}'")
    print("-" * 40)
    
    initial_state = {
        "query": query,
        "search_results": "",
        "quality_score": 0.0,
        "iteration": 0,
        "max_iterations": 3,
        "final_answer": ""
    }
    
    result = app.invoke(initial_state)
    
    print(f"\n📈 Results:")
    print(f"   Total iterations: {result['iteration']}")
    print(f"   Final quality: {result['quality_score']:.1f}")
    print(f"   Answer: {result['final_answer'][:100]}...")

print("\n" + "=" * 40)
print("💡 KEY INSIGHTS")
print("=" * 40)
print("""
1. Loops enable iterative refinement
2. Router functions control loop flow
3. Always set max iterations to prevent infinite loops
4. Quality thresholds provide early exit
5. State accumulates through iterations
""")

# Save completion marker
try:
    with open('/root/loops-complete.txt', 'w') as f:
        f.write('LOOPS_COMPLETE\n')
        f.write('Task 6: Loops and iterations completed successfully\n')
    print("\n✅ Completion marker saved to /root/loops-complete.txt")
except:
    # For local testing
    print("\n✅ Task 6: Loops demonstration completed!")