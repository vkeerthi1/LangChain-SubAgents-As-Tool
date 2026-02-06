"""
LangChain: Subagents as Tools Pattern
======================================
Shows how to wrap specialized agents as tools that a supervisor can call.

Pattern:
1. Create specialized subagents using create_agent()
2. Wrap each subagent as a @tool
3. Create supervisor agent that uses these tool-wrapped subagents
"""

import os
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
AZURE_OPENAI_ENDPOINT = os.getenv("OPEN_AI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
AZURE_DEPLOYMENT_NAME = "gpt-4.1"
AZURE_API_VERSION = "2024-02-15-preview"


# ============================================================================
# CREATE LLM INSTANCE (shared across agents)
# ============================================================================
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version=AZURE_API_VERSION,
    temperature=0.7
)


# ============================================================================
# STEP 1: CREATE SUBAGENTS
# ============================================================================

# Calculator Subagent
CALCULATOR_AGENT_PROMPT = """You are a calculator specialist.
Extract mathematical expressions from requests and compute them accurately.
Always show the full calculation and final result."""

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation: {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

calculator_agent = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt=CALCULATOR_AGENT_PROMPT,
)


# Analyzer Subagent
ANALYZER_AGENT_PROMPT = """You are a number analyst specialist.
Analyze mathematical results and explain their properties:
- Is it even or odd?
- Is it positive, negative, or zero?
- Size category (small <100, medium 100-1000, large >1000)
- Any special mathematical properties

Be concise and clear."""

analyzer_agent = create_agent(
    model=llm,
    tools=[],  # No tools needed
    system_prompt=ANALYZER_AGENT_PROMPT,
)


# Approval Subagent
APPROVAL_AGENT_PROMPT = """You are an approval coordinator.
When asked to get approval, use the request_approval tool with a clear summary of what needs approval.
Always be specific about what action requires approval."""

@tool
def request_approval(message: str) -> str:
    """Request human approval for an action"""
    print("\n" + "="*70)
    print("👤 [APPROVAL AGENT] HUMAN APPROVAL REQUIRED")
    print("="*70)
    print(f"\n{message}\n")
    
    while True:
        approval = input("Do you approve? (yes/no): ").strip().lower()
        if approval in ['yes', 'y']:
            return "Approved - you may proceed with the action"
        elif approval in ['no', 'n']:
            return "Rejected - do not proceed with the action"
        else:
            print("Please answer 'yes' or 'no'")

approval_agent = create_agent(
    model=llm,
    tools=[request_approval],
    system_prompt=APPROVAL_AGENT_PROMPT,
)


# ============================================================================
# STEP 2: WRAP SUBAGENTS AS TOOLS
# ============================================================================

@tool
def use_calculator(request: str) -> str:
    """Perform mathematical calculations.
    
    Use this when you need to calculate mathematical expressions.
    Handles arithmetic operations like addition, subtraction, multiplication, division.
    
    Args:
        request: Natural language math request (e.g., "calculate 15 times 8 plus 100 divided by 4")
    
    Returns:
        Calculation result
    """
    print("\n🔧 [SUPERVISOR] Delegating to Calculator Agent...")
    result = calculator_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content


@tool
def analyze_result(number_or_calculation: str) -> str:
    """Analyze properties of a number or calculation result.
    
    Use this to understand mathematical properties of numbers.
    Provides analysis of even/odd, positive/negative, size category, and special properties.
    
    Args:
        number_or_calculation: A number or calculation result to analyze
    
    Returns:
        Analysis of the number's properties
    """
    print("\n🔍 [SUPERVISOR] Delegating to Analyzer Agent...")
    result = analyzer_agent.invoke({
        "messages": [{"role": "user", "content": f"Analyze the number: {number_or_calculation}"}]
    })
    return result["messages"][-1].content


@tool
def get_approval(summary: str) -> str:
    """Request human approval before proceeding.
    
    Use this when you need human confirmation before taking an action or 
    providing final results. Always provide a clear summary of what needs approval.
    
    Args:
        summary: Clear summary of what needs approval
    
    Returns:
        Approval status (approved or rejected)
    """
    print("\n👤 [SUPERVISOR] Delegating to Approval Agent...")
    result = approval_agent.invoke({
        "messages": [{"role": "user", "content": f"Get approval for: {summary}"}]
    })
    return result["messages"][-1].content


# ============================================================================
# STEP 3: CREATE SUPERVISOR AGENT
# ============================================================================

SUPERVISOR_PROMPT = """You are a helpful coordinator agent that manages specialized subagents.

Available subagent tools:
- use_calculator: For mathematical calculations
- analyze_result: For analyzing number properties
- get_approval: For getting human approval

Workflow for math tasks:
1. Use use_calculator to perform calculations
2. Use analyze_result to explain the result
3. Use get_approval to get human confirmation
4. Provide a final summary

Always explain what you're doing at each step. Call the tools in the order shown above."""

supervisor_agent = create_agent(
    model=llm,
    tools=[use_calculator, analyze_result, get_approval],
    system_prompt=SUPERVISOR_PROMPT,
)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("🤖 LangChain: Subagents as Tools Pattern")
    print("="*70)
    
    print("\n✅ Supervisor agent ready with tools:")
    print("   • use_calculator (wraps Calculator Agent)")
    print("   • analyze_result (wraps Analyzer Agent)")
    print("   • get_approval (wraps Approval Agent)")
    
    # Task
    task = """
    I need you to:
    1. Calculate (15 * 8) + (100 / 4)
    2. Analyze the properties of the result
    3. Get my approval before giving me the final summary
    """
    
    print("\n" + "="*70)
    print("📝 Task:")
    print("="*70)
    print(task)
    
    print("\n" + "="*70)
    print("🧠 Supervisor Agent Execution:")
    print("="*70 + "\n")
    
    try:
        # Run supervisor agent
        result = supervisor_agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        # Display final result
        print("\n" + "="*70)
        print("✅ FINAL RESULT")
        print("="*70)
        print(result["messages"][-1].content)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """Interactive mode with supervisor and subagent tools"""
    print("\n" + "="*70)
    print("💬 Interactive Mode - Type 'quit' to exit")
    print("="*70)
    
    print("\n🤖 Supervisor Agent ready!\n")
    
    while True:
        print("-" * 70)
        user_input = input("\nYour task (or 'quit'): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            result = supervisor_agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })
            print("\n" + "="*70)
            print("✅ RESULT")
            print("="*70)
            print(result["messages"][-1].content)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            🤖 SUBAGENTS AS TOOLS PATTERN                         ║
║                                                                  ║
║  Supervisor Agent delegates to specialized subagent tools       ║
║  Each subagent is created with create_agent()                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demo
    main()
    
    # Uncomment for interactive mode
    # interactive_mode()