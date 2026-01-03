"""Simple two-agent crew example using Alith-powered agents.

This example demonstrates creating a basic CrewAI crew with two agents:
a researcher and a writer, both powered by Alith for high-performance inference.
"""

import os

from crewai import Crew, Task

from alith_crewai import create_alith_crew_agent


def main():
    """Run a simple two-agent crew."""
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("   Set it with: export GROQ_API_KEY='your-api-key'")
        return
    
    print("🚀 Creating Alith-powered CrewAI agents...")
    
    # Create researcher agent
    researcher = create_alith_crew_agent(
        role="Research Analyst",
        goal="Find and analyze information about specific topics",
        backstory=(
            "You are an experienced research analyst with a keen eye for detail. "
            "You excel at finding accurate information and identifying key insights."
        ),
        model="llama-3.3-70b-versatile",
        verbose=True
    )
    
    # Create writer agent
    writer = create_alith_crew_agent(
        role="Content Writer",
        goal="Create engaging and informative content",
        backstory=(
            "You are a talented writer who can transform complex information "
            "into clear, engaging content that readers love."
        ),
        model="llama-3.3-70b-versatile",
        verbose=True
    )
    
    print("✅ Agents created successfully!")
    print("\n📋 Creating tasks...")
    
    # Create tasks
    research_task = Task(
        description=(
            "Research the latest trends in AI agent frameworks for 2024. "
            "Focus on multi-agent systems, orchestration patterns, and "
            "performance optimizations."
        ),
        agent=researcher,
        expected_output="A comprehensive summary of AI agent framework trends"
    )
    
    writing_task = Task(
        description=(
            "Based on the research findings, write a clear and engaging "
            "blog post about AI agent framework trends. Make it accessible "
            "to developers who are new to multi-agent systems."
        ),
        agent=writer,
        expected_output="A well-written blog post about AI agent frameworks"
    )
    
    print("✅ Tasks created!")
    print("\n👥 Creating crew and starting execution...")
    
    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True
    )
    
    # Execute crew
    print("\n" + "="*60)
    print("CREW EXECUTION STARTED")
    print("="*60 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*60)
    print("CREW EXECUTION COMPLETED")
    print("="*60)
    
    print("\n📊 Final Result:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
