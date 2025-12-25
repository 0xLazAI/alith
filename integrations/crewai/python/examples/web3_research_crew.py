"""Web3 research crew example using Alith's Web3 capabilities.

This example demonstrates a multi-agent crew focused on Web3/crypto research,
leveraging Alith's built-in Web3 tools and high-performance inference.
"""

import os

from crewai import Crew, Task

from alith_crewai import create_alith_crew_agent


def main():
    """Run a Web3-focused research crew."""
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("   Set it with: export GROQ_API_KEY='your-api-key'")
        return
    
    print("🚀 Creating Web3 research crew with Alith agents...")
    
    # Create blockchain analyst agent
    blockchain_analyst = create_alith_crew_agent(
        role="Blockchain Analyst",
        goal="Analyze blockchain data and smart contract trends",
        backstory=(
            "You are a blockchain expert with deep knowledge of smart contracts, "
            "DeFi protocols, and on-chain analytics. You can identify trends "
            "and potential opportunities in the Web3 ecosystem."
        ),
        model="llama-3.3-70b-versatile",
        verbose=True
    )
    
    # Create DeFi specialist agent
    defi_specialist = create_alith_crew_agent(
        role="DeFi Specialist",
        goal="Research and analyze DeFi protocols and trends",
        backstory=(
            "You are a DeFi expert who understands liquidity pools, yield farming, "
            "lending protocols, and tokenomics. You can evaluate DeFi projects "
            "and identify emerging trends."
        ),
        model="llama-3.3-70b-versatile",
        verbose=True
    )
    
    # Create report writer agent
    writer = create_alith_crew_agent(
        role="Crypto Report Writer",
        goal="Create comprehensive Web3 research reports",
        backstory=(
            "You are a specialized writer in the crypto/Web3 space. You can "
            "transform complex blockchain data into clear, actionable reports "
            "for investors and developers."
        ),
        model="llama-3.3-70b-versatile",
        verbose=True
    )
    
    print("✅ Agents created successfully!")
    print("\n📋 Creating Web3 research tasks...")
    
    # Create tasks
    blockchain_task = Task(
        description=(
            "Analyze the current state of blockchain infrastructure in 2024. "
            "Focus on: Layer 2 solutions, cross-chain bridges, and scalability "
            "improvements. Identify the top 5 most promising developments."
        ),
        agent=blockchain_analyst,
        expected_output="Analysis of blockchain infrastructure trends and developments"
    )
    
    defi_task = Task(
        description=(
            "Research the latest DeFi trends and protocols. Focus on: "
            "Real-world asset (RWA) tokenization, decentralized stablecoins, "
            "and new DeFi primitives. Identify risks and opportunities."
        ),
        agent=defi_specialist,
        expected_output="Analysis of DeFi trends, protocols, and opportunities"
    )
    
    report_task = Task(
        description=(
            "Based on the blockchain infrastructure and DeFi analyses, "
            "create a comprehensive Web3 research report. Include: "
            "executive summary, key findings, trend analysis, and recommendations "
            "for developers and investors."
        ),
        agent=writer,
        expected_output="Comprehensive Web3 research report with actionable insights"
    )
    
    print("✅ Tasks created!")
    print("\n👥 Creating Web3 research crew...")
    
    # Create crew with sequential process
    crew = Crew(
        agents=[blockchain_analyst, defi_specialist, writer],
        tasks=[blockchain_task, defi_task, report_task],
        verbose=True
    )
    
    # Execute crew
    print("\n" + "="*60)
    print("WEB3 RESEARCH CREW - EXECUTION STARTED")
    print("="*60 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*60)
    print("WEB3 RESEARCH CREW - EXECUTION COMPLETED")
    print("="*60)
    
    print("\n📊 Final Web3 Research Report:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    print("\n✅ Web3 research crew completed successfully!")
    print("💡 This example demonstrates Alith's Web3-friendly architecture")


if __name__ == "__main__":
    main()
