#!/usr/bin/env python3
"""
Interactive FRED Economic Data Agent Tester
Run this to test your agent with any query without modifying main.py
"""

from crew_definition import FREDEconomicCrew
from logging_config import setup_logging

logger = setup_logging()

def test_query(query: str):
    """
    Test the FRED agent with a custom query
    """
    print("\n" + "="*80)
    print(f"🔍 QUERY: {query}")
    print("="*80 + "\n")
    
    try:
        crew = FREDEconomicCrew(logger=logger)
        result = crew.crew.kickoff(inputs={"text": query})
        
        print("\n" + "="*80)
        print("📊 FRED AGENT RESPONSE:")
        print("="*80)
        print(result)
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")


def main():
    """
    Run interactive query session
    """
    print("\n" + "🏦 FRED Economic Data Agent - Interactive Tester 🏦".center(80))
    print("="*80)
    print("\nThis tool lets you test your FRED agent with any economic data query.")
    print("\nExample queries:")
    print("  • What is the current unemployment rate?")
    print("  • Show me GDP growth data")
    print("  • Get inflation rate trends")
    print("  • What is the federal funds rate?")
    print("  • Show me consumer price index data")
    print("\n" + "="*80 + "\n")
    
    # Option to run predefined queries or custom
    print("Choose an option:")
    print("1. Enter your own query")
    print("2. Run example queries")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Custom query
        print("\n" + "-"*80)
        query = input("Enter your economic data query: ").strip()
        if query:
            test_query(query)
        else:
            print("❌ No query entered.")
            
    elif choice == "2":
        # Example queries
        examples = [
            "What is the current unemployment rate in the United States?",
            "Show me GDP growth data",
            "What is the inflation rate?",
            "Get consumer price index trends",
        ]
        
        print("\n🧪 Running example queries...\n")
        for idx, query in enumerate(examples, 1):
            print(f"\n📋 Example {idx}/{len(examples)}")
            test_query(query)
            
            if idx < len(examples):
                cont = input("\nPress Enter to continue to next query (or 'q' to quit): ").strip().lower()
                if cont == 'q':
                    break
                    
    elif choice == "3":
        print("\n👋 Goodbye!\n")
        return
    else:
        print("\n❌ Invalid choice.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}\n")
