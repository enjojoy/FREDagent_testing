#!/usr/bin/env python3
"""
Test FRED Economic Data Agent without payment integration
"""

from crew_definition import FREDEconomicCrew
from logging_config import setup_logging

def test_fred_query(query: str):
    """Test FRED agent with a specific query"""
    print(f"\n🔍 Testing: {query}")
    print("="*60)
    
    try:
        logger = setup_logging()
        crew = FREDEconomicCrew(logger=logger)
        result = crew.crew.kickoff(inputs={"text": query})
        
        print("📊 RESULT:")
        print(result)
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("="*60)

if __name__ == "__main__":
    # Test with a simple economic query
    test_fred_query("What is the current unemployment rate in the United States?")
