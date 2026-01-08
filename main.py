#!/usr/bin/env python3
"""
FRED Economic Data Agent - A runnable agent that can be used locally and deployed.

This agent processes economic data queries using FRED (Federal Reserve Economic Data)
and integrates with Masumi's decentralized payment solution.

To run:
    # API mode (default) - runs as FastAPI server
    masumi run main.py
    # Or: python main.py

    # Standalone mode - executes job directly without API
    masumi run main.py --standalone
    # Or: masumi run main.py --standalone --input '{"text": "What is the current unemployment rate?"}'

Required environment variables for API mode:
    - AGENT_IDENTIFIER: Your agent ID from admin interface (REQUIRED for API)
    - PAYMENT_API_KEY: Your payment API key (REQUIRED for API)
    - PAYMENT_SERVICE_URL: Payment service URL (optional, defaults to production)
    - NETWORK: Network to use - 'Preprod' or 'Mainnet' (optional, defaults to 'Preprod')
    - OPENAI_API_KEY: OpenAI API key for CrewAI (REQUIRED)
    - FRED_API_KEY: FRED API key for economic data (REQUIRED)
"""

import os
from masumi import run
from crew_definition import FREDEconomicCrew
from testing.logging_config import setup_logging

# Note: .env files are automatically loaded by masumi.run() from the current directory

# Configure logging
logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Agent Logic - This is where you implement your actual agent functionality
# ─────────────────────────────────────────────────────────────────────────────

async def process_job(identifier_from_purchaser: str, input_data: dict):
    """
    Process a job - can be run locally or via Masumi API.
    
    When used with Masumi API: This function runs automatically after payment is confirmed.
    The payment creation, monitoring, and completion are all handled automatically.
    
    When run locally: This function can be executed directly without any masumi setup.
    
    Args:
        identifier_from_purchaser: Identifier from the purchaser (can be any string for local testing)
        input_data: Input data matching your input schema
    
    Returns:
        Result of the processing (string, dict, or any serializable type)
    """
    logger.info(f"Processing job for purchaser: {identifier_from_purchaser}")
    logger.info(f"Input data: {input_data}")
    
    # Extract input
    text = input_data.get("text", "")
    
    if not text or len(text.strip()) < 5:
        error_msg = "Input text must contain at least 5 characters"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    
    try:
        # Execute the CrewAI task with FRED Economic Data Agents
        logger.info(f"Starting FRED Economic Data query with input: {text[:100]}...")
        crew = FREDEconomicCrew(logger=logger)
        result = crew.crew.kickoff(inputs={"text": text})
        logger.info("FRED Economic Data query completed successfully")
        
        # Convert result to string for payment completion
        # Check if result has .raw attribute (CrewOutput), otherwise convert to string
        result_string = result.raw if hasattr(result, "raw") else str(result)
        
        logger.info(f"Processing complete. Result length: {len(result_string)} characters")
        
        # Return result - can be string, dict, or any serializable type
        return result_string
    except Exception as e:
        error_msg = f"Error processing FRED query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return 


# ─────────────────────────────────────────────────────────────────────────────
# Input Schema - Define what input fields your agent expects
# ─────────────────────────────────────────────────────────────────────────────

INPUT_SCHEMA = {
    "input_data": [
        {
            "id": "text",
            "type": "string",
            "name": "Economic Data Query",
            "required": True,
            "data": {
                "description": "Your question about economic data from FRED (Federal Reserve Economic Data). Ask about unemployment, inflation, GDP, interest rates, or any other economic indicators.",
                "placeholder": "e.g., What is the current unemployment rate? or Show me GDP growth data for 2023",
                "examples": [
                    "What is the current inflation rate in the United States?",
                    "Show me the unemployment rate over the last 12 months",
                    "What is the current federal funds rate?",
                    "Display GDP growth data for the last quarter"
                ]
            }
        }
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point - Simplified approach using masumi.run()
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        start_job_handler=process_job,
        input_schema_handler=INPUT_SCHEMA
    )
