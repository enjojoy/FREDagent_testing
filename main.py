import os
import uvicorn
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field, field_validator
from masumi.config import Config
from masumi.payment import Payment, Amount
from crew_definition import FREDEconomicCrew
from logging_config import setup_logging

# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Retrieve API Keys and URLs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY")
NETWORK = os.getenv("NETWORK")

logger.info("Starting application with configuration:")
logger.info(f"PAYMENT_SERVICE_URL: {PAYMENT_SERVICE_URL}")

# Initialize FastAPI
app = FastAPI(
    title="FRED Economic Data Agent - Masumi API Standard",
    description="AI agent for querying Federal Reserve Economic Data (FRED) with Masumi payment integration",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# Temporary in-memory job store (DO NOT USE IN PRODUCTION)
# ─────────────────────────────────────────────────────────────────────────────
jobs = {}
payment_instances = {}

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Masumi Payment Config
# ─────────────────────────────────────────────────────────────────────────────
config = Config(
    payment_service_url=PAYMENT_SERVICE_URL,
    payment_api_key=PAYMENT_API_KEY
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────
class StartJobRequest(BaseModel):
    identifier_from_purchaser: str = Field(..., min_length=1, description="Unique identifier from the purchaser")
    input_data: dict[str, str] = Field(..., description="Input data containing the economic query")
    
    @field_validator('input_data')
    @classmethod
    def validate_input_data(cls, v):
        if not v or 'text' not in v:
            raise ValueError('input_data must contain a "text" field with the economic query')
        if not v['text'] or len(v['text'].strip()) < 5:
            raise ValueError('text field must contain at least 5 characters')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "identifier_from_purchaser": "example_purchaser_123",
                "input_data": {
                    "text": "What is the current unemployment rate in the United States?"
                }
            }
        }

class ProvideInputRequest(BaseModel):
    job_id: str

# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Task Execution
# ─────────────────────────────────────────────────────────────────────────────
async def execute_crew_task(input_data: str) -> str:
    """ Execute a CrewAI task with FRED Economic Data Agents """
    logger.info(f"Starting FRED Economic Data query with input: {input_data}")
    crew = FREDEconomicCrew(logger=logger)
    result = crew.crew.kickoff(inputs={"text": input_data})
    logger.info("FRED Economic Data query completed successfully")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 1) Start Job (MIP-003: /start_job)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_job")
async def start_job(data: StartJobRequest):
    """ Initiates a job and creates a payment request """
    print(f"Received data: {data}")
    print(f"Received data.input_data: {data.input_data}")
    try:
        job_id = str(uuid.uuid4())
        agent_identifier = os.getenv("AGENT_IDENTIFIER")
        
        # Log the input text (truncate if too long)
        input_text = data.input_data["text"]
        truncated_input = input_text[:100] + "..." if len(input_text) > 100 else input_text
        logger.info(f"Received job request with input: '{truncated_input}'")
        logger.info(f"Starting job {job_id} with agent {agent_identifier}")

        # Define payment amounts
        payment_amount = os.getenv("PAYMENT_AMOUNT", "10000000")  # Default 10 ADA
        payment_unit = os.getenv("PAYMENT_UNIT", "lovelace") # Default lovelace

        amounts = [Amount(amount=payment_amount, unit=payment_unit)]
        logger.info(f"Using payment amount: {payment_amount} {payment_unit}")
        
        # Create a payment request using Masumi
        payment = Payment(
            agent_identifier=agent_identifier,
            amounts=amounts,
            config=config,
            identifier_from_purchaser=data.identifier_from_purchaser,
            input_data=data.input_data,
            network=NETWORK
        )
        
        logger.info("Creating payment request...")
        payment_request = await payment.create_payment_request()
        blockchain_identifier = payment_request["data"]["blockchainIdentifier"]
        payment.payment_ids.add(blockchain_identifier)
        logger.info(f"Created payment request with blockchain identifier: {blockchain_identifier}")

        # Store job info (Awaiting payment)
        jobs[job_id] = {
            "status": "awaiting_payment",
            "payment_status": "pending",
            "blockchain_identifier": blockchain_identifier,
            "input_data": data.input_data,
            "result": None,
            "identifier_from_purchaser": data.identifier_from_purchaser
        }

        async def payment_callback(blockchain_identifier: str):
            await handle_payment_status(job_id, blockchain_identifier)

        # Start monitoring the payment status
        payment_instances[job_id] = payment
        logger.info(f"Starting payment status monitoring for job {job_id}")
        await payment.start_status_monitoring(payment_callback)

        # Return the response in the required format
        return {
            "status": "success",
            "job_id": job_id,
            "blockchainIdentifier": blockchain_identifier,
            "submitResultTime": payment_request["data"]["submitResultTime"],
            "unlockTime": payment_request["data"]["unlockTime"],
            "externalDisputeUnlockTime": payment_request["data"]["externalDisputeUnlockTime"],
            "agentIdentifier": agent_identifier,
            "sellerVKey": os.getenv("SELLER_VKEY"),
            "identifierFromPurchaser": data.identifier_from_purchaser,
            "amounts": amounts,
            "input_hash": payment.input_hash,
            "payByTime": payment_request["data"]["payByTime"],
        }
    except ValueError as e:
        logger.error(f"Validation error in request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )
    except KeyError as e:
        logger.error(f"Missing required field in request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Bad Request: Missing required field in request data."
        )
    except Exception as e:
        logger.error(f"Error in start_job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing the request."
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2) Process Payment and Execute AI Task
# ─────────────────────────────────────────────────────────────────────────────
async def handle_payment_status(job_id: str, payment_id: str) -> None:
    """ Executes CrewAI task after payment confirmation """
    try:
        logger.info(f"Payment {payment_id} completed for job {job_id}, executing task...")
        
        # Update job status to running
        jobs[job_id]["status"] = "running"
        logger.info(f"Input data: {jobs[job_id]["input_data"]}")

        # Execute the AI task
        result = await execute_crew_task(jobs[job_id]["input_data"])
        print(f"Result: {result}")
        logger.info(f"Crew task completed for job {job_id}")
        
        # Convert result to string for payment completion
        # Check if result has .raw attribute (CrewOutput), otherwise convert to string
        result_string = result.raw if hasattr(result, "raw") else str(result)
        
        # Mark payment as completed on Masumi
        # Use a shorter string for the result hash
        await payment_instances[job_id].complete_payment(payment_id, result_string)
        logger.info(f"Payment completed for job {job_id}")

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["payment_status"] = "completed"
        jobs[job_id]["result"] = result

        # Stop monitoring payment status
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]
    except Exception as e:
        print(f"Error processing payment {payment_id} for job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        
        # Still stop monitoring to prevent repeated failures
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# 3) Check Job and Payment Status (MIP-003: /status)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
async def get_status(job_id: str):
    """ Retrieves the current status of a specific job """
    logger.info(f"Checking status for job {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Check latest payment status if payment instance exists
    if job_id in payment_instances:
        try:
            status = await payment_instances[job_id].check_payment_status()
            job["payment_status"] = status.get("data", {}).get("status")
            logger.info(f"Updated payment status for job {job_id}: {job['payment_status']}")
        except ValueError as e:
            logger.warning(f"Error checking payment status: {str(e)}")
            job["payment_status"] = "unknown"
        except Exception as e:
            logger.error(f"Error checking payment status: {str(e)}", exc_info=True)
            job["payment_status"] = "error"


    result_data = job.get("result")
    logger.info(f"Result data: {result_data}")
    result = result_data.raw if result_data and hasattr(result_data, "raw") else None



    return {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job["payment_status"],
        "result": result
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) Check Server Availability (MIP-003: /availability)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/availability")
async def check_availability():
    """ Checks if the server is operational """
    agent_identifier = os.getenv("AGENT_IDENTIFIER")
    
    return {
        "status": "available", 
        "type": "masumi-agent", 
        "agent_type": "fred-economic-data",
        "agentIdentifier": agent_identifier,
        "version": "1.0.0",
        "message": "FRED Economic Data Agent operational and ready to process economic data queries."
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5) Retrieve Input Schema (MIP-003: /input_schema)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/input_schema")
async def input_schema():
    """
    Returns the expected input schema for the /start_job endpoint.
    Fulfills MIP-003 /input_schema endpoint.
    """
    return {
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
# 6) Agent Metadata (Required for Sokosumi)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/metadata")
async def get_agent_metadata():
    """
    Returns agent metadata for Sokosumi marketplace listing.
    """
    agent_identifier = os.getenv("AGENT_IDENTIFIER")
    payment_amount = os.getenv("PAYMENT_AMOUNT", "10000000")
    payment_unit = os.getenv("PAYMENT_UNIT", "lovelace")
    
    return {
        "agentIdentifier": agent_identifier,
        "name": "FRED Economic Data Agent",
        "description": "AI-powered agent that queries and analyzes Federal Reserve Economic Data (FRED). Get real-time economic indicators, historical data, and expert analysis on unemployment, inflation, GDP, interest rates, and more.",
        "version": "1.0.0",
        "category": "Economic Data Analysis",
        "tags": ["economics", "fred", "data-analysis", "federal-reserve", "economic-indicators"],
        "pricing": {
            "amount": payment_amount,
            "unit": payment_unit,
            "currency": "ADA"
        },
        "capabilities": [
            "Search FRED database for economic data series",
            "Retrieve real-time economic indicators",
            "Provide historical data analysis",
            "Explain economic concepts and trends",
            "Generate direct links to FRED data sources"
        ],
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Economic data query or question",
                    "examples": [
                        "What is the current unemployment rate?",
                        "Show me inflation data for 2023",
                        "What is the federal funds rate?"
                    ]
                }
            },
            "required": ["text"]
        },
        "outputFormat": "JSON with economic data, analysis, and FRED links",
        "averageProcessingTime": "30-60 seconds",
        "supportedNetworks": ["PREPROD", "MAINNET"],
        "contact": {
            "website": "https://docs.masumi.network",
            "support": "https://docs.masumi.network/documentation"
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7) Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Returns the health of the server.
    """
    return {
        "status": "healthy"
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main Logic if Called as a Script
# ─────────────────────────────────────────────────────────────────────────────
def test_standalone():
    """
    Standalone test function for FRED Economic Data Agent
    Usage: python main.py
    """
    print("\n" + "="*80)
    print("🧪 FRED Economic Data Agent - Standalone Test")
    print("="*80 + "\n")
    
    input_data = {"text": "What is the current unemployment rate in the United States?"}
    crew = FREDEconomicCrew()
    result = crew.crew.kickoff(input_data)
    
    print("\n" + "="*80)
    print("📊 FRED AGENT RESPONSE:")
    print("="*80)
    print(result)
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    import os
    
    # Check if running in production (Railway sets PORT environment variable)
    if os.getenv("PORT") or (len(sys.argv) > 1 and sys.argv[1] == "api"):
        print("\n" + "="*80)
        print("🚀 Starting FRED Economic Data Agent API Server")
        print("="*80)
        print("\n📡 API will be available at:")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Redoc: http://localhost:8000/redoc")
        print("   • Health: http://localhost:8000/health")
        print("   • Availability: http://localhost:8000/availability")
        print("\n" + "="*80 + "\n")
        
        # Use PORT from environment if available (Railway), otherwise default to 8000
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        test_standalone()