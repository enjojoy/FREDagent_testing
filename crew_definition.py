from crewai import Agent, Crew, Task
from logging_config import get_logger
from crewai.tools import tool
from fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv()

@tool("FRED Search Tool")
def fred_search_tool(query: str) -> str:
    """
    Search the FRED database for economic data series matching the query.
    Returns series IDs, titles, and descriptions of matching datasets.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables. Please add it to your .env file."
        
        fred = Fred(api_key=fred_api_key)
        results = fred.search(query, limit=10)
        
        if results.empty:
            return f"No results found for query: '{query}'"
        
        output = f"Found {len(results)} series matching '{query}':\n\n"
        for idx, (series_id, row) in enumerate(results.iterrows(), 1):
            output += f"{idx}. {row.get('title', 'N/A')} (ID: {series_id})\n"
            output += f"   Description: {row.get('notes', 'No description available')[:200]}...\n"
            output += f"   Frequency: {row.get('frequency_short', 'N/A')} | Units: {row.get('units_short', 'N/A')}\n\n"
        
        return output
    except Exception as e:
        return f"Error searching FRED: {str(e)}"

@tool("FRED Data Retrieval Tool")
def fred_data_tool(series_id: str) -> str:
    """
    Retrieve actual economic data from FRED for a specific series ID.
    Returns the most recent data points and summary statistics.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables."
        
        fred = Fred(api_key=fred_api_key)
        
        # Get series info
        info = fred.get_series_info(series_id)
        
        # Get recent data
        data = fred.get_series(series_id)
        
        if data.empty:
            return f"No data available for series ID: {series_id}"
        
        # Get last 5 data points
        recent_data = data.tail(10)
        
        output = f"Series: {info.get('title', series_id)}\n"
        output += f"ID: {series_id}\n"
        output += f"Frequency: {info.get('frequency', 'N/A')}\n"
        output += f"Units: {info.get('units', 'N/A')}\n"
        output += f"Last Updated: {info.get('last_updated', 'N/A')}\n\n"
        output += "Recent Data Points:\n"
        
        for date, value in recent_data.items():
            output += f"  {date.strftime('%Y-%m-%d')}: {value}\n"
        
        output += f"\nData Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n"
        output += f"Total Observations: {len(data)}\n"
        
        return output
    except Exception as e:
        return f"Error retrieving data for {series_id}: {str(e)}"

@tool("FRED Series Info Tool")
def fred_series_info_tool(series_id: str) -> str:
    """
    Get detailed information about a FRED data series including metadata and source information.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables."
        
        fred = Fred(api_key=fred_api_key)
        info = fred.get_series_info(series_id)
        
        output = f"Series Information for {series_id}:\n\n"
        output += f"Title: {info.get('title', 'N/A')}\n"
        output += f"Observation Start: {info.get('observation_start', 'N/A')}\n"
        output += f"Observation End: {info.get('observation_end', 'N/A')}\n"
        output += f"Frequency: {info.get('frequency', 'N/A')}\n"
        output += f"Units: {info.get('units', 'N/A')}\n"
        output += f"Seasonal Adjustment: {info.get('seasonal_adjustment', 'N/A')}\n"
        output += f"Last Updated: {info.get('last_updated', 'N/A')}\n"
        output += f"Popularity: {info.get('popularity', 'N/A')}\n\n"
        output += f"Notes: {info.get('notes', 'No notes available')}\n\n"
        output += f"🔗 View on FRED: https://fred.stlouisfed.org/series/{series_id}\n"
        
        return output
    except Exception as e:
        return f"Error getting info for {series_id}: {str(e)}"


class FREDEconomicCrew:
    """
    A specialized CrewAI crew for querying and analyzing FRED economic data.
    """
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = self.create_crew()
        self.logger.info("FRED Economic Crew initialized")

    def create_crew(self):
        self.logger.info("Creating FRED economic data crew")
        
        # Agent 1: FRED Data Analyst
        fred_analyst = Agent(
            role='FRED Data Analyst',
            goal='Search and retrieve economic data from the FRED database based on user queries',
            backstory="""You are an expert at navigating the Federal Reserve Economic Data (FRED) database.
            You understand economic indicators, can search for relevant data series, and retrieve actual data values.
            You know how to interpret series IDs and understand economic terminology.""",
            tools=[fred_search_tool, fred_data_tool, fred_series_info_tool],
            verbose=self.verbose
        )

        # Agent 2: Economic Advisor
        economic_advisor = Agent(
            role='Economic Data Advisor',
            goal='Interpret economic data and guide users to find relevant information on FRED',
            backstory="""You are an experienced economist who helps people understand economic data.
            You explain what different economic indicators mean, provide context for data trends,
            and direct people to the right FRED data series for their needs. You make economics accessible
            and provide clear guidance on where to find specific types of economic information.""",
            verbose=self.verbose
        )

        self.logger.info("Created FRED analyst and economic advisor agents")

        crew = Crew(
            agents=[fred_analyst, economic_advisor],
            tasks=[
                Task(
                    description="""Analyze the following economic data query and search FRED for relevant data: {text}
                    
                    Steps:
                    1. Search FRED for data series related to the query
                    2. Retrieve data for the most relevant series (use the series ID)
                    3. Get detailed information about the series
                    4. Provide the actual data points and context""",
                    expected_output="""A comprehensive response including:
                    - List of relevant FRED data series found
                    - Actual recent data values for the most relevant series
                    - Series metadata (frequency, units, date range)
                    - Direct links to FRED for further exploration""",
                    agent=fred_analyst
                ),
                Task(
                    description="""Based on the FRED data retrieved, provide clear guidance to the user.
                    Explain what the data means and where they can find more related information.""",
                    expected_output="""A user-friendly summary that includes:
                    - Clear explanation of what the data shows
                    - Context about the economic indicators
                    - Specific guidance on where to find related data on FRED
                    - Suggestions for related series or topics to explore
                    - Direct FRED links for all mentioned series""",
                    agent=economic_advisor
                )
            ],
            verbose=True
        )
        
        self.logger.info("FRED Economic Crew setup completed")
        return crew