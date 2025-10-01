from crewai import Agent, Crew, Task
from logging_config import get_logger
from crewai.tools import tool
from fredapi import Fred
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

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
    Retrieve actual economic data from FRED for a specific series ID with comprehensive analysis.
    Returns recent data points, calculated metrics (MoM, YoY, percentiles), and statistical context.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables."
        
        fred = Fred(api_key=fred_api_key)
        
        # Get series info
        info = fred.get_series_info(series_id)
        
        # Get complete data history
        data = fred.get_series(series_id)
        
        if data.empty:
            return f"No data available for series ID: {series_id}"
        
        # Get recent data points (last 24 for comprehensive analysis - 2 years monthly or 6 years quarterly)
        recent_data = data.tail(24)
        
        # Calculate metrics
        current_value = data.iloc[-1]
        
        # MoM change (if monthly or higher frequency)
        mom_change = None
        mom_pct = None
        if len(data) >= 2:
            prev_value = data.iloc[-2]
            mom_change = current_value - prev_value
            if prev_value != 0:
                mom_pct = (mom_change / prev_value) * 100
        
        # YoY change (if we have 12+ months of data)
        yoy_change = None
        yoy_pct = None
        freq = info.get('frequency_short', 'N/A')
        lookback = 12 if freq in ['M', 'Monthly'] else 4 if freq in ['Q', 'Quarterly'] else 1
        if len(data) >= lookback + 1:
            year_ago_value = data.iloc[-(lookback + 1)]
            yoy_change = current_value - year_ago_value
            if year_ago_value != 0:
                yoy_pct = (yoy_change / year_ago_value) * 100
        
        # Historical statistics
        mean_value = data.mean()
        std_value = data.std()
        min_value = data.min()
        max_value = data.max()
        
        # Percentile rank of current value
        percentile = (data < current_value).sum() / len(data) * 100
        
        # Standard deviations from mean
        std_from_mean = (current_value - mean_value) / std_value if std_value != 0 else 0
        
        # 3-month or 3-period average
        period_avg = data.tail(3).mean() if len(data) >= 3 else current_value
        
        # Build comprehensive output with FIXED f-string syntax
        output = f"=== SERIES ANALYSIS: {info.get('title', series_id)} ===\n\n"
        output += f"📊 CURRENT DATA:\n"
        output += f"Series ID: {series_id}\n"
        # FIXED: Proper f-string syntax
        output += f"Current Value: {'N/A' if pd.isna(current_value) else f'{current_value:.2f}'}\n"
        output += f"Date: {data.index[-1].strftime('%Y-%m-%d')}\n"
        output += f"Frequency: {info.get('frequency', 'N/A')}\n"
        output += f"Units: {info.get('units', 'N/A')}\n"
        output += f"Seasonal Adjustment: {info.get('seasonal_adjustment', 'N/A')}\n\n"
        
        output += f"📈 CALCULATED METRICS:\n"
        if mom_change is not None:
            output += f"Month-over-Month Change: {mom_change:+.2f} ({mom_pct:+.2f}%)\n"
        if yoy_change is not None:
            output += f"Year-over-Year Change: {yoy_change:+.2f} ({yoy_pct:+.2f}%)\n"
        output += f"3-Period Average: {period_avg:.2f}\n\n"
        
        output += f"📉 HISTORICAL CONTEXT:\n"
        output += f"Historical Mean: {mean_value:.2f}\n"
        output += f"Standard Deviation: {std_value:.2f}\n"
        output += f"Historical Range: {min_value:.2f} to {max_value:.2f}\n"
        output += f"Current Percentile Rank: {percentile:.1f}th percentile\n"
        output += f"Distance from Mean: {std_from_mean:+.2f} standard deviations\n"
        output += f"Data Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n"
        output += f"Total Observations: {len(data)}\n\n"
        
        output += f"📋 RECENT DATA POINTS (Last 15):\n"
        for date, value in recent_data.tail(15).items():
            output += f"  {date.strftime('%Y-%m-%d')}: {value:.2f}\n"
        
        # Add summary of full dataset
        output += f"\n📊 FULL DATASET SUMMARY:\n"
        output += f"Total data points retrieved: {len(data)}\n"
        output += f"Oldest data: {data.index[0].strftime('%Y-%m-%d')} = {data.iloc[0]:.2f}\n"
        output += f"Newest data: {data.index[-1].strftime('%Y-%m-%d')} = {data.iloc[-1]:.2f}\n"
        output += f"Average over entire period: {mean_value:.2f}\n"
        output += f"Peak value: {max_value:.2f} on {data.idxmax().strftime('%Y-%m-%d')}\n"
        output += f"Trough value: {min_value:.2f} on {data.idxmin().strftime('%Y-%m-%d')}\n"
        
        output += f"\n🔗 View on FRED: https://fred.stlouisfed.org/series/{series_id}\n"
        
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
    Enhanced with analytical capabilities for comprehensive economic analysis.
    """
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        self.crew = self.create_crew()
        self.logger.info("FRED Economic Crew initialized")

    def create_crew(self):
        self.logger.info("Creating FRED economic data crew")
        
        # Agent 1: FRED Data Analyst - Enhanced with analytical requirements
        fred_analyst = Agent(
            role='Senior FRED Data Analyst',
            goal='Retrieve ALL relevant economic data series requested and provide comprehensive statistical analysis',
            backstory="""You are a senior economic data analyst with deep expertise in the Federal Reserve 
            Economic Data (FRED) database. You have a PhD in Economics and 15 years of experience analyzing 
            economic indicators. You are meticulous about retrieving ALL data series mentioned in queries - 
            if someone asks for 3 metrics, you retrieve ALL 3, not just one. You understand economic terminology, 
            series IDs, and know how to find both current and historical data. You always use the data retrieval 
            tool to get actual numbers with calculated metrics, never just search results.
            
            IMPORTANT: If a tool fails 2-3 times, try a different series ID or inform the user about the issue.""",
            tools=[fred_search_tool, fred_data_tool, fred_series_info_tool],
            verbose=self.verbose
        )

        # Agent 2: Economic Advisor - Enhanced to provide structured, actionable analysis
        economic_advisor = Agent(
            role='Chief Economic Interpreter',
            goal='Transform raw economic data into actionable insights with historical context and clear implications',
            backstory="""You are the Chief Economist at a major financial institution with 20 years of experience 
            interpreting economic data for investors, policymakers, and business leaders. You never just report 
            numbers - you ANALYZE them. You always provide:
            
            1. EXECUTIVE SUMMARY with key metrics and changes
            2. DETAILED ANALYSIS with calculated metrics (MoM, YoY, percentiles)
            3. HISTORICAL CONTEXT explaining if values are high/low vs historical norms
            4. INTERPRETATION of what the data means for the economy, policy, and markets
            5. ACTIONABLE GUIDANCE for different stakeholders
            
            You format responses in clear sections with bullet points and tables where appropriate. You make 
            complex economics accessible. You NEVER give generic boilerplate - every insight is specific to 
            the query. When users ask for comparisons, you create comparison tables. When they ask about 
            historical periods, you retrieve that specific historical data.
            
            CRITICAL: Only analyze data that was successfully retrieved. If data retrieval failed, acknowledge 
            this clearly and don't fabricate numbers.""",
            verbose=self.verbose
        )

        self.logger.info("Created enhanced FRED analyst and economic advisor agents")

        crew = Crew(
            agents=[fred_analyst, economic_advisor],
            tasks=[
                Task(
                    description="""Analyze this economic data query and retrieve ALL relevant data: {text}
                    
                    CRITICAL REQUIREMENTS:
                    1. Identify EVERY economic indicator mentioned in the query
                    2. If the query asks for multiple metrics (e.g., "compare A, B, and C"), retrieve ALL of them
                    3. If the query mentions specific time periods (e.g., "2008 crisis"), retrieve data from that period
                    4. Use fred_data_tool to get actual data with calculations - don't just search
                    5. Retrieve enough historical data to provide meaningful context
                    6. If a tool fails 2-3 times, try alternative series IDs or report the issue
                    
                    STEPS:
                    1. Parse query to identify ALL indicators requested
                    2. Search FRED for each indicator
                    3. Retrieve data for each relevant series using fred_data_tool
                    4. Verify you've retrieved data for EVERY part of the query
                    
                    FORBIDDEN:
                    - Never provide search results without retrieving actual data
                    - Never answer only part of a multi-part question
                    - Never skip historical periods specifically mentioned
                    - Never retry the same failing tool more than 3 times
                    """,
                    expected_output="""Complete data retrieval including:
                    - Actual data values for ALL series mentioned in query
                    - Calculated metrics (MoM, YoY, percentiles) for each series
                    - Historical context data if requested
                    - All series metadata and FRED links
                    - Clear indication if any data retrieval failed""",
                    agent=fred_analyst
                ),
                Task(
                    description="""Transform the retrieved FRED data into a comprehensive, actionable analysis.
                    
                    REQUIRED STRUCTURE:
                    
                    ## 📊 EXECUTIVE SUMMARY
                    - Bullet points with key findings
                    - Current values and most important changes
                    - One-line historical context (e.g., "highest since 2008")
                    
                    ## 📈 DETAILED ANALYSIS
                    - Present all retrieved data clearly
                    - For comparisons: create a table showing metrics side-by-side
                    - Include ALL calculated metrics provided (MoM, YoY, percentiles)
                    - Identify trends (increasing/decreasing, acceleration/deceleration)
                    
                    ## 🔍 HISTORICAL CONTEXT
                    - Is this high/low relative to historical norms?
                    - Compare to relevant historical periods if mentioned
                    - Explain significance of current percentile rank
                    
                    ## 💡 WHAT THIS MEANS
                    - Economic implications (growth, inflation, labor market health)
                    - Policy implications (likely Fed actions, fiscal policy relevance)
                    - Market implications (investor considerations)
                    - Business implications (hiring, pricing, investment decisions)
                    
                    ## 🔗 FURTHER EXPLORATION
                    - Direct FRED links for all series mentioned
                    - 2-3 query-specific related indicators to explore (no generic suggestions)
                    
                    CRITICAL REQUIREMENTS:
                    1. Answer EVERY part of the user's query - if they asked for 3 metrics, analyze all 3
                    2. Include ALL calculations available in the data
                    3. Use specific numbers from the actual data retrieved
                    4. Make insights actionable - tell users what to do with this information
                    5. Format clearly with sections, bullets, and tables
                    6. If data retrieval failed, clearly state this and don't fabricate numbers
                    
                    FORBIDDEN:
                    - Never give wall-of-text responses
                    - Never provide generic boilerplate
                    - Never skip parts of multi-part queries
                    - Never report numbers without explaining if they're high/low
                    - Never forget to format with clear sections
                    - Never fabricate data if retrieval failed""",
                    expected_output="""Structured economic analysis with:
                    - Executive summary with key findings
                    - Detailed analysis with all requested metrics and calculated changes
                    - Historical context with percentile rankings
                    - Clear interpretation for different stakeholders
                    - Actionable recommendations
                    - Query-specific related series suggestions
                    - Proper formatting with sections and tables""",
                    agent=economic_advisor
                )
            ],
            verbose=True
        )
        
        self.logger.info("FRED Economic Crew setup completed with enhanced analytical capabilities")
        return crew