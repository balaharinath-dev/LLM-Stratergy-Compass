# import sys
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import time
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
import requests
import pandas as pd
import plotly.express as px
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="LLM Strategy Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Tool Definitions ---
class CloudPricingToolInput(BaseModel):
    """Input schema for Cloud Pricing Tool."""
    provider: str = Field(..., description="Cloud provider (AWS, GCP, Azure)")
    model: str = Field(..., description="LLM model name")
    region: str = Field(..., description="Cloud region")

class CloudPricingTool(BaseTool):
    name: str = "get_cloud_pricing"
    description: str = "Get current pricing for cloud-based LLM services"
    args_schema: Type[BaseModel] = CloudPricingToolInput

    def _run(self, provider: str, model: str, region: str) -> Dict[str, Any]:
        """Get current pricing for cloud-based LLM services."""
        # This tool will now fetch real-time data from APIs when possible
        try:
            # For demo, we'll simulate API calls but in production these would be real API calls
            if provider.lower() == "aws":
                # Simulate AWS Pricing API response
                return {
                    "provider": provider,
                    "model": model,
                    "region": region,
                    "pricing": {
                        "input": 0.003 if "claude" in model.lower() else 0.03,
                        "output": 0.015 if "claude" in model.lower() else 0.06,
                        "unit": "per 1000 tokens"
                    },
                    "status": "success"
                }
            elif provider.lower() == "gcp":
                # Simulate GCP Pricing API response
                return {
                    "provider": provider,
                    "model": model,
                    "region": region,
                    "pricing": {
                        "input": 0.0005 if "gemini" in model.lower() else 0.002,
                        "output": 0.0015 if "gemini" in model.lower() else 0.003,
                        "unit": "per 1000 tokens"
                    },
                    "status": "success"
                }
            elif provider.lower() == "azure":
                # Simulate Azure Pricing API response
                return {
                    "provider": provider,
                    "model": model,
                    "region": region,
                    "pricing": {
                        "input": 0.03 if "gpt-4" in model.lower() else 0.0015,
                        "output": 0.06 if "gpt-4" in model.lower() else 0.002,
                        "unit": "per 1000 tokens"
                    },
                    "status": "success"
                }
            else:
                return {"status": "error", "message": "Provider not supported"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class GPUPricingToolInput(BaseModel):
    """Input schema for GPU Pricing Tool."""
    gpu_type: str = Field(..., description="GPU type (e.g., A100, H100, L4)")
    quantity: int = Field(..., description="Number of GPUs")

class GPUPricingTool(BaseTool):
    name: str = "get_gpu_pricing"
    description: str = "Get current pricing for self-hosted GPU solutions"
    args_schema: Type[BaseModel] = GPUPricingToolInput

    def _run(self, gpu_type: str, quantity: int) -> Dict[str, Any]:
        """Get current pricing for self-hosted GPU solutions."""
        try:
            # Simulate fetching real-time GPU pricing from a vendor API
            # In production, this would call actual vendor APIs
            
            # Base pricing data that would normally come from an API
            gpu_specs = {
                "A100": {
                    "purchase_price": self._get_current_price("A100", "purchase"),
                    "monthly_cloud": self._get_current_price("A100", "cloud"),
                    "power_consumption": 300,
                    "memory": 40,
                    "tokens_per_second": self._estimate_performance("A100")
                },
                "H100": {
                    "purchase_price": self._get_current_price("H100", "purchase"),
                    "monthly_cloud": self._get_current_price("H100", "cloud"),
                    "power_consumption": 350,
                    "memory": 80,
                    "tokens_per_second": self._estimate_performance("H100")
                },
                "L4": {
                    "purchase_price": self._get_current_price("L4", "purchase"),
                    "monthly_cloud": self._get_current_price("L4", "cloud"),
                    "power_consumption": 150,
                    "memory": 24,
                    "tokens_per_second": self._estimate_performance("L4")
                }
            }
            
            gpu_data = gpu_specs.get(gpu_type.upper(), {})
            if not gpu_data:
                return {"status": "error", "message": "GPU type not supported"}
            
            return {
                "gpu_type": gpu_type,
                "quantity": quantity,
                "total_purchase_price": gpu_data.get("purchase_price", 0) * quantity,
                "total_monthly_cloud": gpu_data.get("monthly_cloud", 0) * quantity,
                "total_power": gpu_data.get("power_consumption", 0) * quantity,
                "total_memory": gpu_data.get("memory", 0) * quantity,
                "total_tokens_per_second": gpu_data.get("tokens_per_second", 0) * quantity,
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _get_current_price(self, gpu_type: str, price_type: str) -> float:
        """Simulate fetching current market price (would be API call in production)."""
        # These would be API calls in production
        base_prices = {
            "A100": {"purchase": 10000, "cloud": 3000},
            "H100": {"purchase": 30000, "cloud": 5000},
            "L4": {"purchase": 5000, "cloud": 1000}
        }
        return base_prices.get(gpu_type, {}).get(price_type, 0)
    
    def _estimate_performance(self, gpu_type: str) -> float:
        """Estimate tokens per second based on GPU type."""
        # This would be more sophisticated in production, possibly using benchmarks
        performance = {
            "A100": 100,
            "H100": 200,
            "L4": 50
        }
        return performance.get(gpu_type, 0)

class TokenEstimationToolInput(BaseModel):
    """Input schema for Token Estimation Tool."""
    use_case: str = Field(..., description="LLM use case description")
    users: int = Field(..., description="Number of users")
    usage_intensity: str = Field(..., description="Usage intensity (low, medium, high)")
    token_pattern: str = Field(..., description="Token usage pattern description")

class TokenEstimationTool(BaseTool):
    name: str = "estimate_token_usage"
    description: str = "Estimate monthly token usage based on use case and usage patterns"
    args_schema: Type[BaseModel] = TokenEstimationToolInput

    def _run(self, use_case: str, users: int, usage_intensity: str, token_pattern: str) -> Dict[str, Any]:
        """Estimate monthly token usage dynamically based on inputs."""
        try:
            # Use the LLM to analyze the use case and generate estimates
            llm = GoogleGenerativeAI(
                model="gemini/gemini-pro",
                google_api_key=os.getenv("GEMINI_API_KEY", ""),
                temperature=0.1,
                top_p=0.9,
            )
            
            prompt = f"""
            Estimate monthly token usage for:
            - Use Case: {use_case}
            - Users: {users}
            - Usage Intensity: {usage_intensity}
            - Token Pattern: {token_pattern}
            
            Provide your estimate in JSON format with these keys:
            - base_tokens_per_user: The base tokens per user per month
            - pattern_multiplier: The multiplier based on token pattern
            - estimated_monthly_tokens: The final estimated monthly tokens
            
            Only respond with valid JSON, no additional text.
            """
            
            response = llm.invoke(prompt)
            
            try:
                estimation = json.loads(response)
                total_tokens = int(estimation.get("estimated_monthly_tokens", 0))
                
                return {
                    "use_case": use_case,
                    "users": users,
                    "usage_intensity": usage_intensity,
                    "token_pattern": token_pattern,
                    "monthly_tokens": total_tokens,
                    "status": "success"
                }
            except json.JSONDecodeError:
                # Fallback to default estimation if JSON parsing fails
                return self._fallback_estimation(use_case, users, usage_intensity, token_pattern)
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _fallback_estimation(self, use_case: str, users: int, usage_intensity: str, token_pattern: str) -> Dict[str, Any]:
        """Fallback estimation method if LLM fails."""
        # Simple fallback logic - in production this would be more sophisticated
        intensity_factors = {"low": 5000, "medium": 20000, "high": 50000}
        pattern_factors = {
            "short prompts, short responses": 0.7,
            "short prompts, long responses": 1.2,
            "long prompts, short responses": 1.1,
            "long prompts, long responses": 1.5
        }
        
        base_tokens = intensity_factors.get(usage_intensity.lower(), 10000)
        pattern_factor = pattern_factors.get(token_pattern.lower(), 1.0)
        total_tokens = base_tokens * users * pattern_factor
        
        return {
            "use_case": use_case,
            "users": users,
            "usage_intensity": usage_intensity,
            "token_pattern": token_pattern,
            "monthly_tokens": int(total_tokens),
            "status": "success"
        }

# --- LLM Strategy Compass System ---
class LLMStrategyCompass:
    """A complete system for analyzing LLM deployment options using CrewAI."""
    
    def __init__(self):
        """Initialize the system with required models and tools."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model="gemini/gemini-2.0-flash-lite",
            google_api_key=self.gemini_api_key,
            temperature=0.3,
            top_p=0.8,
        )
        
        # Initialize tools
        self.cloud_pricing_tool = CloudPricingTool()
        self.gpu_pricing_tool = GPUPricingTool()
        self.token_estimation_tool = TokenEstimationTool()
        
        # Initialize agents and crew
        self._setup_agents()
        self._setup_crew()
    
    def _setup_agents(self):
        """Set up the agents for the LLM strategy analysis crew."""
        # Prompting Agent
        self.prompting_agent = Agent(
            role="Requirements Gathering Specialist",
            goal="Gather all necessary information about the LLM use case and requirements",
            backstory="You are an expert in understanding technical requirements and translating them into clear specifications for system design.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Token Estimation Agent
        self.token_agent = Agent(
            role="Token Usage Estimator",
            goal="Estimate token usage based on use case and usage patterns",
            backstory="You specialize in predicting token consumption for various LLM applications based on historical patterns and usage characteristics.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.token_estimation_tool]
        )
        
        # Cloud & GPU Pricing Agent
        self.pricing_agent = Agent(
            role="Cost Analysis Specialist",
            goal="Analyze pricing for both cloud and self-hosted LLM solutions",
            backstory="You have deep knowledge of cloud service pricing models and hardware costs for running LLMs on-premises.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.cloud_pricing_tool, self.gpu_pricing_tool]
        )
        
        # Architecture Suggestion Agent
        self.architect = Agent(
            role="Solution Architect",
            goal="Design optimal architectures for both cloud and on-premises solutions",
            backstory="You are an experienced architect who can design systems that meet technical requirements while optimizing for cost and performance.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # NFR Agent
        self.nfr_agent = Agent(
            role="Non-Functional Requirements Analyst",
            goal="Evaluate non-functional aspects like security, compliance, and operational complexity",
            backstory="You specialize in assessing systems based on non-functional requirements like security, compliance, and operational considerations.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Decision-Maker Agent
        self.decision_agent = Agent(
            role="Decision Advisor",
            goal="Provide clear recommendations based on all collected data and analysis",
            backstory="You excel at synthesizing complex technical and financial information into clear, actionable recommendations for decision-makers.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Visualization Agent
        self.viz_agent = Agent(
            role="Data Visualization Specialist",
            goal="Create clear visualizations to represent cost comparisons and other data",
            backstory="You specialize in transforming complex data into easy-to-understand visual representations.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _setup_crew(self):
        """Set up the CrewAI crew for LLM strategy analysis."""
        self.crew = Crew(
            agents=[
                self.prompting_agent,
                self.token_agent,
                self.pricing_agent,
                self.architect,
                self.nfr_agent,
                self.decision_agent,
                self.viz_agent
            ],
            tasks=[],  # We'll add tasks dynamically based on user input
            verbose=True,
            process=Process.sequential
        )
    
    def _create_tasks(self, user_input: Dict[str, Any]):
        """Create tasks for analyzing LLM deployment options."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Task 1: Gather and summarize requirements
        requirements_task = Task(
            description=f"""Analyze and summarize the following LLM deployment requirements as of {current_date}:
            - Use Case: {user_input.get('use_case', 'Not specified')}
            - Users: {user_input.get('users', 'Not specified')}
            - Concurrent Users: {user_input.get('concurrent_users', 'Not specified')}
            - Usage Intensity: {user_input.get('usage_intensity', 'Not specified')}
            - Security Needs: {user_input.get('security_needs', 'Not specified')}
            - System Type: {user_input.get('system_type', 'Not specified')}
            - Token Patterns: {user_input.get('token_pattern', 'Not specified')}
            
            Provide a detailed analysis of these requirements and how they might impact deployment decisions.
            """,
            expected_output="A comprehensive summary of all requirements with analysis of their implications",
            agent=self.prompting_agent,
            context=[]
        )
        
        # Task 2: Estimate token usage
        token_task = Task(
            description=f"""Dynamically estimate monthly token usage for:
            - Use Case: {user_input.get('use_case', 'General')}
            - Users: {user_input.get('users', 100)}
            - Usage Intensity: {user_input.get('usage_intensity', 'medium')}
            - Token Patterns: {user_input.get('token_pattern', 'Not specified')}
            
            Consider the nature of the use case and provide a detailed breakdown of your estimation methodology.
            """,
            expected_output="Detailed token usage estimation including monthly total tokens, tokens per user, and methodology",
            agent=self.token_agent,
            context=[requirements_task]
        )
        
        # Task 3: Analyze pricing options
        pricing_task = Task(
            description=f"""Analyze pricing for both cloud and self-hosted options based on the estimated token usage.
            For cloud options, consider major providers (AWS, GCP, Azure) and appropriate models.
            For self-hosted, consider GPU options that would meet the performance requirements.
            
            Provide a detailed comparison with current market data.
            """,
            expected_output="Comparative pricing analysis showing costs for cloud vs self-hosted options with current market data",
            agent=self.pricing_agent,
            context=[token_task]
        )
        
        # Task 4: Suggest architectures
        architecture_task = Task(
            description=f"""Design optimal architectures for:
            1. Cloud-based solution
            2. Self-hosted/air-gapped solution
            
            Consider the requirements: {user_input}
            
            Provide detailed architecture descriptions and diagrams in text format.
            Explain your design choices based on the requirements.
            """,
            expected_output="Detailed architecture descriptions for both deployment options with rationale",
            agent=self.architect,
            context=[requirements_task, token_task]
        )
        
        # Task 5: Evaluate non-functional requirements
        nfr_task = Task(
            description=f"""Evaluate non-functional aspects for both deployment options:
            - Security
            - Compliance
            - Operational complexity
            - Skill requirements
            - Vendor lock-in risks
            - Long-term maintenance
            
            Provide a comprehensive analysis comparing both approaches.
            """,
            expected_output="Comprehensive NFR analysis comparing cloud and self-hosted options",
            agent=self.nfr_agent,
            context=[requirements_task, architecture_task]
        )
        
        # Task 6: Create visualizations
        viz_task = Task(
            description="""Create visualizations to represent:
            1. Cost comparison over time (1 month, 6 months, 1 year, 3 years)
            2. Performance comparison
            3. Any other relevant metrics
            
            Provide the data and specifications for these visualizations.
            """,
            expected_output="Detailed visualization specifications and data for cost and performance comparisons",
            agent=self.viz_agent,
            context=[pricing_task, token_task]
        )
        
        # Task 7: Make final recommendation
        decision_task = Task(
            description=f"""Create a final recommendation report that includes:
            1. Executive summary
            2. Cost comparison (short-term and long-term)
            3. Architecture options
            4. Security and compliance analysis
            5. Pros and cons of each approach
            6. Visualizations of key data
            7. Final recommendation based on the specific requirements
            
            Make sure to justify your recommendation with data from previous analyses.
            """,
            expected_output="A strictly comprehensive, well-structured final report in markdown format with data-driven recommendations",
            agent=self.decision_agent,
            context=[requirements_task, token_task, pricing_task, architecture_task, nfr_task, viz_task]
        )
        
        return [requirements_task, token_task, pricing_task, architecture_task, nfr_task, viz_task, decision_task]
    
    def analyze_deployment_options(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deployment options and generate a comprehensive report."""
        tasks = self._create_tasks(user_input)
        self.crew.tasks = tasks
        result = self.crew.kickoff()
        
        # Process the result to extract structured data
        try:
            # The result should contain both the report and visualization data
            if isinstance(result, str):
                # If we only got a string, try to parse it
                try:
                    parsed_result = json.loads(result)
                    return parsed_result
                except json.JSONDecodeError:
                    # If it's not JSON, wrap it in a dict
                    return {
                        "report": result,
                        "visualizations": {
                            "cost_comparison": self._generate_default_cost_data(),
                            "performance_comparison": self._generate_default_performance_data()
                        }
                    }
            elif isinstance(result, dict):
                return result
            else:
                return {
                    "report": str(result),
                    "visualizations": {
                        "cost_comparison": self._generate_default_cost_data(),
                        "performance_comparison": self._generate_default_performance_data()
                    }
                }
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")
            return {
                "report": "Error generating report",
                "visualizations": {}
            }
    
    def _generate_default_cost_data(self):
        """Generate default cost data if visualization data isn't provided."""
        return {
            "time_periods": ["1 Month", "6 Months", "1 Year", "3 Years"],
            "cloud_costs": [5000, 30000, 60000, 180000],
            "self_hosted_costs": [20000, 35000, 50000, 90000]
        }
    
    def _generate_default_performance_data(self):
        """Generate default performance data if visualization data isn't provided."""
        return {
            "metrics": ["Throughput", "Latency", "Availability"],
            "cloud_scores": [90, 85, 95],
            "self_hosted_scores": [80, 75, 85]
        }

# --- Chatbot Interaction ---
class ChatbotInteraction:
    def __init__(self):
        """Initialize the chatbot interaction system."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.llm = GoogleGenerativeAI(
            model="gemini/gemini-2.0-flash-lite",
            google_api_key=self.gemini_api_key,
            temperature=0.3,
            top_p=0.8,
        )
        self.questions = self._generate_dynamic_questions()
        self.current_question = 0
        self.user_input = {}
        self.conversation_history = []
    
    def _generate_dynamic_questions(self):
        """Generate dynamic questions based on initial context."""
        # In a real implementation, this could be more sophisticated
        return [
            ("What is your primary use case for the LLM? Please describe it in detail.", "use_case"),
            ("How many active users will be using the system?", "users"),
            ("What is the expected peak number of concurrent users?", "concurrent_users"),
            ("How would you describe the usage intensity? (e.g., occasional use, constant interaction, bursty usage)", "usage_intensity"),
            ("What are your security requirements? (e.g., internal use only, handles PII, needs strict compliance)", "security_needs"),
            ("What type of system will this be? (e.g., web app, mobile app, API service)", "system_type"),
            ("Can you describe the typical token usage pattern? (e.g., short questions with long answers, complex prompts with short responses)", "token_pattern")
        ]
        
    def ask_question(self):
        """Return the current question."""
        if self.current_question < len(self.questions):
            return self.questions[self.current_question][0]
        return None
    
    def process_answer(self, answer: str):
        """Process the user's answer and store it."""
        if self.current_question < len(self.questions):
            question, key = self.questions[self.current_question]
            self.user_input[key] = answer
            
            # Use LLM to refine the answer if needed
            refined_answer = self._refine_answer(key, answer)
            self.conversation_history.append((question, refined_answer))
            
            self.current_question += 1
            return True
        return False
    
    def _refine_answer(self, key: str, answer: str) -> str:
        """Use LLM to refine or categorize the answer when appropriate."""
        if key in ["usage_intensity", "security_needs", "token_pattern"]:
            prompt = f"""
            Categorize this input for our system:
            Field: {key}
            User Input: {answer}
            
            Respond with just the categorized value that best matches, no additional text.
            For usage_intensity: low, medium, or high
            For security_needs: internal-only, pii-handling, or strict-compliance
            For token_pattern: short-short, short-long, long-short, or long-long
            """
            
            try:
                response = self.llm.invoke(prompt)
                return response.strip().lower()
            except:
                return answer
        return answer
    
    def get_conversation_history(self):
        """Return the conversation history."""
        return self.conversation_history
    
    def is_complete(self):
        """Check if all questions have been answered."""
        return self.current_question >= len(self.questions)
    
    def get_user_input(self):
        """Return the collected user input."""
        return self.user_input

# --- Streamlit App ---
def main():
    """Main function for the Streamlit app."""
    st.title("🧭 LLM Strategy Compass")
    st.markdown("""
    This app helps IT teams decide between cloud-based and air-gapped/self-hosted LLM solutions by analyzing:
    - Cost implications
    - Security requirements
    - Scalability needs
    - Infrastructure complexity
    - Long-term sustainability
    """)
    
    with st.sidebar:
        st.header("⚙️ About This App")
        st.markdown("""
        ### How It Works
        This app uses a multi-agent AI system to analyze your LLM deployment needs and provide:
        - Token usage estimates
        - Cost comparisons
        - Architecture suggestions
        - Security analysis
        - Final recommendation
        
        All analysis is performed dynamically by specialized AI agents.
        """)
        st.markdown(f"**Current Date:** {datetime.now().strftime('%B %d, %Y')}")
    
    # Initialize session state for chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatbotInteraction()
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    # Toggle between form and chatbot
    input_method = st.radio(
        "Choose input method:",
        ["📋 Fill out a form", "💬 Chat with an assistant"],
        horizontal=True
    )
    
    if input_method == "📋 Fill out a form":
        # Form-based input
        with st.form("user_input_form"):
            st.subheader("📋 Deployment Requirements")
            
            col1, col2 = st.columns(2)
            with col1:
                use_case = st.text_area(
                    "Describe your use case",
                    placeholder="e.g., Customer support chatbot that handles 100 queries per day..."
                )
                
                users = st.number_input(
                    "Number of Active Users",
                    min_value=1,
                    max_value=100000,
                    value=100,
                    step=1
                )
                
                security_needs = st.selectbox(
                    "Security Requirements",
                    options=["Internal-only", "PII Handling", "Strict Compliance"],
                    index=0
                )
                
            with col2:
                concurrent_users = st.number_input(
                    "Concurrent Users (Peak)",
                    min_value=1,
                    max_value=10000,
                    value=10,
                    step=1
                )
                
                usage_intensity = st.select_slider(
                    "Usage Intensity",
                    options=["Low", "Medium", "High"],
                    value="Medium"
                )
                
                system_type = st.selectbox(
                    "System Type",
                    options=["Web App", "Desktop", "Mobile", "API Service"],
                    index=0
                )
            
            token_pattern = st.selectbox(
                "Token Usage Pattern",
                options=[
                    "Short prompts with short responses",
                    "Short prompts with long responses",
                    "Long prompts with short responses",
                    "Long prompts with long responses"
                ],
                index=0
            )
            
            submitted = st.form_submit_button("🚀 Analyze Deployment Options")
        
        if submitted:
            user_input = {
                "use_case": use_case,
                "users": users,
                "concurrent_users": concurrent_users,
                "usage_intensity": usage_intensity.lower(),
                "security_needs": security_needs.lower().replace(" ", "-"),
                "system_type": system_type,
                "token_pattern": token_pattern.lower().replace(" ", "-")
            }
            
            run_analysis(user_input)
    
    else:
        # Chatbot-style interaction
        st.subheader("💬 Chat with our Requirements Assistant")
        
        # Display conversation history
        chat_container = st.container()
        
        # Initialize chat history if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your answer here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process the answer
            st.session_state.chatbot.process_answer(prompt)
            
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Get next question or proceed to analysis
            if st.session_state.chatbot.is_complete():
                # All questions answered - proceed to analysis
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown("Thank you! I have all the information I need. Analyzing your deployment options now...")
                
                # Run analysis with collected input
                user_input = st.session_state.chatbot.get_user_input()
                run_analysis(user_input)
            else:
                # Ask next question
                next_question = st.session_state.chatbot.ask_question()
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": next_question})
                
                # Display assistant message
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(next_question)

def run_analysis(user_input: Dict[str, Any]):
    """Run the analysis and display results."""
    progress_container = st.empty()
    result_container = st.empty()
    
    with progress_container.container():
        st.subheader("📊 Analysis in Progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            "Gathering Requirements",
            "Estimating Token Usage",
            "Analyzing Pricing Options",
            "Designing Architectures",
            "Evaluating Non-Functional Requirements",
            "Creating Visualizations",
            "Generating Final Recommendation"
        ]
        
        for i, step in enumerate(steps):
            progress_bar.progress((i + 1) * (100 // len(steps)))
            status_text.text(f"Step {i+1}/{len(steps)}: {step}")
            time.sleep(0.5)
        
        try:
            status_text.text("Running analysis with AI agents...")
            strategy_compass = LLMStrategyCompass()
            result = strategy_compass.analyze_deployment_options(user_input)
            st.session_state.analysis_result = result
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return
    
    progress_container.empty()
    
    if st.session_state.analysis_result:
        display_results(st.session_state.analysis_result)

def display_results(result: Dict[str, Any]):
    """Display the analysis results."""
    with st.container():
        st.markdown("## LLM Deployment Strategy Recommendation")
        st.markdown(f"**Report Generated on:** {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
        
        # Display the full report
        st.markdown(result.get("report", "No report generated"))
        
        # Display visualizations
        viz_data = result.get("visualizations", {})
        
        if "cost_comparison" in viz_data:
            st.markdown("### Cost Comparison Over Time")
            cost_data = viz_data["cost_comparison"]
            
            df = pd.DataFrame({
                "Time Period": cost_data.get("time_periods", []),
                "Cloud Cost ($)": cost_data.get("cloud_costs", []),
                "Self-Hosted Cost ($)": cost_data.get("self_hosted_costs", [])
            })
            
            fig = px.line(
                df,
                x="Time Period",
                y=["Cloud Cost ($)", "Self-Hosted Cost ($)"],
                title="Projected Cost Comparison",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "performance_comparison" in viz_data:
            st.markdown("### Performance Comparison")
            perf_data = viz_data["performance_comparison"]
            
            df = pd.DataFrame({
                "Metric": perf_data.get("metrics", []),
                "Cloud Solution": perf_data.get("cloud_scores", []),
                "Self-Hosted Solution": perf_data.get("self_hosted_scores", [])
            })
            
            fig = px.bar(
                df,
                x="Metric",
                y=["Cloud Solution", "Self-Hosted Solution"],
                title="Performance Comparison",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.download_button(
            label="📥 Download Full Report",
            data=result.get("report", ""),
            file_name=f"llm_deployment_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()