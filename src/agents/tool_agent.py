"""
Autonomous Tool Agent with:
- Dynamic tool discovery and selection
- Tool composition and chaining
- Self-correction capabilities
- Confidence-based decision making
"""
import json
import re
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum

from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from src import config


class ToolStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """Structured tool execution result."""
    status: ToolStatus
    data: Any
    error_message: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Tool:
    """Enhanced tool definition with metadata."""
    name: str
    function: Callable
    description: str
    input_types: List[str]  # e.g., ['song_name', 'era_name']
    output_type: str
    success_rate: float = 1.0  # Track reliability
    avg_execution_time: float = 0.0


class AutonomousToolAgent:
    """
    Tool agent that:
    1. Analyzes questions to determine needed operations
    2. Discovers and selects appropriate tools
    3. Chains tools when necessary
    4. Self-corrects on failures
    """
    
    def __init__(self, model: str = config.REASONING_MODEL):
        if config.USE_OPENAI:
            self.client = OpenAIClient()
        else:
            self.client = OllamaClient(model=model)
        
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict] = []
        self.df = None
        self.similarity_results = None
        
    def register_tool(self, tool: Tool):
        """Register a tool with full metadata."""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")
    
    def _analyze_question(self, question: str) -> Dict[str, Any]:
        """
        STEP 1: Analyze question to understand intent and requirements.
        
        This is the first autonomous step - understanding WHAT is being asked
        before deciding HOW to answer it.
        """
        analysis_prompt = f"""Analyze this question to determine how to answer it:

Question: "{question}"

Available tool categories:
- get_song_info: Get details about a specific song
- get_era_stats: Get statistics for a specific era
- find_similar_songs: Find songs similar to a given song
- compare_eras: Compare two eras
- search_by_feature: Search songs by audio features

Provide your analysis in this JSON format:
{{
  "intent": "what the user wants to know",
  "required_operations": ["list", "of", "operations", "needed"],
  "entities": {{"song_names": [], "era_names": [], "features": []}},
  "complexity": "simple|moderate|complex",
  "requires_chaining": true/false,
  "suggested_approach": "step-by-step plan"
}}

Think step by step about what information is needed and how to get it.
"""
        
        self.client.reset_conversation()
        response = self.client.chat_interactive(analysis_prompt, system_prompt="You are a query analysis expert. Respond ONLY with valid JSON.")
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                print(f"\n[ANALYSIS] Intent: {analysis.get('intent')}")
                print(f"[ANALYSIS] Complexity: {analysis.get('complexity')}")
                print(f"[ANALYSIS] Requires chaining: {analysis.get('requires_chaining')}")
                return analysis
        except json.JSONDecodeError:
            print(f"[WARNING] Could not parse analysis, using fallback")
        
        # Fallback to simple analysis
        return {
            "intent": "answer question",
            "required_operations": ["unknown"],
            "entities": {},
            "complexity": "moderate",
            "requires_chaining": False,
            "suggested_approach": "Use available tools"
        }
    
    def _select_tools(self, analysis: Dict[str, Any]) -> List[str]:
        """
        STEP 2: Select appropriate tools based on analysis.
        
        This demonstrates REASONING about which tools to use rather than
        blindly having access to all tools.
        """
        required_ops = analysis.get('required_operations', [])
        entities = analysis.get('entities', {})
        
        selection_prompt = f"""Based on this analysis, which tools should be used?

Analysis:
- Intent: {analysis.get('intent')}
- Required operations: {required_ops}
- Entities found: {entities}
- Complexity: {analysis.get('complexity')}

Available tools:
"""
        for name, tool in self.tools.items():
            selection_prompt += f"- {name}: {tool.description}\n"
            selection_prompt += f"  Success rate: {tool.success_rate:.2f}\n"
        
        selection_prompt += """
Select the tools needed and explain your reasoning.
Format: TOOLS: [tool1, tool2, ...]
Then explain why these tools in 1-2 sentences.
"""
        
        self.client.reset_conversation()
        response = self.client.chat_interactive(selection_prompt)

        # Extract selected tools
        tools_match = re.search(r'TOOLS:\s*\[(.*?)\]', response)
        if tools_match:
            tools_str = tools_match.group(1)
            selected = [t.strip().strip('"\'') for t in tools_str.split(',')]
            selected = [t for t in selected if t in self.tools]
            print(f"\n[TOOL SELECTION] Selected: {selected}")
            print(f"[REASONING] {response.split('TOOLS:')[1] if 'TOOLS:' in response else response[:100]}")
            return selected
        
        return []
    
    def _plan_execution(self, analysis: Dict[str, Any], selected_tools: List[str]) -> List[Dict]:
        """
        STEP 3: Create execution plan (tool chaining logic).
        
        This is where the agent demonstrates PLANNING - organizing tools
        into a sequence rather than executing randomly.
        """
        if not analysis.get('requires_chaining') and len(selected_tools) <= 1:
            # Simple case - single tool execution
            if selected_tools:
                tool_name = selected_tools[0]
                entities = analysis.get('entities', {})
                args = []
                tool = self.tools.get(tool_name)
                if tool:
                    for input_type in tool.input_types:
                        if 'song' in input_type and entities.get('song_names'):
                            args.append(entities['song_names'][0])
                        elif 'era' in input_type and entities.get('era_names'):
                            args.append(entities['era_names'][0])
                        elif 'feature' in input_type and entities.get('features'):
                            args.append(entities['features'][0])
                return [{"tool": tool_name, "args": args, "depends_on": None}]
            return []
        
        planning_prompt = f"""Create an execution plan for these tools:

Selected tools: {selected_tools}
Goal: {analysis.get('intent')}
Suggested approach: {analysis.get('suggested_approach')}

Create a step-by-step execution plan. For each step specify:
1. Which tool to use
2. What arguments it needs
3. Whether it depends on previous steps

Format as JSON array:
[
  {{"step": 1, "tool": "tool_name", "args": ["arg1"], "depends_on": null, "purpose": "why"}},
  {{"step": 2, "tool": "tool_name", "args": ["use_result_from_step_1"], "depends_on": 1, "purpose": "why"}}
]
"""
        
        self.client.reset_conversation()
        response = self.client.chat_interactive(planning_prompt, system_prompt="You are an execution planner. Respond with valid JSON.")
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                print(f"\n[EXECUTION PLAN] {len(plan)} steps")
                for step in plan:
                    print(f"  Step {step.get('step')}: {step.get('tool')} - {step.get('purpose')}")
                return plan
        except json.JSONDecodeError:
            print("[WARNING] Could not parse plan, using sequential execution")
        
        # Fallback - execute tools sequentially
        return [{"tool": tool, "args": [], "depends_on": None} for tool in selected_tools]
    
    def _execute_with_retry(self, tool_name: str, args: List[Any], max_retries: int = 2) -> ToolResult:
        """
        STEP 4: Execute tool with self-correction.
        
        This demonstrates ERROR HANDLING and SELF-CORRECTION - if something
        fails, the agent tries to fix it rather than giving up.
        """
        for attempt in range(max_retries + 1):
            try:
                result = self.execute_tool(tool_name, args)
                
                # Check if result indicates success
                if isinstance(result, str) and "Error" in result:
                    if attempt < max_retries:
                        print(f"[RETRY] Attempt {attempt + 1} failed: {result}")
                        # Try to self-correct
                        correction = self._attempt_correction(tool_name, args, result)
                        if correction:
                            args = correction
                            continue
                    return ToolResult(ToolStatus.FAILURE, result, result, 0.0)
                
                return ToolResult(ToolStatus.SUCCESS, result, None, 1.0)
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"[RETRY] Attempt {attempt + 1} exception: {str(e)}")
                    continue
                return ToolResult(ToolStatus.FAILURE, None, str(e), 0.0)
        
        return ToolResult(ToolStatus.FAILURE, None, "Max retries exceeded", 0.0)
    
    def _attempt_correction(self, tool_name: str, args: List[Any], error: str) -> Optional[List[Any]]:
        """
        Autonomous error correction - agent tries to fix issues itself.
        
        This is SELF-CORRECTION in action - reasoning about what went wrong
        and how to fix it.
        """
        correction_prompt = f"""A tool execution failed. Can you suggest a correction?

Tool: {tool_name}
Arguments: {args}
Error: {error}

Common issues:
- Song/era name not found: Try variations or partial matches
- Invalid parameters: Check data types and ranges
- Missing data: Try alternative approach

Suggest corrected arguments in format: CORRECTED_ARGS: [arg1, arg2, ...]
Or respond with: NO_CORRECTION_POSSIBLE
"""
        
        self.client.reset_conversation()
        response = self.client.chat_interactive(correction_prompt)
        
        if "NO_CORRECTION_POSSIBLE" in response:
            return None
        
        match = re.search(r'CORRECTED_ARGS:\s*\[(.*?)\]', response)
        if match:
            corrected = [arg.strip().strip('"\'') for arg in match.group(1).split(',')]
            print(f"[SELF-CORRECTION] Trying: {corrected}")
            return corrected
        
        return None
    
    def execute_tool(self, tool_name: str, args: List[Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            func = self.tools[tool_name].function
            result = func(*args)

            # Update tool statistics — treat error strings as failures
            if isinstance(result, str) and result.startswith("Error"):
                self.tools[tool_name].success_rate *= 0.9
            else:
                self.tools[tool_name].success_rate = (
                    self.tools[tool_name].success_rate * 0.9 + 0.1
                )

            return result
        except Exception as e:
            self.tools[tool_name].success_rate *= 0.9
            return f"Error executing {tool_name}: {str(e)}"
    
    def ask(self, question: str, max_iterations: int = 5) -> str:
        """
        Main autonomous reasoning loop.
        
        This ties everything together in a AUTONOMOUS WORKFLOW:
        Analyze → Select → Plan → Execute → Synthesize
        """
        print(f"\n{'='*80}")
        print(f"AUTONOMOUS TOOL AGENT - PROCESSING QUERY")
        print(f"{'='*80}")
        
        # STEP 1: Understand the question
        analysis = self._analyze_question(question)
        
        # STEP 2: Select appropriate tools
        selected_tools = self._select_tools(analysis)
        
        if not selected_tools:
            return "I don't have the appropriate tools to answer this question."
        
        # STEP 3: Plan execution
        execution_plan = self._plan_execution(analysis, selected_tools)
        
        # STEP 4: Execute plan with results tracking
        print(f"\n{'='*80}")
        print(f"EXECUTING PLAN")
        print(f"{'='*80}")
        
        results = {}
        for i, step in enumerate(execution_plan):
            tool_name = step.get('tool')
            args = step.get('args', [])
            
            # Handle dependencies (tool chaining)
            if step.get('depends_on') and step['depends_on'] in results:
                # Replace placeholder args with actual result from prior step
                prev_result = results[step['depends_on']]
                if isinstance(prev_result.data, str):
                    args = [prev_result.data]
            
            print(f"\nStep {i+1}: Executing {tool_name}({args})")
            result = self._execute_with_retry(tool_name, args)
            results[i+1] = result
            
            if result.status == ToolStatus.FAILURE:
                print(f"[WARNING] Step {i+1} failed: {result.error_message}")
                # Try to continue with remaining steps if possible
        
        # STEP 5: Synthesize results into final answer
        print(f"\n{'='*80}")
        print(f"SYNTHESIZING ANSWER")
        print(f"{'='*80}")
        
        return self._synthesize_answer(question, analysis, results)
    
    def _synthesize_answer(self, question: str, analysis: Dict, results: Dict[int, ToolResult]) -> str:
        """
        Final synthesis - combine tool results into coherent answer.
        
        This demonstrates REASONING about results rather than just returning raw data.
        """
        synthesis_prompt = f"""Synthesize a final answer based on these tool execution results.

Original question: {question}
Intent: {analysis.get('intent')}

Tool execution results:
"""
        
        for step, result in results.items():
            synthesis_prompt += f"\nStep {step}:"
            if result.status == ToolStatus.SUCCESS:
                synthesis_prompt += f"\nSuccess (confidence: {result.confidence})\n{result.data}\n"
            else:
                synthesis_prompt += f"\nFailed: {result.error_message}\n"
        
        synthesis_prompt += """
Provide a clear, concise answer to the original question based on these results.
If some steps failed, work with the successful results.
Be specific and cite the data when possible.
"""
        
        self.client.reset_conversation()
        answer = self.client.chat_interactive(synthesis_prompt)
        return answer
    
    def load_analysis_data(self, df, similarity_results):
        """Load data for tools to access."""
        self.df = df
        self.similarity_results = similarity_results


# Tool implementations (same as before but wrapped in Tool objects)
def create_song_info_tool(agent) -> Tool:
    def get_song_info(song_name: str) -> str:
        if agent.df is None:
            return "Error: Data not loaded"
        
        matches = agent.df[agent.df['Song_Name'].str.lower() == song_name.lower()]
        if matches.empty:
            # Try partial match
            partial = agent.df[agent.df['Song_Name'].str.lower().str.contains(song_name.lower())]
            if not partial.empty:
                return f"Error: Exact match not found. Did you mean: {partial['Song_Name'].iloc[0]}?"
            return f"Error: Song '{song_name}' not found"
        
        song = matches.iloc[0]
        return f"""Song: {song['Song_Name']}
Album: {song['Album']}
Era: {song.get('era', 'Unknown')}
Danceability: {song.get('danceability', 0):.2f}
Energy: {song.get('energy', 0):.2f}
Valence: {song.get('valence', 0):.2f}
Topic: {song.get('dominant_topic', 'Unknown')}"""
    
    return Tool(
        name="get_song_info",
        function=get_song_info,
        description="Get detailed information about a specific song by name",
        input_types=["song_name"],
        output_type="song_details"
    )

def create_era_stats_tool(agent) -> Tool:
    def get_era_stats(era_name: str) -> str:
        if agent.df is None:
            return "Error: Data not loaded"
        era_df = agent.df[agent.df['era'].str.lower() == era_name.lower()]
        if era_df.empty:
            return f"Error: Era '{era_name}' not found. Available eras: {agent.df['era'].unique().tolist()}"
        
        return f"""Era: {era_name}
Songs: {len(era_df)}
Avg Energy: {era_df['energy'].mean():.2f}
Avg Valence: {era_df['valence'].mean():.2f}
Avg Danceability: {era_df['danceability'].mean():.2f}
Avg Tempo: {era_df['tempo'].mean():.1f} BPM"""
    
    return Tool(
        name="get_era_stats",
        function=get_era_stats,
        description="Get statistical summary for a specific era",
        input_types=["era_name"],
        output_type="era_statistics"
    )


def interactive_autonomous_agent():
    """Interactive session with autonomous tool agent."""
    import numpy as np
    from src.data_loading import load_and_merge_data
    from src.similarity_analysis import create_hybrid_similarity_system
    
    print("="*80)
    print("AUTONOMOUS TOOL AGENT")
    print("="*80)
    print(f"Model: {config.REASONING_MODEL}\n")
    
    print("Loading data...")
    df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    
    print("Creating similarity system...")
    sim_results = create_hybrid_similarity_system(df)
    
    print("Initializing autonomous agent...")
    agent = AutonomousToolAgent()
    agent.load_analysis_data(sim_results['df'], sim_results)
    
    # Register tools
    agent.register_tool(create_song_info_tool(agent))
    agent.register_tool(create_era_stats_tool(agent))
    
    print("\nAgent ready with autonomous capabilities!")
    print("\nThe agent will:")
    print("  1. Analyze your question")
    print("  2. Select appropriate tools")
    print("  3. Plan execution strategy")
    print("  4. Execute with self-correction")
    print("  5. Synthesize final answer")
    print("\nTry complex questions that require multiple steps!")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            answer = agent.ask(question)
            print(f"\n{'-'*80}")
            print(f"FINAL ANSWER:")
            print(f"{'-'*80}")
            print(answer)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_autonomous_agent()
