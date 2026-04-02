"""
Autonomous Multi-Agent System with:
- Dynamic agent assembly based on question type
- Inter-agent debate and dialogue
- Iterative refinement
- Confidence-based routing
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from src import config


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AgentResponse:
    """Structured agent response with metadata."""
    content: str
    confidence: Confidence
    key_points: List[str]
    uncertainties: List[str]
    questions_for_others: List[str] = None
    
    def __post_init__(self):
        if self.questions_for_others is None:
            self.questions_for_others = []


class BaseAnalystAgent:
    """
    Enhanced base agent with confidence assessment and questioning ability.
    
    KEY IMPROVEMENT: Agents now express confidence and can ask questions
    to other agents, enabling debate and collaboration.
    """
    
    def __init__(self, role: str, expertise: str, model: str = config.MODEL):
        self.role = role
        self.expertise = expertise
        if config.USE_OPENAI:
            self.client = OpenAIClient()
        else:
            self.client = OllamaClient(model=model)
    
    def analyze(self, song_data: Dict, context: Optional[str] = None) -> AgentResponse:
        """Analyze with confidence and uncertainty tracking."""
        raise NotImplementedError
    
    def respond_to_question(self, question: str, original_analysis: AgentResponse, context: str) -> str:
        """
        Respond to questions from other agents.
        
        This enables INTER-AGENT DIALOGUE - agents can challenge and
        question each other's analyses.
        """
        prompt = f"""Another analyst has a question about your analysis.

Your role: {self.role}
Your original analysis: {original_analysis.content}

Question from colleague: {question}

Additional context: {context}

Provide a thoughtful response that either:
1. Clarifies your position with additional evidence
2. Acknowledges the concern and revises your view
3. Respectfully disagrees with supporting reasoning

Be specific and analytical.
"""
        if config.USE_OPENAI:
            return self.client.chat_interactive(prompt)
        else:
            return self.client.generate(prompt, max_tokens=300)
    
    def _parse_response_with_confidence(self, raw_response: str) -> AgentResponse:
        """
        Parse LLM response into structured format with confidence.
        
        This extracts METADATA from responses - not just content but also
        confidence levels and uncertainties.
        """
        # Extract confidence markers
        confidence = Confidence.MEDIUM
        if any(word in raw_response.lower() for word in ['definitely', 'clearly', 'certainly', 'strongly']):
            confidence = Confidence.HIGH
        elif any(word in raw_response.lower() for word in ['unclear', 'uncertain', 'possibly', 'might']):
            confidence = Confidence.LOW
        
        # Extract key points (simplified - look for bullet points or numbered lists)
        key_points = []
        for line in raw_response.split('\n'):
            if line.strip().startswith(('-', '•', '1.', '2.', '3.')):
                key_points.append(line.strip())
        
        # Extract uncertainties (look for uncertainty phrases)
        uncertainties = []
        uncertainty_markers = ['unclear', 'uncertain', 'not sure', 'difficult to determine', 'ambiguous']
        for marker in uncertainty_markers:
            if marker in raw_response.lower():
                # Extract sentence containing uncertainty
                sentences = raw_response.split('.')
                for sent in sentences:
                    if marker in sent.lower():
                        uncertainties.append(sent.strip())
        
        return AgentResponse(
            content=raw_response,
            confidence=confidence,
            key_points=key_points[:3],  # Top 3
            uncertainties=uncertainties[:2]  # Top 2
        )


class LyricalAnalystAgent(BaseAnalystAgent):
    """Lyrical analyst with enhanced capabilities."""
    
    def __init__(self, model: str = config.MODEL):
        super().__init__(
            role="Lyrical Analyst",
            expertise="songwriting patterns, narrative techniques, sentiment analysis",
            model=model
        )
    
    def analyze(self, song_data: Dict, context: Optional[str] = None) -> AgentResponse:
        prompt = f"""You are a lyrical analyst. Analyze this song's writing:

Song: {song_data['Song_Name']} ({song_data['Album']})

Lyrical metrics:
- Sentiment polarity: {song_data.get('polarity', 0):.2f}
- Subjectivity: {song_data.get('subjectivity', 0):.2f}
- Dominant theme: {song_data.get('dominant_topic', 'Unknown')}
- Word complexity: {song_data.get('avg_word_length', 0):.2f}

{f'Additional context: {context}' if context else ''}

Provide analysis covering:
1. Emotional tone and narrative approach
2. Lyrical complexity and style
3. How this fits Taylor's songwriting evolution

Express your confidence level (high/medium/low) and note any uncertainties.
Be specific and analytical (3-4 sentences).
"""
        
        if config.USE_OPENAI:
            response = self.client.chat_interactive(prompt)
        else:
            response = self.client.generate(prompt, max_tokens=400)
        
        return self._parse_response_with_confidence(response)


class MusicalAnalystAgent(BaseAnalystAgent):
    """Musical analyst with enhanced capabilities."""
    
    def __init__(self, model: str = config.MODEL):
        super().__init__(
            role="Musical Analyst",
            expertise="audio production, sonic characteristics, arrangement",
            model=model
        )
    
    def analyze(self, song_data: Dict, context: Optional[str] = None) -> AgentResponse:
        prompt = f"""You are a musical/production analyst. Analyze these audio features:

Song: {song_data['Song_Name']} ({song_data['Album']})

Audio features:
- Energy: {song_data.get('energy', 0):.2f}
- Valence: {song_data.get('valence', 0):.2f}
- Danceability: {song_data.get('danceability', 0):.2f}
- Tempo: {song_data.get('tempo', 0):.0f} BPM
- Acousticness: {song_data.get('acousticness', 0):.2f}

{f'Additional context: {context}' if context else ''}

Analyze:
1. Production style and sonic palette
2. Mood and energy characteristics
3. How this fits Taylor's musical evolution

Express confidence and note uncertainties (3-4 sentences).
"""
        
        if config.USE_OPENAI:
            response = self.client.chat_interactive(prompt)
        else:
            response = self.client.generate(prompt, max_tokens=400)
        
        return self._parse_response_with_confidence(response)


class ContextualAnalystAgent(BaseAnalystAgent):
    """Contextual analyst with enhanced capabilities."""
    
    def __init__(self, model: str = config.MODEL):
        super().__init__(
            role="Contextual Analyst",
            expertise="artistic evolution, era characteristics, career context",
            model=model
        )
    
    def analyze(self, song_data: Dict, context: Optional[str] = None) -> AgentResponse:
        prompt = f"""You are a music historian analyzing Taylor Swift's evolution.

Song: {song_data['Song_Name']} ({song_data['Album']})
Era: {song_data.get('era', 'Unknown')}

{f'Additional context: {context}' if context else ''}

Analyze:
1. How this song fits within its era
2. Its place in Taylor's artistic trajectory
3. What it reveals about her evolution as an artist

Express confidence and uncertainties (3-4 sentences).
"""
        
        if config.USE_OPENAI:
            response = self.client.chat_interactive(prompt)
        else:
            response = self.client.generate(prompt, max_tokens=400)
        
        return self._parse_response_with_confidence(response)


class AutonomousOrchestrator:
    """
    Orchestrator with:
    - Dynamic agent assembly
    - Inter-agent debate facilitation
    - Iterative refinement
    - Confidence-based routing
    """
    
    def __init__(self, model: str = config.REASONING_MODEL):
        if config.USE_OPENAI:
            self.client = OpenAIClient()
        else:
            self.client = OllamaClient(model=model)
        
        # Agent pool - spawn agents as needed
        self.available_agents = {
            'lyrical': LyricalAnalystAgent,
            'musical': MusicalAnalystAgent,
            'contextual': ContextualAnalystAgent
        }
        
        self.active_agents: Dict[str, BaseAnalystAgent] = {}
    
    def _determine_needed_agents(self, question: str, song_data: Dict) -> List[str]:
        """
        DYNAMIC AGENT ASSEMBLY - decide which agents to activate.
        
        This demonstrates ADAPTIVE RESOURCE ALLOCATION - only using
        the agents that are actually needed for the task.
        """
        analysis_prompt = f"""Determine which specialist analysts are needed for this question:

Question: {question}
Song: {song_data.get('Song_Name')} ({song_data.get('Album')})

Available specialists:
- lyrical: Analyzes songwriting, themes, narrative, sentiment
- musical: Analyzes production, sound, audio features, mood
- contextual: Analyzes era context, artistic evolution, career placement

Which specialists are needed? Consider:
- What aspects of the song does the question address?
- Which specialists have relevant expertise?
- Avoid unnecessary specialists to be efficient

Respond with: NEEDED: [agent1, agent2, ...]
Then briefly explain why (1 sentence per agent).
"""
        
        response = self.client.chat_interactive(analysis_prompt)
        
        # Parse response
        import re
        match = re.search(r'NEEDED:\s*\[(.*?)\]', response)
        if match:
            agents = [a.strip().strip('"\'') for a in match.group(1).split(',')]
            agents = [a for a in agents if a in self.available_agents]
            print(f"\n[ORCHESTRATOR] Assembling agents: {agents}")
            print(f"[REASONING] {response.split('NEEDED:')[1].split('Explain')[0][:200] if 'NEEDED:' in response else 'Using heuristics'}")
            return agents
        
        # Fallback - use all agents
        return list(self.available_agents.keys())
    
    def _activate_agents(self, agent_types: List[str]):
        """Instantiate needed agents."""
        self.active_agents = {}
        for agent_type in agent_types:
            if agent_type in self.available_agents:
                self.active_agents[agent_type] = self.available_agents[agent_type]()
                print(f"  ✓ Activated {agent_type} analyst")
    
    def _facilitate_debate(self, analyses: Dict[str, AgentResponse], song_data: Dict) -> Dict[str, str]:
        """
        INTER-AGENT DEBATE - facilitate discussion between agents.
        
        This is where agents CHALLENGE each other and refine their thinking
        through dialogue, not just parallel analysis.
        """
        print(f"\n[ORCHESTRATOR] Facilitating inter-agent debate...")
        
        # Find contradictions or uncertainties
        contradictions = self._identify_contradictions(analyses)
        
        if not contradictions:
            print("  No major contradictions found")
            return {}
        
        print(f"  Found {len(contradictions)} points of debate")
        
        debate_results = {}
        
        for contradiction in contradictions[:2]:  # Limit to 2 debates for efficiency
            print(f"\n  Debate topic: {contradiction['topic']}")
            
            agent1_type = contradiction['agent1']
            agent2_type = contradiction['agent2']
            
            # Agent 1 poses question to Agent 2
            question = f"You said '{contradiction['point1']}', but I observed '{contradiction['point2']}'. How do you reconcile this?"
            
            print(f"    {agent1_type} → {agent2_type}: Questioning...")
            response = self.active_agents[agent2_type].respond_to_question(
                question,
                analyses[agent2_type],
                f"Song: {song_data.get('Song_Name')}"
            )
            
            debate_results[f"{agent1_type}_questions_{agent2_type}"] = response
            print(f"    Response received")
        
        return debate_results
    
    def _identify_contradictions(self, analyses: Dict[str, AgentResponse]) -> List[Dict]:
        """
        Identify contradictions between agent analyses.
        
        This uses REASONING to detect when agents disagree, which is
        valuable information that should be explored.
        """
        # Use LLM to identify contradictions
        contradiction_prompt = """Identify contradictions or tensions between these analyses:

"""
        for agent_type, response in analyses.items():
            contradiction_prompt += f"\n{agent_type.upper()} ANALYST:\n{response.content}\n"
        
        contradiction_prompt += """
Are there any contradictions, tensions, or interesting disagreements?
For each, specify:
- What the contradiction is about
- Which agents disagree
- Why this is interesting

Format: If contradictions exist, list them. If not, say "NO_CONTRADICTIONS"
"""
        
        response = self.client.chat_interactive(contradiction_prompt)
        
        if "NO_CONTRADICTIONS" in response:
            return []
        
        # Simplified parsing - in production you'd want structured output
        contradictions = []
        if "lyrical" in response.lower() and "musical" in response.lower():
            contradictions.append({
                'topic': 'lyrical vs musical sentiment',
                'agent1': 'lyrical',
                'agent2': 'musical',
                'point1': 'lyrical sentiment',
                'point2': 'musical mood'
            })
        
        return contradictions
    
    def _iterative_refinement(self, analyses: Dict[str, AgentResponse], 
                             debate_results: Dict[str, str], 
                             song_data: Dict) -> Dict[str, AgentResponse]:
        """
        ITERATIVE REFINEMENT - agents update their analyses based on debate.
        
        This demonstrates LEARNING and ADAPTATION - agents don't just
        defend their initial position, they refine it based on new information.
        """
        print(f"\n[ORCHESTRATOR] Round 2 - Refinement based on debate...")
        
        if not debate_results:
            return analyses
        
        refined_analyses = {}
        
        for agent_type, agent in self.active_agents.items():
            # Check if this agent participated in debate
            relevant_debates = [v for k, v in debate_results.items() if agent_type in k]
            
            if relevant_debates:
                refinement_prompt = f"""Review your initial analysis considering colleague feedback:

Your initial analysis:
{analyses[agent_type].content}

Colleague feedback from debate:
{relevant_debates[0]}

Refine your analysis if needed. You may:
1. Strengthen your position with additional reasoning
2. Acknowledge valid points and adjust your view
3. Identify nuances you initially missed

Provide refined analysis (3-4 sentences).
"""
                
                if config.USE_OPENAI:
                    refined = agent.client.chat_interactive(refinement_prompt)
                else:
                    refined = agent.client.generate(refinement_prompt, max_tokens=400)
                
                refined_analyses[agent_type] = agent._parse_response_with_confidence(refined)
                print(f"  ✓ {agent_type} refined their analysis")
            else:
                refined_analyses[agent_type] = analyses[agent_type]
        
        return refined_analyses
    
    def _confidence_based_weighting(self, analyses: Dict[str, AgentResponse]) -> Dict[str, float]:
        """
        Weight agent contributions by confidence.
        
        This implements CONFIDENCE-BASED ROUTING - giving more weight to
        agents who are more certain about their analysis.
        """
        weights = {}
        confidence_values = {
            Confidence.HIGH: 1.0,
            Confidence.MEDIUM: 0.7,
            Confidence.LOW: 0.4
        }
        
        for agent_type, response in analyses.items():
            weights[agent_type] = confidence_values[response.confidence]
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        print(f"\n[ORCHESTRATOR] Confidence-based weights: {weights}")
        return weights
    
    def analyze_song(self, question: str, song_data: Dict) -> Dict[str, any]:
        """
        Main autonomous multi-agent workflow.
        
        COMPLETE AUTONOMOUS WORKFLOW:
        1. Determine needed agents (dynamic assembly)
        2. Initial analysis round
        3. Identify contradictions
        4. Facilitate debate
        5. Refinement round
        6. Confidence-based synthesis
        """
        print(f"\n{'='*80}")
        print(f"AUTONOMOUS MULTI-AGENT ANALYSIS")
        print(f"{'='*80}")
        
        # STEP 1: Dynamic agent assembly
        needed_agents = self._determine_needed_agents(question, song_data)
        self._activate_agents(needed_agents)
        
        # STEP 2: Initial analysis round
        print(f"\n[ROUND 1] Initial analyses...")
        initial_analyses = {}
        for agent_type, agent in self.active_agents.items():
            print(f"  {agent_type} analyst working...")
            analysis = agent.analyze(song_data)
            initial_analyses[agent_type] = analysis
            print(f"    Confidence: {analysis.confidence.value}")
            if analysis.uncertainties:
                print(f"    Uncertainties: {len(analysis.uncertainties)}")
        
        # STEP 3: Facilitate debate
        debate_results = self._facilitate_debate(initial_analyses, song_data)
        
        # STEP 4: Iterative refinement
        final_analyses = self._iterative_refinement(initial_analyses, debate_results, song_data)
        
        # STEP 5: Confidence-based weighting
        weights = self._confidence_based_weighting(final_analyses)
        
        # STEP 6: Synthesis
        print(f"\n[ORCHESTRATOR] Synthesizing final answer...")
        synthesis = self._synthesize_with_weights(question, final_analyses, weights, debate_results)
        
        return {
            'initial_analyses': initial_analyses,
            'debate_results': debate_results,
            'final_analyses': final_analyses,
            'confidence_weights': weights,
            'synthesis': synthesis
        }
    
    def _synthesize_with_weights(self, question: str, analyses: Dict[str, AgentResponse], 
                                 weights: Dict[str, float], debate_results: Dict) -> str:
        """
        Synthesize final answer considering confidence weights.
        
        This demonstrates WEIGHTED SYNTHESIS - not all opinions are equal,
        agents with higher confidence get more influence.
        """
        synthesis_prompt = f"""Synthesize a final answer to this question:

Question: {question}

Specialist analyses (with confidence weights):
"""
        
        for agent_type, response in analyses.items():
            weight = weights.get(agent_type, 0.5)
            synthesis_prompt += f"\n{agent_type.upper()} (weight: {weight:.2f}, confidence: {response.confidence.value}):\n"
            synthesis_prompt += f"{response.content}\n"
            if response.uncertainties:
                synthesis_prompt += f"Uncertainties: {', '.join(response.uncertainties[:2])}\n"
        
        if debate_results:
            synthesis_prompt += f"\nKey debates:\n"
            for debate, result in list(debate_results.items())[:2]:
                synthesis_prompt += f"- {debate}: {result[:150]}...\n"
        
        synthesis_prompt += """
Provide a comprehensive answer that:
1. Integrates insights from all specialists
2. Weights contributions by confidence levels
3. Acknowledges areas of uncertainty or debate
4. Highlights the most interesting cross-specialist insights

Be specific, analytical, and concise (4-5 sentences).
"""
        
        if config.USE_OPENAI:
            return self.client.chat_interactive(synthesis_prompt)
        else:
            return self.client.generate(synthesis_prompt, max_tokens=512)
    
    def compare_songs(self, song1_data: Dict, song2_data: Dict) -> str:
        """Enhanced comparison with agent collaboration."""
        print(f"\n[ORCHESTRATOR] Assembling comparison team...")
        
        # Activate all agents for comparison
        self._activate_agents(['lyrical', 'musical', 'contextual'])
        
        comparison_analyses = {}
        
        for agent_type, agent in self.active_agents.items():
            prompt = f"""Compare these two songs from your expertise area ({agent.expertise}):

Song 1: {song1_data['Song_Name']} ({song1_data['Album']})
Song 2: {song2_data['Song_Name']} ({song2_data['Album']})

Provide:
1. Key similarities in your domain
2. Key differences in your domain
3. What this comparison reveals

Be specific (3-4 sentences).
"""
            
            if config.USE_OPENAI:
                response = self.client.chat_interactive(prompt)
            else:
                response = self.client.generate(prompt, max_tokens=400)
            
            comparison_analyses[agent_type] = response
        
        # Synthesize comparison
        synthesis_prompt = f"""Synthesize these specialist comparisons into a cohesive answer:

"""
        for agent_type, analysis in comparison_analyses.items():
            synthesis_prompt += f"\n{agent_type.upper()}:\n{analysis}\n"
        
        synthesis_prompt += "\nProvide integrated comparison highlighting cross-domain insights (4-5 sentences)."
        
        if config.USE_OPENAI:
            return self.client.chat_interactive(synthesis_prompt)
        else:
            return self.client.generate(synthesis_prompt, max_tokens=512)


def interactive_autonomous_multi_agent():
    """Interactive session with autonomous multi-agent system."""
    from src.data_loading import load_and_merge_data
    
    print("="*80)
    print("AUTONOMOUS MULTI-AGENT SYSTEM")
    print("="*80)
    print(f"Models: {config.MODEL} (specialists), {config.REASONING_MODEL} (orchestrator)")
    print("\nFeatures:")
    print("  ✓ Dynamic agent assembly")
    print("  ✓ Inter-agent debate")
    print("  ✓ Iterative refinement")
    print("  ✓ Confidence-based synthesis")
    print()
    
    print("Loading data...")
    df = load_and_merge_data(config.DATA_DIR, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    
    print("Initializing orchestrator...")
    orchestrator = AutonomousOrchestrator()
    
    print("\n✓ System ready!")
    print("\nCommands:")
    print("  analyze <song> [question]  - Deep analysis with autonomous agents")
    print("  compare <song1> vs <song2> - Comparative analysis")
    print("  quit                       - Exit")
    print("\nExample:")
    print("  analyze Blank Space")
    print("  analyze All Too Well what makes this song emotionally powerful?")
    print()
    
    while True:
        try:
            command = input("\nCommand: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if command.lower().startswith('analyze '):
                remainder = command[8:].strip()
                import re as _re
                q_match = _re.search(r'\s+(what|how|why|when|where|who|which|is|does)\b', remainder, _re.IGNORECASE)
                if q_match:
                    song_name = remainder[:q_match.start()].strip()
                    question = remainder[q_match.start():].strip()
                else:
                    song_name = remainder
                    question = f"Provide a comprehensive analysis of {song_name}"
                
                matches = df[df['Song_Name'].str.lower() == song_name.lower()]
                
                if matches.empty:
                    print(f"✗ Song '{song_name}' not found.")
                    continue
                
                song_data = matches.iloc[0].to_dict()
                result = orchestrator.analyze_song(question, song_data)
                
                print("\n" + "="*80)
                print("MULTI-AGENT ANALYSIS RESULTS")
                print("="*80)
                
                print("\n[INITIAL ANALYSES]")
                for agent_type, analysis in result['initial_analyses'].items():
                    print(f"\n{agent_type.upper()} ({analysis.confidence.value} confidence):")
                    print(analysis.content)
                    if analysis.uncertainties:
                        print(f"  Uncertainties: {analysis.uncertainties[0]}")
                
                if result['debate_results']:
                    print("\n[DEBATE OUTCOMES]")
                    for debate, outcome in result['debate_results'].items():
                        print(f"\n{debate}:")
                        print(outcome[:300] + "..." if len(outcome) > 300 else outcome)
                
                print("\n[FINAL SYNTHESIS]")
                print(result['synthesis'])
                
                print(f"\n[CONFIDENCE WEIGHTS]")
                for agent, weight in result['confidence_weights'].items():
                    print(f"  {agent}: {weight:.2f}")
            
            elif ' vs ' in command.lower() and command.lower().startswith('compare '):
                parts = command[8:].split(' vs ')
                if len(parts) != 2:
                    print("Usage: compare <song1> vs <song2>")
                    continue
                
                song1_name, song2_name = [s.strip() for s in parts]
                
                matches1 = df[df['Song_Name'].str.lower() == song1_name.lower()]
                matches2 = df[df['Song_Name'].str.lower() == song2_name.lower()]
                
                if matches1.empty or matches2.empty:
                    print("✗ One or both songs not found.")
                    continue
                
                comparison = orchestrator.compare_songs(
                    matches1.iloc[0].to_dict(),
                    matches2.iloc[0].to_dict()
                )
                
                print("\n" + "="*80)
                print("COMPARATIVE ANALYSIS")
                print("="*80)
                print(comparison)
            
            else:
                print("Unknown command. Try:")
                print("  analyze Blank Space")
                print("  analyze Style what makes this song distinctive?")
                print("  compare Style vs Wildest Dreams")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_autonomous_multi_agent()