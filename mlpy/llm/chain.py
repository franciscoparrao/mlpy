"""
LLM Chains and Conversation Management
======================================

Chains for complex LLM workflows and conversation management.
"""

from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime
from enum import Enum

from .base import BaseLLM, LLMResponse
from .prompts import PromptTemplate, ChatTemplate

logger = logging.getLogger(__name__)


class ChainType(Enum):
    """Types of LLM chains."""
    SIMPLE = "simple"
    SEQUENTIAL = "sequential"
    CONVERSATION = "conversation"
    ROUTER = "router"
    MAP_REDUCE = "map_reduce"


class LLMChain:
    """Basic LLM chain with prompt template."""
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt: Union[str, PromptTemplate],
        output_parser: Optional[Callable[[str], Any]] = None,
        verbose: bool = False
    ):
        """
        Initialize LLM chain.
        
        Args:
            llm: LLM instance
            prompt: Prompt template
            output_parser: Function to parse output
            verbose: Whether to log operations
        """
        self.llm = llm
        
        if isinstance(prompt, str):
            self.prompt = PromptTemplate(prompt)
        else:
            self.prompt = prompt
        
        self.output_parser = output_parser
        self.verbose = verbose
        self.run_count = 0
    
    def run(self, **kwargs) -> Union[str, Any]:
        """
        Run the chain with inputs.
        
        Args:
            **kwargs: Input variables for prompt
            
        Returns:
            Output from LLM (optionally parsed)
        """
        # Format prompt
        formatted_prompt = self.prompt.format(**kwargs)
        
        if self.verbose:
            logger.info(f"Running chain with prompt: {formatted_prompt[:100]}...")
        
        # Get LLM response
        response = self.llm.complete(formatted_prompt)
        
        self.run_count += 1
        
        # Parse output if parser provided
        if self.output_parser:
            try:
                return self.output_parser(response)
            except Exception as e:
                logger.error(f"Output parsing failed: {e}")
                return response
        
        return response
    
    def __call__(self, **kwargs):
        """Allow chain to be called directly."""
        return self.run(**kwargs)


class SequentialChain:
    """Chain that runs multiple chains sequentially."""
    
    def __init__(
        self,
        chains: List[LLMChain],
        input_variables: List[str],
        output_variables: List[str],
        verbose: bool = False
    ):
        """
        Initialize sequential chain.
        
        Args:
            chains: List of chains to run
            input_variables: Required input variables
            output_variables: Variables to return
            verbose: Whether to log operations
        """
        self.chains = chains
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.verbose = verbose
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run chains sequentially."""
        # Validate inputs
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing input variables: {missing}")
        
        # Working memory for intermediate results
        memory = dict(kwargs)
        
        # Run each chain
        for i, chain in enumerate(self.chains):
            if self.verbose:
                logger.info(f"Running chain {i+1}/{len(self.chains)}")
            
            # Get required inputs for this chain
            chain_inputs = {}
            for var in chain.prompt.input_variables:
                if var in memory:
                    chain_inputs[var] = memory[var]
            
            # Run chain and store output
            output = chain.run(**chain_inputs)
            
            # Store output with chain-specific key
            output_key = f"output_{i}"
            if hasattr(chain, 'output_key'):
                output_key = chain.output_key
            
            memory[output_key] = output
        
        # Return requested outputs
        results = {}
        for var in self.output_variables:
            if var in memory:
                results[var] = memory[var]
        
        return results


class ConversationChain:
    """Chain for managing conversations with memory."""
    
    def __init__(
        self,
        llm: BaseLLM,
        memory_size: int = 10,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize conversation chain.
        
        Args:
            llm: LLM instance
            memory_size: Number of exchanges to remember
            system_prompt: System prompt for conversation
            verbose: Whether to log operations
        """
        self.llm = llm
        self.memory_size = memory_size
        self.verbose = verbose
        
        self.chat_template = ChatTemplate(system_message=system_prompt)
        self.conversation_history: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
    
    def chat(self, user_input: str) -> str:
        """
        Send message and get response.
        
        Args:
            user_input: User message
            
        Returns:
            Assistant response
        """
        # Get response
        messages = self.chat_template.format_for_completion(user_input)
        response = self.llm.provider.chat(messages)
        
        # Update template and history
        self.chat_template.add_user_message(user_input)
        self.chat_template.add_assistant_message(response.text)
        
        self.conversation_history.append({
            "user": user_input,
            "assistant": response.text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Manage memory size
        if len(self.conversation_history) > self.memory_size:
            # Summarize old conversations
            self._summarize_old_conversations()
        
        if self.verbose:
            logger.info(f"Conversation turn {len(self.conversation_history)}")
        
        return response.text
    
    def _summarize_old_conversations(self):
        """Summarize old conversations to save memory."""
        if len(self.conversation_history) <= self.memory_size:
            return
        
        # Get conversations to summarize
        to_summarize = self.conversation_history[:-self.memory_size]
        
        # Build summary prompt
        summary_prompt = """Summarize the following conversation concisely:

"""
        for exchange in to_summarize:
            summary_prompt += f"User: {exchange['user']}\n"
            summary_prompt += f"Assistant: {exchange['assistant']}\n\n"
        
        summary_prompt += "\nSummary:"
        
        # Get summary
        self.summary = self.llm.complete(summary_prompt)
        
        # Remove summarized conversations
        self.conversation_history = self.conversation_history[-self.memory_size:]
        
        # Update system message with summary
        if self.summary:
            new_system = f"{self.chat_template.system_message}\n\nPrevious conversation summary: {self.summary}"
            self.chat_template.system_message = new_system
    
    def clear(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.chat_template.clear_history()
        self.summary = None
    
    def save(self, filepath: str):
        """Save conversation to file."""
        data = {
            "history": self.conversation_history,
            "summary": self.summary,
            "system_prompt": self.chat_template.system_message
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load conversation from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.conversation_history = data.get("history", [])
        self.summary = data.get("summary")
        
        # Rebuild chat template
        system_prompt = data.get("system_prompt")
        self.chat_template = ChatTemplate(system_message=system_prompt)
        
        # Rebuild message history
        for exchange in self.conversation_history:
            self.chat_template.add_user_message(exchange["user"])
            self.chat_template.add_assistant_message(exchange["assistant"])


class RouterChain:
    """Chain that routes to different chains based on input."""
    
    def __init__(
        self,
        llm: BaseLLM,
        routes: Dict[str, LLMChain],
        default_chain: Optional[LLMChain] = None,
        router_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize router chain.
        
        Args:
            llm: LLM for routing decision
            routes: Dictionary of route_name -> chain
            default_chain: Default chain if no route matches
            router_prompt: Custom prompt for routing
            verbose: Whether to log operations
        """
        self.llm = llm
        self.routes = routes
        self.default_chain = default_chain
        self.verbose = verbose
        
        # Default router prompt
        self.router_prompt = router_prompt or """Given the user input, determine which category it belongs to.

Categories:
{categories}

User input: {input}

Respond with ONLY the category name, nothing else.
Category:"""
    
    def run(self, user_input: str, **kwargs) -> Any:
        """Route and run appropriate chain."""
        # Determine route
        categories = "\n".join([f"- {name}" for name in self.routes.keys()])
        
        routing_prompt = self.router_prompt.format(
            categories=categories,
            input=user_input
        )
        
        route_decision = self.llm.complete(routing_prompt).strip().lower()
        
        if self.verbose:
            logger.info(f"Routing to: {route_decision}")
        
        # Find matching route
        selected_chain = None
        for route_name, chain in self.routes.items():
            if route_name.lower() in route_decision or route_decision in route_name.lower():
                selected_chain = chain
                break
        
        # Use default if no match
        if selected_chain is None:
            selected_chain = self.default_chain
            if selected_chain is None:
                raise ValueError(f"No route matched for: {route_decision}")
        
        # Run selected chain
        return selected_chain.run(input=user_input, **kwargs)


class MapReduceChain:
    """Chain for map-reduce operations over documents."""
    
    def __init__(
        self,
        llm: BaseLLM,
        map_prompt: Union[str, PromptTemplate],
        reduce_prompt: Union[str, PromptTemplate],
        verbose: bool = False
    ):
        """
        Initialize map-reduce chain.
        
        Args:
            llm: LLM instance
            map_prompt: Prompt for map step
            reduce_prompt: Prompt for reduce step
            verbose: Whether to log operations
        """
        self.llm = llm
        
        if isinstance(map_prompt, str):
            self.map_prompt = PromptTemplate(map_prompt)
        else:
            self.map_prompt = map_prompt
        
        if isinstance(reduce_prompt, str):
            self.reduce_prompt = PromptTemplate(reduce_prompt)
        else:
            self.reduce_prompt = reduce_prompt
        
        self.verbose = verbose
    
    def run(
        self,
        documents: List[str],
        question: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Run map-reduce over documents.
        
        Args:
            documents: List of document texts
            question: Optional question to answer
            **kwargs: Additional variables
            
        Returns:
            Reduced output
        """
        # Map step: Process each document
        mapped_results = []
        
        for i, doc in enumerate(documents):
            if self.verbose:
                logger.info(f"Mapping document {i+1}/{len(documents)}")
            
            map_inputs = {"document": doc}
            if question:
                map_inputs["question"] = question
            map_inputs.update(kwargs)
            
            result = self.llm.complete(self.map_prompt.format(**map_inputs))
            mapped_results.append(result)
        
        # Reduce step: Combine results
        if self.verbose:
            logger.info("Reducing mapped results")
        
        reduce_inputs = {
            "mapped_results": "\n\n".join(mapped_results),
            "num_documents": len(documents)
        }
        if question:
            reduce_inputs["question"] = question
        reduce_inputs.update(kwargs)
        
        final_result = self.llm.complete(self.reduce_prompt.format(**reduce_inputs))
        
        return final_result


class ChainBuilder:
    """Builder for creating complex chains."""
    
    @staticmethod
    def create_qa_chain(llm: BaseLLM) -> LLMChain:
        """Create question-answering chain."""
        prompt = PromptTemplate("""Answer the question based on your knowledge.

Question: {question}

Provide a clear, accurate, and helpful answer.

Answer:""")
        
        return LLMChain(llm, prompt)
    
    @staticmethod
    def create_summarization_chain(llm: BaseLLM) -> LLMChain:
        """Create summarization chain."""
        prompt = PromptTemplate("""Summarize the following text concisely:

Text:
{text}

Summary:""")
        
        return LLMChain(llm, prompt)
    
    @staticmethod
    def create_extraction_chain(
        llm: BaseLLM,
        schema: Dict[str, str]
    ) -> LLMChain:
        """Create information extraction chain."""
        schema_str = json.dumps(schema, indent=2)
        
        prompt = PromptTemplate(f"""Extract information from the text according to this schema:

Schema:
{schema_str}

Text:
{{text}}

Return the extracted information as valid JSON.

Extracted:""")
        
        def json_parser(text: str) -> Dict[str, Any]:
            try:
                # Find JSON in response
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except:
                pass
            return {}
        
        return LLMChain(llm, prompt, output_parser=json_parser)