"""
Prompt Engineering Framework
============================

Templates and tools for effective prompt engineering.
"""

from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import re
import json
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Basic prompt template with variable substitution."""
    
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        validate_inputs: bool = True
    ):
        """
        Initialize prompt template.
        
        Args:
            template: Template string with {variables}
            input_variables: Expected input variables
            validate_inputs: Whether to validate inputs
        """
        self.template = template
        self.validate_inputs = validate_inputs
        
        # Extract variables from template
        self.input_variables = input_variables or self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt
        """
        if self.validate_inputs:
            # Check all required variables are provided
            missing = set(self.input_variables) - set(kwargs.keys())
            if missing:
                raise ValueError(f"Missing required variables: {missing}")
            
            # Warn about extra variables
            extra = set(kwargs.keys()) - set(self.input_variables)
            if extra:
                logger.warning(f"Extra variables provided: {extra}")
        
        return self.template.format(**kwargs)
    
    def partial(self, **kwargs) -> 'PromptTemplate':
        """
        Create a partial template with some variables filled.
        
        Args:
            **kwargs: Variables to fill
            
        Returns:
            New PromptTemplate with partial variables filled
        """
        new_template = self.template.format_map(
            {k: v for k, v in kwargs.items() if k in self.input_variables}
        )
        
        remaining_vars = [v for v in self.input_variables if v not in kwargs]
        
        return PromptTemplate(new_template, remaining_vars, self.validate_inputs)
    
    def save(self, filepath: str):
        """Save template to file."""
        data = {
            'template': self.template,
            'input_variables': self.input_variables,
            'validate_inputs': self.validate_inputs
        }
        
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(data, f)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'PromptTemplate':
        """Load template from file."""
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        return cls(
            template=data['template'],
            input_variables=data.get('input_variables'),
            validate_inputs=data.get('validate_inputs', True)
        )


class FewShotTemplate:
    """Few-shot learning prompt template."""
    
    def __init__(
        self,
        prefix: str,
        examples: List[Dict[str, str]],
        suffix: str,
        example_template: str,
        input_variables: Optional[List[str]] = None
    ):
        """
        Initialize few-shot template.
        
        Args:
            prefix: Introduction text
            examples: List of example input-output pairs
            suffix: Text after examples, before input
            example_template: Template for formatting each example
            input_variables: Variables for the final input
        """
        self.prefix = prefix
        self.examples = examples
        self.suffix = suffix
        self.example_template = example_template
        self.input_variables = input_variables or []
    
    def format(self, **kwargs) -> str:
        """Format the few-shot prompt."""
        # Format examples
        formatted_examples = []
        for example in self.examples:
            formatted = self.example_template.format(**example)
            formatted_examples.append(formatted)
        
        # Build complete prompt
        prompt_parts = [
            self.prefix,
            "\n\n".join(formatted_examples),
            self.suffix.format(**kwargs)
        ]
        
        return "\n\n".join(prompt_parts)
    
    def add_example(self, example: Dict[str, str]):
        """Add a new example."""
        self.examples.append(example)
    
    def remove_example(self, index: int):
        """Remove an example by index."""
        if 0 <= index < len(self.examples):
            self.examples.pop(index)


class ChatTemplate:
    """Template for chat-based interactions."""
    
    def __init__(
        self,
        system_message: Optional[str] = None,
        human_template: Optional[str] = None,
        ai_template: Optional[str] = None
    ):
        """
        Initialize chat template.
        
        Args:
            system_message: System prompt
            human_template: Template for human messages
            ai_template: Template for AI messages
        """
        self.system_message = system_message
        self.human_template = human_template or "{input}"
        self.ai_template = ai_template or "{output}"
        self.messages: List[Dict[str, str]] = []
        
        if system_message:
            self.messages.append({"role": "system", "content": system_message})
    
    def add_user_message(self, content: str):
        """Add user message."""
        formatted = self.human_template.format(input=content)
        self.messages.append({"role": "user", "content": formatted})
    
    def add_assistant_message(self, content: str):
        """Add assistant message."""
        formatted = self.ai_template.format(output=content)
        self.messages.append({"role": "assistant", "content": formatted})
    
    def add_example(self, user_input: str, assistant_output: str):
        """Add an example exchange."""
        self.add_user_message(user_input)
        self.add_assistant_message(assistant_output)
    
    def format_for_completion(self, user_input: str) -> List[Dict[str, str]]:
        """Format messages for chat completion."""
        messages = self.messages.copy()
        messages.append({
            "role": "user",
            "content": self.human_template.format(input=user_input)
        })
        return messages
    
    def clear_history(self, keep_system: bool = True):
        """Clear message history."""
        if keep_system and self.system_message:
            self.messages = [{"role": "system", "content": self.system_message}]
        else:
            self.messages = []


class ChainOfThoughtTemplate(PromptTemplate):
    """Template for chain-of-thought reasoning."""
    
    def __init__(
        self,
        task_description: str,
        reasoning_steps: Optional[List[str]] = None,
        output_format: Optional[str] = None
    ):
        """
        Initialize chain-of-thought template.
        
        Args:
            task_description: Description of the task
            reasoning_steps: Optional list of reasoning steps
            output_format: Expected output format
        """
        # Build template
        template_parts = [task_description]
        
        if reasoning_steps:
            template_parts.append("Let's think step by step:")
            for i, step in enumerate(reasoning_steps, 1):
                template_parts.append(f"{i}. {step}")
        else:
            template_parts.append("Let's think step by step.")
        
        template_parts.append("\nInput: {input}")
        
        if output_format:
            template_parts.append(f"\nProvide your answer in the following format:\n{output_format}")
        
        template_parts.append("\nReasoning:")
        
        super().__init__("\n".join(template_parts), ["input"])


class PromptLibrary:
    """Library of pre-built prompts for ML tasks."""
    
    @staticmethod
    def get_ml_explanation_prompt() -> PromptTemplate:
        """Prompt for explaining ML predictions."""
        return PromptTemplate("""You are an expert data scientist explaining a machine learning model's prediction.

Model: {model_type}
Features used: {features}
Prediction: {prediction}
Confidence: {confidence}

Feature importance (top 5):
{feature_importance}

Explain this prediction in simple terms that a non-technical stakeholder would understand. 
Focus on:
1. What the prediction means
2. Which factors were most important
3. How confident we should be in this prediction
4. Any caveats or limitations

Explanation:""")
    
    @staticmethod
    def get_data_analysis_prompt() -> PromptTemplate:
        """Prompt for data analysis insights."""
        return PromptTemplate("""Analyze the following dataset statistics and provide insights:

Dataset shape: {shape}
Column types: {dtypes}
Missing values: {missing}
Basic statistics:
{statistics}

Provide:
1. Key observations about the data
2. Potential data quality issues
3. Suggested preprocessing steps
4. Recommended features for modeling

Analysis:""")
    
    @staticmethod
    def get_code_generation_prompt() -> PromptTemplate:
        """Prompt for generating ML code."""
        return PromptTemplate("""Generate Python code for the following machine learning task:

Task: {task_description}
Dataset: {dataset_info}
Requirements: {requirements}

Use the MLPY framework where applicable. Include:
- Data loading and preprocessing
- Model training
- Evaluation
- Best practices and error handling

Code:
```python""")
    
    @staticmethod
    def get_error_diagnosis_prompt() -> PromptTemplate:
        """Prompt for diagnosing ML errors."""
        return PromptTemplate("""Help diagnose and fix this machine learning error:

Error message:
{error_message}

Code context:
```python
{code_context}
```

Model type: {model_type}
Dataset shape: {data_shape}

Provide:
1. Root cause of the error
2. Step-by-step solution
3. Code fix
4. How to prevent this in the future

Diagnosis:""")
    
    @staticmethod
    def get_feature_engineering_prompt() -> PromptTemplate:
        """Prompt for feature engineering suggestions."""
        return PromptTemplate("""Suggest feature engineering for this dataset:

Target variable: {target}
Current features: {features}
Data sample:
{sample_data}

Task type: {task_type}
Domain: {domain}

Suggest:
1. New features to create
2. Feature transformations
3. Interaction terms
4. Domain-specific features
5. Python code to implement these features

Suggestions:""")


class PromptOptimizer:
    """Optimize prompts for better performance."""
    
    def __init__(self, llm_provider: Any):
        """
        Initialize prompt optimizer.
        
        Args:
            llm_provider: LLM provider for testing prompts
        """
        self.llm = llm_provider
        self.test_results: List[Dict[str, Any]] = []
    
    def test_prompt_variations(
        self,
        base_prompt: str,
        variations: List[str],
        test_inputs: List[Dict[str, Any]],
        scorer: Callable[[str], float]
    ) -> Dict[str, float]:
        """
        Test prompt variations and score them.
        
        Args:
            base_prompt: Base prompt template
            variations: List of prompt variations
            test_inputs: Test input data
            scorer: Function to score outputs
            
        Returns:
            Dictionary of prompt -> average score
        """
        results = {}
        
        for i, prompt in enumerate([base_prompt] + variations):
            scores = []
            
            for test_input in test_inputs:
                # Format and run prompt
                formatted = prompt.format(**test_input)
                response = self.llm.complete(formatted)
                
                # Score response
                score = scorer(response)
                scores.append(score)
                
                self.test_results.append({
                    'prompt_index': i,
                    'prompt': prompt[:100] + "...",
                    'input': test_input,
                    'response': response,
                    'score': score
                })
            
            avg_score = sum(scores) / len(scores)
            results[f"prompt_{i}"] = avg_score
            
            logger.info(f"Prompt {i} average score: {avg_score:.3f}")
        
        return results
    
    def auto_improve_prompt(
        self,
        initial_prompt: str,
        examples: List[Dict[str, str]],
        num_iterations: int = 3
    ) -> str:
        """
        Automatically improve a prompt using LLM.
        
        Args:
            initial_prompt: Starting prompt
            examples: Example inputs and desired outputs
            num_iterations: Number of improvement iterations
            
        Returns:
            Improved prompt
        """
        current_prompt = initial_prompt
        
        for i in range(num_iterations):
            improvement_prompt = f"""Improve this prompt to get better results:

Current prompt:
{current_prompt}

Examples of desired input/output:
{json.dumps(examples, indent=2)}

Provide an improved version that will produce more accurate and consistent results.
Focus on clarity, specificity, and including any necessary context.

Improved prompt:"""
            
            improved = self.llm.complete(improvement_prompt)
            current_prompt = improved
            
            logger.info(f"Iteration {i+1}: Prompt improved")
        
        return current_prompt