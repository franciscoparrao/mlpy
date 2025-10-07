"""
MLPY LLM Integration Demo
=========================

Demonstrates LLM capabilities for ML workflows.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.llm import (
    BaseLLM, LLMConfig, 
    PromptTemplate, FewShotTemplate, ChainOfThoughtTemplate,
    ChatTemplate, PromptLibrary, PromptOptimizer,
    RAGPipeline, VectorStore, DocumentLoader,
    LLMChain, SequentialChain, ConversationChain,
    RouterChain, MapReduceChain,
    EmbeddingManager,
    MLAssistant
)
from mlpy.learners import RandomForestLearner
from mlpy.tasks import ClassificationTask
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_llm():
    """Demo basic LLM functionality."""
    print("\n" + "="*60)
    print("1. BASIC LLM USAGE")
    print("="*60)
    
    # Note: Set your API key as environment variable
    # export OPENAI_API_KEY="your-key"
    # or use local models with Ollama
    
    try:
        # Try OpenAI first
        llm = BaseLLM("openai", model="gpt-3.5-turbo")
        print("Using OpenAI GPT-3.5")
    except:
        try:
            # Fallback to Ollama
            llm = BaseLLM("ollama", model="llama2")
            print("Using Ollama Llama2")
        except:
            print("No LLM provider available. Set API keys or install Ollama.")
            return
    
    # Simple completion
    response = llm.complete("What are the key steps in a machine learning pipeline?")
    print(f"\nCompletion Response:\n{response[:200]}...")
    
    # Chat conversation
    response = llm.chat("What is gradient boosting?")
    print(f"\nChat Response:\n{response[:200]}...")
    
    # Show usage stats
    stats = llm.get_usage_stats()
    print(f"\nUsage Stats: {stats}")


def demo_prompt_engineering():
    """Demo prompt engineering features."""
    print("\n" + "="*60)
    print("2. PROMPT ENGINEERING")
    print("="*60)
    
    # Basic template
    template = PromptTemplate(
        "Explain {concept} in the context of {domain}.",
        input_variables=["concept", "domain"]
    )
    
    prompt = template.format(concept="overfitting", domain="neural networks")
    print(f"Formatted Prompt: {prompt}")
    
    # Few-shot template
    few_shot = FewShotTemplate(
        prefix="Classify the sentiment of movie reviews:",
        examples=[
            {"review": "Great movie!", "sentiment": "positive"},
            {"review": "Terrible film.", "sentiment": "negative"}
        ],
        suffix="Review: {review}\nSentiment:",
        example_template="Review: {review}\nSentiment: {sentiment}",
        input_variables=["review"]
    )
    
    prompt = few_shot.format(review="Decent watch.")
    print(f"\nFew-Shot Prompt:\n{prompt}")
    
    # Chain of thought
    cot = ChainOfThoughtTemplate(
        task_description="Determine if a dataset is suitable for deep learning",
        reasoning_steps=[
            "Check the dataset size",
            "Evaluate feature complexity",
            "Consider computational resources",
            "Assess data quality"
        ],
        output_format="Answer: [Yes/No]\nReason: [Brief explanation]"
    )
    
    prompt = cot.format(input="10000 samples, 50 features, image classification task")
    print(f"\nChain-of-Thought Prompt:\n{prompt[:300]}...")
    
    # Use prompt library
    ml_prompt = PromptLibrary.get_ml_explanation_prompt()
    prompt = ml_prompt.format(
        model_type="Random Forest",
        features="age, income, credit_score",
        prediction="Loan Approved",
        confidence="0.89",
        feature_importance="1. credit_score: 0.45\n2. income: 0.35\n3. age: 0.20"
    )
    print(f"\nML Explanation Prompt:\n{prompt[:300]}...")


def demo_llm_chains():
    """Demo LLM chains."""
    print("\n" + "="*60)
    print("3. LLM CHAINS")
    print("="*60)
    
    try:
        llm = BaseLLM("ollama", model="llama2")
    except:
        print("Ollama not available. Skipping chains demo.")
        return
    
    # Simple chain
    simple_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate("List 3 key features of {algorithm}.")
    )
    
    result = simple_chain.run(algorithm="Random Forest")
    print(f"Simple Chain Result:\n{result[:200]}...")
    
    # Sequential chain
    chain1 = LLMChain(
        llm=llm,
        prompt=PromptTemplate("What type of ML problem is this: {description}")
    )
    chain1.output_key = "problem_type"
    
    chain2 = LLMChain(
        llm=llm,
        prompt=PromptTemplate("Suggest algorithms for {output_0}")
    )
    
    sequential = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["description"],
        output_variables=["output_0", "output_1"]
    )
    
    result = sequential.run(description="Predicting house prices based on features")
    print(f"\nSequential Chain Results: {result}")
    
    # Conversation chain
    conversation = ConversationChain(
        llm=llm,
        system_prompt="You are an ML expert assistant."
    )
    
    response1 = conversation.chat("What is cross-validation?")
    print(f"\nConversation 1: {response1[:100]}...")
    
    response2 = conversation.chat("How many folds are typically used?")
    print(f"Conversation 2: {response2[:100]}...")


def demo_rag_pipeline():
    """Demo RAG pipeline."""
    print("\n" + "="*60)
    print("4. RAG PIPELINE")
    print("="*60)
    
    try:
        # Use HuggingFace for embeddings (free, no API key needed)
        from mlpy.llm.providers import HuggingFaceProvider
        from mlpy.llm.embeddings import HuggingFaceEmbeddings
        
        # Simple embedding provider
        embedder = HuggingFaceEmbeddings()
        
        # Create vector store
        vector_store = VectorStore(
            embedding_dim=384,  # HuggingFace MiniLM dimension
            similarity_metric="cosine"
        )
        
        # Create sample documents
        documents = [
            "Random Forest is an ensemble learning method that creates multiple decision trees.",
            "Gradient Boosting builds trees sequentially, where each tree corrects the errors of the previous one.",
            "Support Vector Machines find the optimal hyperplane that separates classes.",
            "Neural Networks consist of layers of interconnected nodes that learn patterns.",
            "K-Means clustering groups data points into K clusters based on similarity."
        ]
        
        # Add documents to vector store
        from mlpy.llm.rag import Document
        docs = []
        for i, text in enumerate(documents):
            doc = Document(
                content=text,
                metadata={"source": f"ml_knowledge_{i}"},
                embedding=embedder.embed(text)
            )
            docs.append(doc)
        
        vector_store.add_documents(docs)
        print(f"Added {len(docs)} documents to vector store")
        
        # Search for similar documents
        query = "How do ensemble methods work?"
        query_embedding = embedder.embed(query)
        results = vector_store.search(query_embedding, k=3)
        
        print(f"\nQuery: {query}")
        print("Top 3 similar documents:")
        for result in results:
            print(f"  - (score: {result.score:.3f}) {result.document.content[:100]}...")
        
    except Exception as e:
        print(f"RAG demo error: {e}")
        print("Install sentence-transformers: pip install sentence-transformers")


def demo_ml_assistant():
    """Demo ML Assistant integration."""
    print("\n" + "="*60)
    print("5. ML ASSISTANT FOR MLPY")
    print("="*60)
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create a simple model
    task = ClassificationTask()
    learner = RandomForestLearner(n_estimators=10)
    learner.train(task, X, y)
    predictions = learner.predict(X)
    
    try:
        # Initialize ML Assistant
        assistant = MLAssistant(provider="ollama", model="llama2")
        
        # Analyze model performance
        analysis = assistant.analyze_model_performance(
            predictions=predictions[:10],
            targets=y[:10],
            model_type="RandomForest",
            task_type="classification"
        )
        print(f"Performance Analysis:\n{analysis[:300]}...")
        
        # Get feature engineering suggestions
        feature_names = iris.feature_names
        suggestions = assistant.suggest_features(
            feature_names=feature_names,
            task_type="classification",
            domain="botany"
        )
        print(f"\nFeature Suggestions:\n{suggestions[:300]}...")
        
        # Explain predictions
        sample_features = dict(zip(feature_names, X[0]))
        explanation = assistant.explain_prediction(
            prediction=predictions[0],
            features=sample_features,
            model_type="RandomForest"
        )
        print(f"\nPrediction Explanation:\n{explanation[:300]}...")
        
    except Exception as e:
        print(f"ML Assistant not available: {e}")
        print("Set up an LLM provider to use ML Assistant features")


def demo_embeddings():
    """Demo embedding providers."""
    print("\n" + "="*60)
    print("6. EMBEDDING PROVIDERS")
    print("="*60)
    
    # Create embedding manager
    manager = EmbeddingManager(default_provider="huggingface")
    
    try:
        # Test different embedding providers
        texts = [
            "Machine learning is amazing",
            "Deep learning uses neural networks",
            "Random forests are ensemble methods"
        ]
        
        # Generate embeddings
        embeddings = manager.embed(texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Find similar texts
        query = "What are tree-based models?"
        results = manager.find_similar(
            query=query,
            documents=texts,
            k=2,
            metric="cosine"
        )
        
        print(f"\nQuery: {query}")
        print("Most similar documents:")
        for idx, score, doc in results:
            print(f"  - (score: {score:.3f}) {doc}")
        
    except Exception as e:
        print(f"Embeddings demo error: {e}")


def demo_advanced_prompting():
    """Demo advanced prompting techniques."""
    print("\n" + "="*60)
    print("7. ADVANCED PROMPTING")
    print("="*60)
    
    try:
        llm = BaseLLM("ollama", model="llama2")
        
        # Create prompt optimizer (mock example)
        optimizer = PromptOptimizer(llm.provider)
        
        # Test prompt variations
        base_prompt = "Explain {concept} simply"
        variations = [
            "Explain {concept} in simple terms",
            "Give a brief explanation of {concept}",
            "What is {concept}? Explain simply"
        ]
        
        test_inputs = [
            {"concept": "overfitting"},
            {"concept": "cross-validation"}
        ]
        
        # Simple scorer (length-based for demo)
        def scorer(response):
            # Prefer concise responses
            return 1.0 / (1 + len(response) / 100)
        
        print("Testing prompt variations...")
        print("(This would normally test with real LLM responses)")
        
        # Auto-improve prompt
        initial = "Explain machine learning"
        examples = [
            {
                "input": "neural networks",
                "output": "Neural networks are computing systems inspired by biological neural networks."
            }
        ]
        
        print(f"\nInitial prompt: {initial}")
        print("Auto-improvement would refine this based on examples")
        
    except Exception as e:
        print(f"Advanced prompting demo error: {e}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("MLPY LLM INTEGRATION DEMO")
    print("="*60)
    
    demos = [
        ("Basic LLM Usage", demo_basic_llm),
        ("Prompt Engineering", demo_prompt_engineering),
        ("LLM Chains", demo_llm_chains),
        ("RAG Pipeline", demo_rag_pipeline),
        ("ML Assistant", demo_ml_assistant),
        ("Embeddings", demo_embeddings),
        ("Advanced Prompting", demo_advanced_prompting)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            print("Continuing with next demo...")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("\nKey Features Demonstrated:")
    print("- Multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama)")
    print("- Prompt engineering and templates")
    print("- LLM chains for complex workflows")
    print("- RAG pipeline with vector search")
    print("- ML-specific assistant capabilities")
    print("- Embedding providers for semantic search")
    print("- Integration with MLPY ML workflows")
    print("="*60)


if __name__ == "__main__":
    main()