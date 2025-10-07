"""
MLPY LLM - Prueba Interactiva
==============================

Script para probar el módulo LLM de forma interactiva.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlpy.llm import (
    BaseLLM, MLAssistant, RAGPipeline,
    PromptTemplate, LLMChain, ConversationChain,
    EmbeddingManager, VectorStore, Document
)
import numpy as np

print("\n" + "="*60)
print("MLPY LLM - PRUEBA INTERACTIVA")
print("="*60)

def test_basic_llm():
    """Prueba básica con diferentes providers."""
    print("\n1. PRUEBA BÁSICA DE LLM")
    print("-" * 40)
    
    providers = {
        "1": ("ollama", "llama2", "Ollama (Local)"),
        "2": ("openai", "gpt-3.5-turbo", "OpenAI"),
        "3": ("gemini", "gemini-pro", "Google Gemini"),
        "4": ("anthropic", "claude-3-opus-20240229", "Anthropic Claude"),
    }
    
    print("\nSelecciona un provider:")
    for key, (_, _, name) in providers.items():
        print(f"  {key}. {name}")
    print("  5. Saltar esta prueba")
    
    choice = input("\nOpción: ").strip()
    
    if choice == "5":
        return
    
    if choice in providers:
        provider, model, name = providers[choice]
        
        try:
            print(f"\nIntentando conectar con {name}...")
            
            # Crear LLM
            llm = BaseLLM(provider, model=model)
            
            # Prueba simple
            prompt = "Explica en una línea qué es machine learning"
            print(f"\nPrompt: {prompt}")
            print("Generando respuesta...")
            
            response = llm.complete(prompt)
            print(f"\nRespuesta: {response}")
            
            # Mostrar estadísticas
            stats = llm.get_usage_stats()
            print(f"\nEstadísticas: {stats}")
            
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            print("\nSugerencias:")
            if provider == "ollama":
                print("  - Asegúrate de que Ollama está instalado y ejecutándose")
                print("  - Instala un modelo: ollama pull llama2")
            elif provider == "openai":
                print("  - Configura tu API key: set OPENAI_API_KEY=tu-clave")
            elif provider == "gemini":
                print("  - Configura tu API key: set GEMINI_API_KEY=tu-clave")
                print("  - Instala: pip install google-generativeai")
            elif provider == "anthropic":
                print("  - Configura tu API key: set ANTHROPIC_API_KEY=tu-clave")


def test_ml_assistant():
    """Prueba el ML Assistant."""
    print("\n2. PRUEBA DE ML ASSISTANT")
    print("-" * 40)
    
    try:
        # Crear assistant (usará mock si no hay LLM disponible)
        assistant = MLAssistant()
        
        # Simular predicciones
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        targets = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 1])
        
        print("\nAnalizando rendimiento del modelo...")
        analysis = assistant.analyze_model_performance(
            predictions=predictions,
            targets=targets,
            model_type="RandomForest",
            task_type="classification"
        )
        print(f"\nAnálisis: {analysis[:200]}...")
        
        print("\n" + "-"*40)
        
        # Sugerir features
        print("\nSugiriendo feature engineering...")
        suggestions = assistant.suggest_features(
            feature_names=["edad", "ingresos", "educacion"],
            task_type="classification",
            domain="credit risk"
        )
        print(f"\nSugerencias: {suggestions[:200]}...")
        
        print("\n" + "-"*40)
        
        # Generar código
        print("\nGenerando código ML...")
        code = assistant.generate_ml_code(
            task="classification",
            algorithm="random_forest",
            requirements="Include cross-validation"
        )
        print(f"\nCódigo generado:\n{code[:300]}...")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")


def test_rag_simple():
    """Prueba RAG con embeddings locales."""
    print("\n3. PRUEBA DE RAG (BÚSQUEDA SEMÁNTICA)")
    print("-" * 40)
    
    try:
        from mlpy.llm.embeddings import HuggingFaceEmbeddings
        
        print("\nCreando sistema RAG...")
        
        # Usar embeddings locales (no requiere API key)
        embedder = HuggingFaceEmbeddings()
        
        # Crear vector store
        vector_store = VectorStore(
            embedding_dim=384,  # Dimensión de MiniLM
            similarity_metric="cosine"
        )
        
        # Documentos de ejemplo sobre ML
        documents_text = [
            "Random Forest es un algoritmo de ensemble que combina múltiples árboles de decisión para mejorar la precisión.",
            "Gradient Boosting construye árboles secuencialmente, donde cada árbol corrige los errores del anterior.",
            "Las redes neuronales profundas tienen múltiples capas ocultas y pueden aprender representaciones complejas.",
            "Support Vector Machines busca el hiperplano óptimo que maximiza el margen entre clases.",
            "K-Means es un algoritmo de clustering que agrupa datos en K clusters basándose en distancia.",
            "La validación cruzada divide los datos en folds para evaluar el modelo de forma robusta.",
            "El overfitting ocurre cuando el modelo memoriza los datos de entrenamiento y no generaliza bien.",
            "La regularización L1 (Lasso) puede hacer selección de features al llevar coeficientes a cero.",
            "El learning rate controla qué tan rápido aprende un modelo de gradient descent.",
            "El feature engineering es crucial para mejorar el rendimiento del modelo."
        ]
        
        print(f"\nIndexando {len(documents_text)} documentos...")
        
        # Crear y añadir documentos
        for i, text in enumerate(documents_text):
            doc = Document(
                content=text,
                metadata={"id": i, "topic": "ML"},
                embedding=embedder.embed(text)
            )
            vector_store.add_documents([doc])
        
        print("Documentos indexados correctamente!")
        
        # Búsqueda interactiva
        while True:
            print("\n" + "-"*40)
            query = input("\nHaz una pregunta sobre ML (o 'salir'): ").strip()
            
            if query.lower() in ['salir', 'exit', 'quit', '']:
                break
            
            # Buscar documentos relevantes
            query_embedding = embedder.embed(query)
            results = vector_store.search(query_embedding, k=3)
            
            print(f"\nTop 3 documentos relevantes para: '{query}'")
            print("-" * 40)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. (Score: {result.score:.3f})")
                print(f"   {result.document.content}")
        
    except ImportError:
        print("\n[WARNING] sentence-transformers no instalado")
        print("Instala con: pip install sentence-transformers")
        print("\nUsando embeddings simulados...")
        
        # Versión simplificada con embeddings aleatorios
        vector_store = VectorStore(embedding_dim=100)
        
        docs_simple = [
            "Machine Learning es aprendizaje automático",
            "Deep Learning usa redes neuronales",
            "Random Forest es un ensemble"
        ]
        
        for i, text in enumerate(docs_simple):
            doc = Document(
                content=text,
                metadata={"id": i},
                embedding=np.random.randn(100).tolist()
            )
            vector_store.add_documents([doc])
        
        print(f"\nIndexados {len(docs_simple)} documentos con embeddings simulados")
        print("(Nota: Los resultados no serán semánticamente precisos)")


def test_prompt_templates():
    """Prueba templates de prompts."""
    print("\n4. PRUEBA DE PROMPT TEMPLATES")
    print("-" * 40)
    
    from mlpy.llm import (
        PromptTemplate, FewShotTemplate, 
        ChainOfThoughtTemplate, PromptLibrary
    )
    
    # Template básico
    print("\nTemplate Básico:")
    template = PromptTemplate(
        "Explica {concepto} para alguien con nivel {nivel}",
        input_variables=["concepto", "nivel"]
    )
    
    prompt = template.format(
        concepto="redes neuronales",
        nivel="principiante"
    )
    print(f"Prompt generado: {prompt}")
    
    # Few-shot
    print("\n" + "-"*40)
    print("Template Few-Shot:")
    
    few_shot = FewShotTemplate(
        prefix="Clasifica el sentimiento de estos textos:",
        examples=[
            {"texto": "¡Excelente producto!", "sentimiento": "positivo"},
            {"texto": "No lo recomiendo", "sentimiento": "negativo"},
            {"texto": "Es aceptable", "sentimiento": "neutro"}
        ],
        suffix="Texto: {input_text}\nSentimiento:",
        example_template="Texto: {texto}\nSentimiento: {sentimiento}",
        input_variables=["input_text"]
    )
    
    prompt = few_shot.format(input_text="Me encanta este framework")
    print(f"Prompt generado:\n{prompt}")
    
    # Chain of Thought
    print("\n" + "-"*40)
    print("Template Chain-of-Thought:")
    
    cot = ChainOfThoughtTemplate(
        task_description="Determina si un modelo tiene overfitting",
        reasoning_steps=[
            "Compara accuracy en train vs test",
            "Revisa la complejidad del modelo",
            "Analiza las curvas de aprendizaje",
            "Evalúa la varianza del modelo"
        ],
        output_format="Diagnóstico: [Sí/No]\nRazón: [Explicación]"
    )
    
    prompt = cot.format(input="Train acc: 0.99, Test acc: 0.65")
    print(f"Prompt generado:\n{prompt[:300]}...")
    
    # Prompt Library
    print("\n" + "-"*40)
    print("Prompt Library (ML específico):")
    
    ml_prompt = PromptLibrary.get_data_analysis_prompt()
    prompt = ml_prompt.format(
        shape="(1000, 20)",
        dtypes="19 numeric, 1 categorical",
        missing="Column 'age': 5%, Column 'income': 2%",
        statistics="Mean age: 35, Std income: 25000"
    )
    print(f"Prompt para análisis:\n{prompt[:300]}...")


def test_conversation():
    """Prueba conversación con memoria."""
    print("\n5. PRUEBA DE CONVERSACIÓN")
    print("-" * 40)
    
    print("\n[INFO] Esta prueba requiere un LLM configurado")
    print("Simulando conversación con respuestas mock...")
    
    # Crear una conversación simulada
    print("\nConversación simulada sobre ML:")
    print("-" * 40)
    
    exchanges = [
        ("¿Qué es overfitting?", 
         "Overfitting es cuando un modelo aprende demasiado bien los datos de entrenamiento, "
         "incluyendo el ruido, y no generaliza bien a datos nuevos."),
        
        ("¿Cómo se puede prevenir?",
         "Se puede prevenir con regularización (L1/L2), dropout en redes neuronales, "
         "validación cruzada, más datos de entrenamiento o modelos más simples."),
        
        ("¿Qué es mejor, L1 o L2?",
         "L1 (Lasso) es mejor para selección de features ya que lleva coeficientes a cero. "
         "L2 (Ridge) es mejor para manejar multicolinealidad. ElasticNet combina ambos.")
    ]
    
    for i, (pregunta, respuesta) in enumerate(exchanges, 1):
        print(f"\n[Usuario]: {pregunta}")
        print(f"[Assistant]: {respuesta}")
    
    print("\n[INFO] Con un LLM real, la conversación mantendría contexto entre mensajes")


def main_menu():
    """Menú principal."""
    while True:
        print("\n" + "="*60)
        print("MENÚ PRINCIPAL - PRUEBAS LLM")
        print("="*60)
        
        print("\n1. Prueba básica de LLM")
        print("2. ML Assistant")
        print("3. RAG y búsqueda semántica")
        print("4. Prompt Templates")
        print("5. Conversación con memoria")
        print("6. Ejecutar todas las pruebas")
        print("0. Salir")
        
        choice = input("\nSelecciona una opción: ").strip()
        
        if choice == "0":
            print("\n¡Hasta luego!")
            break
        elif choice == "1":
            test_basic_llm()
        elif choice == "2":
            test_ml_assistant()
        elif choice == "3":
            test_rag_simple()
        elif choice == "4":
            test_prompt_templates()
        elif choice == "5":
            test_conversation()
        elif choice == "6":
            test_basic_llm()
            test_ml_assistant()
            test_rag_simple()
            test_prompt_templates()
            test_conversation()
        else:
            print("\nOpción no válida")
        
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    print("\n¡Bienvenido a las pruebas interactivas del módulo LLM de MLPY!")
    print("\nNOTA: Algunas pruebas requieren:")
    print("  - API keys configuradas (OpenAI, Gemini, etc.)")
    print("  - Ollama instalado y ejecutándose")
    print("  - sentence-transformers para embeddings locales")
    
    print("\nLas pruebas funcionarán con mocks si no hay providers disponibles.")
    
    main_menu()