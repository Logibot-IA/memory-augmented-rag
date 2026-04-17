import os
import uuid
import dotenv
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent

from langgraph.checkpoint.memory import MemorySaver

from transformers import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI

logging.set_verbosity_error()

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "benchmark-memory-augmented-rag")

# %%
model = init_chat_model(
    model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
    model_provider="openai",
    base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")
)


# %%
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = InMemoryVectorStore(embeddings)

loader = DirectoryLoader(
    path="../docs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

prompt = (
    "You have access to a tool that retrieves context from a book for logic programming. "
    "Use the tool to help answer user queries."
    "Responda sempre em português, mesmo que a pergunta seja feita em outro idioma."
)

checkpointer = MemorySaver()

agent = create_agent(
    model, 
    tools, 
    system_prompt=prompt,
    checkpointer=checkpointer
)

test_queries = [
    "O que são conectivos lógicos e quais são os cinco conectivos apresentados na Apostila de Lógica de Programação?",
    "Como a Apostila de Lógica de Programação define tabela-verdade e qual princípio a fundamenta?",
    "Como funciona a pesquisa binária e qual é a sua complexidade de tempo segundo o livro Entendendo Algoritmos?",
    "Qual é a estratégia de dividir para conquistar utilizada pelo quicksort conforme descrito em Entendendo Algoritmos?",
    "Quais são os principais domínios de programação e suas linguagens associadas segundo Sebesta em Conceitos de Linguagens de Programação?",
    "Como Szwarcfiter define complexidade de pior caso de um algoritmo no livro Estruturas de Dados e Seus Algoritmos?",
    "Como o livro Fundamentos da Programação de Computadores de Ascencio descreve a função do computador e a necessidade de algoritmos?",
    "Como Forbellone define algoritmo e o conceito de sequenciação no livro Lógica de Programação?",
    "Segundo Nilo Menezes no livro Introdução à Programação com Python, o que realmente significa saber programar?",
    "Como Manzano define o termo algoritmo no livro Algoritmos: Lógica para Desenvolvimento de Programação de Computadores e qual é a sua origem etimológica?"
]

ground_truths = [
    "Conectivos lógicos são palavras ou frases usadas para formar novas proposições a partir de outras proposições. Os cinco conectivos apresentados são: negação (~), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "A tabela-verdade é um dispositivo que apresenta todos os possíveis valores lógicos de uma proposição composta, correspondentes a todas as atribuições de valores às proposições simples. Ela é fundamentada no Princípio do Terceiro Excluído, segundo o qual toda proposição é verdadeira ou falsa, e o valor lógico de uma proposição composta depende unicamente dos valores lógicos das proposições simples componentes.",
    "A pesquisa binária funciona eliminando metade dos elementos a cada iteração, comparando o elemento central com o valor buscado. Para uma lista de n elementos, requer no máximo log₂n verificações, resultando em complexidade O(log n), muito mais eficiente que a pesquisa simples O(n) para listas grandes.",
    "A estratégia de dividir para conquistar consiste em identificar o caso-base mais simples e dividir o problema até chegar a ele. O quicksort escolhe um elemento pivô, separa o array em dois subarrays com elementos menores e maiores que o pivô, e ordena recursivamente cada subarray. No caso médio, possui complexidade O(n log n).",
    "Sebesta identifica cinco domínios principais: aplicações científicas (Fortran, ALGOL), aplicações comerciais (COBOL), inteligência artificial (LISP, Prolog), programação de sistemas (C, C++) e programação para a Web. Cada domínio possui características distintas que motivaram o desenvolvimento de linguagens específicas.",
    "A complexidade de pior caso é definida como o número máximo de passos que um algoritmo executa considerando todas as entradas possíveis: max Ei ∈ E {ti}. Ela fornece um limite superior para o tempo de execução em qualquer situação e é a medida mais utilizada por garantir que o algoritmo nunca ultrapassará esse limite independentemente da entrada.",
    "Segundo Ascencio, o computador é uma máquina que recebe, manipula e armazena dados, mas não possui iniciativa, independência ou criatividade, precisando de instruções detalhadas. Sua finalidade principal é o processamento de dados: receber dados de entrada, realizar operações e gerar uma resposta de saída, o que requer algoritmos bem definidos.",
    "Forbellone define algoritmo como um conjunto de regras formais para a obtenção de um resultado ou da solução de um problema. A sequenciação é uma convenção que rege o fluxo de execução do algoritmo, determinando a ordem das ações de forma linear, de cima para baixo, assim como se lê um texto, estabelecendo um padrão de comportamento seguido por qualquer pessoa.",
    "Segundo Menezes, saber programar não é decorar comandos, parâmetros e nomes estranhos. Programar é saber utilizar uma linguagem de programação para resolver problemas, ou seja, saber expressar uma solução por meio de uma linguagem de programação. A sintaxe pode ser esquecida, mas quem realmente sabe programar tem pouca dificuldade ao aprender uma nova linguagem.",
    "Segundo Manzano, o termo algoritmo deriva do nome do matemático Muhammad ibn Musā al-Khwārizmī. Do ponto de vista computacional, algoritmo é entendido como regras formais, sequenciais e bem definidas a partir do entendimento lógico de um problema, com o objetivo de transformá-lo em um programa executável por um computador, onde dados de entrada são transformados em dados de saída."
]

def run_agent_and_collect_data(query: str, ground_truth: str) -> Dict[str, Any]:
    thread_id = str(uuid.uuid4())
    
    events = list(agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ))
    
    final_event = events[-1]
    answer = final_event["messages"][-1].content

    retrieved_docs = vector_store.similarity_search(query, k=2)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    return {
        "question": query,
        "contexts": contexts,
        "answer": answer,
        "ground_truth": ground_truth
    }

def evaluate_with_ragas():
    """Executa avaliação completa com RAGAS."""
    print("Executando agent para coletar dados de teste...")
    ragas_data = []
    
    for i, query in enumerate(test_queries):
        print(f"Testando: {query}")
        data_point = run_agent_and_collect_data(query, ground_truths[i])
        ragas_data.append(data_point)
    
    test_dataset = Dataset.from_list(ragas_data)
    
    print("\nExecutando avaliação RAGAS...")

    eval_llm = ChatOpenAI(
        model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
        base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1"),
        temperature=0
    )
    
    result = evaluate(
        test_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=embeddings 
    )
    
    print("\n=== RESULTADOS RAGAS ===")
    print(result)
    
    df = result.to_pandas()
    print("\nDetalhes por query:")
    print(df)
    
    return result

if __name__ == "__main__":
    thread_id = "sael"
    query = (
        "O que é array em programação lógica?\n\n"
        "Depois que obtiver a resposta, pesquise extensões comuns desse método."
    )
    
    print("=== EXECUÇÃO NORMAL DO AGENT ===")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
    
    
    print("\n\n=== AVALIAÇÃO RAGAS ===")
    try:
        evaluate_with_ragas()
    except Exception as e:
        print(f"Erro na avaliação RAGAS: {e}")
        print("Verifique se tem OPENAI_API_KEY válido e ragas instalado.")