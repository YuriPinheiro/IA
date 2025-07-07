from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from memory_manager import MemoryManager
from llama_index.core import Document
import json
import os
from tqdm import tqdm

memory_manager = MemoryManager()

def load_data_sources():
    """Carrega dados de repos, knowledge base, memory/chats e corrections"""
    # Carrega projetos de código
    repos_path = "repos"
    project_dirs = [
        os.path.join(repos_path, nome)
        for nome in os.listdir(repos_path)
        if os.path.isdir(os.path.join(repos_path, nome))
    ]
    
    # Carrega knowledge base
    knowledge_path = "knowledge"
    
    # Combina todos os documentos
    documents = []
    
    # 1. Carrega projetos de código
    if project_dirs:
        print("\n📂 Carregando projetos de código...")
        for project in tqdm(project_dirs, desc="Projetos"):
            reader = SimpleDirectoryReader(
                input_dir=project,
                recursive=True,
                required_exts=[".js", ".json", ".md", ".txt", ".py"]
            )
            documents.extend(reader.load_data())
    
    # 2. Carrega conhecimento pessoal
    if os.path.exists(knowledge_path):
        print("\n📚 Carregando knowledge base...")
        reader = SimpleDirectoryReader(
            input_dir=knowledge_path,
            required_exts=[".txt", ".md", ".csv"]
        )
        documents.extend(reader.load_data())
    
    # 3. Carrega histórico de chats da pasta memory
    memory_chats_path = "memory/chats"
    if os.path.exists(memory_chats_path):
        print("\n💾 Carregando histórico de conversas...")
        for chat_file in os.listdir(memory_chats_path):
            if chat_file.endswith('.json'):
                try:
                    with open(os.path.join(memory_chats_path, chat_file), 'r') as f:
                        chat_data = json.load(f)
                        for message in chat_data:
                            doc = Document(
                                text=f"Conversa em {message.get('timestamp')}:\n"
                                     f"Pergunta: {message.get('query')}\n"
                                     f"Resposta: {message.get('response', '')}\n"
                                     f"Erro: {message.get('error', '')}",
                                metadata={
                                    "source": "memory/chats",
                                    "type": "chat_history",
                                    "timestamp": message.get('timestamp')
                                }
                            )
                            documents.append(doc)
                except Exception as e:
                    print(f"⚠️ Erro ao carregar {chat_file}: {str(e)}")
    
    # 4. Carrega correções do arquivo corrections.json
    corrections_path = "memory/corrections.json"
    if os.path.exists(corrections_path):
        print("\n🔧 Carregando correções...")
        try:
            with open(corrections_path, 'r') as f:
                corrections_data = json.load(f)
                for correction in corrections_data.get("corrections", []):
                    doc = Document(
                        text=f"Correção em {correction.get('timestamp')}:\n"
                             f"Pergunta original: {correction.get('original')}\n"
                             f"Resposta errada: {correction.get('wrong')}\n"
                             f"Correção: {correction.get('corrected')}",
                        metadata={
                            "source": "memory/corrections",
                            "type": "correction",
                            "timestamp": correction.get('timestamp')
                        }
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"⚠️ Erro ao carregar correções: {str(e)}")
    
    print(f"\n✅ Total de documentos carregados: {len(documents)}")
    return documents

def create_jarvis_engine():
    # Configurações otimizadas para o Jarvis
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"  # Embedding mais eficiente
    )

    memory = ChatMemoryBuffer.from_defaults(
        chat_history=memory_manager.get_corrections(),
        token_limit=4000
    )
    
    Settings.llm = Ollama(
        model="jarvis",
        system_prompt=(
            "Você é o Jarvis, assistente pessoal de Yuri Pinheiro Bernardi. "
            "Sempre que perguntarem sobre o Yuri, consulte os arquivos pessoais. "
            "Formato de resposta preferido:\n"
            "1. Dados dos arquivos quando disponíveis\n"
            "2. Respostas diretas\n"
            "3. Tom profissional mas amigável"
        ),
        request_timeout=86400.0,
        temperature=0.1  # Para respostas mais precisas
    )

    # Adicione este parser especial para informações pessoais
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=80,
        embed_model=Settings.embed_model,
        include_metadata=True  # Crucial para rastrear a fonte
    )


    # Sistema de cache com versão
    cache_dir = "cache/jarvis_v1"
    if os.path.exists(cache_dir):
        print("\n♻️ Carregando conhecimento do cache...")
        storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("\n🔨 Construindo novo índice para o Jarvis...")
        documents = load_data_sources()
        
        print("⏳ Processando documentos...")
        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
        
        print(f"\n🧠 Nós de conhecimento criados: {len(nodes)}")
        index = VectorStoreIndex(nodes, show_progress=True)
        
        print("\n💾 Salvando cache...")
        os.makedirs(cache_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=cache_dir)
    
    return index.as_chat_engine(
    chat_mode="condense_plus_context",
    timeout=86400,
    memory=memory,
    verbose=False,  # Desliga os logs internos
    system_prompt=(
        "Você é o Jarvis, assistente pessoal de Yuri Pinheiro Bernardi. "
        "Use as informações dos documentos para responder de forma precisa. "
        "Quando relevante, mencione que a informação vem dos arquivos pessoais."
    )
)