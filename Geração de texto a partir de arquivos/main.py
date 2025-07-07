import os
from rag_config import create_jarvis_engine
from memory_manager import MemoryManager
from datetime import datetime

# Inicializa o gerenciador de memória
memory = MemoryManager()

# Verifica e cria estrutura de diretórios se necessário
os.makedirs("repos", exist_ok=True)
os.makedirs("knowledge", exist_ok=True)

# Inicializa o Jarvis
print("\n🔮 Inicializando Jarvis...")
jarvis = create_jarvis_engine()

# Loop de interação
print("\n✅ Jarvis pronto! Posso ajudar com:")
print("- Seus projetos em /repos")
print("- Documentos pessoais em /knowledge")
print("\nComandos especiais:")
print("- 'atualizar': Recarrega a base de conhecimento")
print("- 'corrigir': Ativa o modo de correção da última resposta")
print("- 'sair': Encerra a sessão\n")

def handle_feedback(query, full_response):
    """Gerencia o sistema de feedback e correções"""
    print(f"\n🤖 Resposta Completa:\n{full_response}")
    feedback = input("\n🔧 A resposta está correta? (s/n) ou 'pular': ")
    
    if feedback.lower() == 'n':
        correction = input("📝 Digite a resposta correta: ")
        memory.add_correction(query, full_response, correction)
        print("✅ Correção salva para aprendizado futuro!")
    elif feedback.lower() == 'sair':
        return 'sair'
    return None

while True:
    try:
        query = input("\n💬 Você: ").strip()
        
        # Comandos especiais
        if query.lower() in ["sair", "exit", "quit"]:
            print("\n👋 Até logo!")
            break
            
        if query.lower() == "atualizar":
            print("\n🔄 Atualizando base de conhecimento...")
            jarvis = create_jarvis_engine()
            continue
        
        # Processa a consulta com streaming
        print("\n⚙️ Processando...\n")
        print("🤖 Jarvis:", end="", flush=True)
        
        full_response = ""
        response_stream = jarvis.stream_chat(query)
        
        # Processa o stream em tempo real
        for chunk in response_stream.response_gen:
            print(chunk, end="", flush=True)
            full_response += chunk
        
        # Salva a conversa completa
        memory.save_chat({
            "query": query,
            "response": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Opção de feedback
        if handle_feedback(query, full_response) == 'sair':
            break
            
    except KeyboardInterrupt:
        print("\n🛑 Operação interrompida pelo usuário")
        break
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        memory.save_chat({
            "query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        continue