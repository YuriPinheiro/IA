import json
from datetime import datetime
from pathlib import Path
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import MessageRole

class MemoryManager:
    def __init__(self, memory_dir="memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        (self.memory_dir/"chats").mkdir(exist_ok=True)
        
        self.corrections_file = self.memory_dir/"corrections.json"
        if not self.corrections_file.exists():
            with open(self.corrections_file, "w") as f:
                json.dump({"corrections": []}, f)

    def save_chat(self, messages):
        today = datetime.now().strftime("%Y-%m-%d")
        chat_file = self.memory_dir/f"chats/{today}_chat.json"
        
        try:
            existing = json.loads(chat_file.read_text()) if chat_file.exists() else []
        except:
            existing = []
            
        existing.append(messages)
        chat_file.write_text(json.dumps(existing, indent=2))

    def add_correction(self, original_query, wrong_response, corrected_response):
        data = json.loads(self.corrections_file.read_text())
        data["corrections"].append({
            "original": original_query,
            "wrong": wrong_response,
            "corrected": corrected_response,
            "timestamp": str(datetime.now())
        })
        self.corrections_file.write_text(json.dumps(data, indent=2))

    def get_corrections(self):
        """Retorna as correções no formato esperado pelo ChatMemoryBuffer"""
        corrections_data = json.loads(self.corrections_file.read_text())["corrections"]
        
        chat_messages = []
        for correction in corrections_data:
            # Adiciona a pergunta original como mensagem do usuário
            chat_messages.append(ChatMessage(
                role=MessageRole.USER,
                content=correction["original"]
            ))
            
            # Adiciona a resposta errada como mensagem do assistente
            chat_messages.append(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=correction["wrong"]
            ))
            
            # Adiciona a correção como mensagem do usuário
            chat_messages.append(ChatMessage(
                role=MessageRole.USER,
                content=f"[CORREÇÃO] A resposta correta é: {correction['corrected']}"
            ))
        
        return chat_messages