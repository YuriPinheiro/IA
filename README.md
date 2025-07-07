Projetos autodidatas feitos durante meu processo de aprendizado. Alguns podem conter erros ou não estar 100% otimizados. Estou aberto a sugestões e contribuições.


# J.A.R.V.I.S - Assistente Pessoal Inteligente

**Tecnologias Principais:**
- Arquitetura RAG (Retrieval-Augmented Generation)
- Modelo Ollama + LlamaIndex
- Embeddings: HuggingFace (bge-small-en-v1.5)
- Processamento de Linguagem Natural

## Funcionalidades

### 📂 Recuperação de Conteúdo
- Indexação inteligente de:
  - Repositórios de código (`/repos`)
  - Arquivos pessoais (`/knowledge`)
  - Histórico de conversas (`/memory/chats`)
  - Correções manuais (`/memory/corrections.json`)
- Divisão semântica de documentos (SemanticSplitterNodeParser)

### 💡 Motor de Geração
- Respostas contextualizadas e precisas
- Cache persistente para otimização
- Tom profissional adaptável
- Memória contextual (ChatMemoryBuffer)

### 🧠 Memória e Aprendizado
- Armazenamento completo do histórico
- Sistema de feedback e correções
- Aprendizado contínuo com interações
- Capacidade de reescrever respostas incorretas

## Benefícios
✔️ Consulta inteligente de projetos locais  
✔️ Personalização contínua  
✔️ Respostas baseadas em documentos reais  
✔️ Evolução constante com interações  

# Análise de dados

### Classificação de Pele vs Não-Pele (LDA vs SVM)  
**Objetivo:** Comparar modelos paramétricos/não paramétricos para detecção de pele em imagens.  
**Técnicas:**  
- Pré-processamento: Normalização, PCA, remoção de correlações.  
- Modelos: LDA (paramétrico) e SVM Polinomial (não paramétrico).  
- Validação: 10-Fold Cross-Validation.  
**Resultados:**  
- LDA: Acurácia `X%`, Sensibilidade `Y%`.  
- SVM: Acurácia `A%`, Sensibilidade `B%`.  
**Ferramentas:** R, caret, ROCR, paralelismo.  

### Classificação de Movimentos de Robô (QDA vs Random Forest)  
**Objetivo:** Comparar modelos paramétricos/não paramétricos para prever direções de robô a partir de sensores.  
**Técnicas:**  
- Pré-processamento: Normalização, PCA, remoção de correlações.  
- Modelos: QDA (paramétrico) e Random Forest (não paramétrico).  
- Validação: 10-Fold Cross-Validation.  
**Resultados:**  
- QDA: Acurácia `X%`, Sensibilidade `Y%`.  
- Random Forest: Acurácia `A%`, Sensibilidade `B%`.  
**Ferramentas:** R, caret, paralelismo, visualização com ggplot2.  
