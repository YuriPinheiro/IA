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
**Ferramentas:** R, caret, ROCR, paralelismo.  

### Classificação de Movimentos de Robô (QDA vs Random Forest)  
**Objetivo:** Comparar modelos paramétricos/não paramétricos para prever direções de robô a partir de sensores.  
**Técnicas:**  
- Pré-processamento: Normalização, PCA, remoção de correlações.  
- Modelos: QDA (paramétrico) e Random Forest (não paramétrico).  
- Validação: 10-Fold Cross-Validation.  
**Ferramentas:** R, caret, paralelismo, visualização com ggplot2.  

# Comparativo de Classificadores para Navegação de Robôs e Segmentação de Pele

## 🚀 Wall-following Robot Navigation

### 🔍 Melhores Resultados
| Método | Acurácia | Tempo/Treinamento | Destaques |
|--------|----------|-------------------|-----------|
| Particle Swarm Optimization | 98.8% | 1/4 do tempo vs grid search | Usa 1/4 dados para treino, 5-fold CV |
| CNN (Conv. Neural Networks) | ~98% | - | Superou SVC, ANN e MLR |
| Gradient Descent NN | 92.67% (2 sensores) | - | 46.5% com 24 sensores |

### ⚠️ Resultados Limitados
- **Gravitational Search + FFNN**: 69.72%  
- **Elman Network/MLP/ME**: Sem métricas claras  

## 🖥️ Skin Segmentation 

### 🔍 Melhores Resultados
| Método | Acurácia | Validação | Dados |
|--------|----------|-----------|-------|
| Fuzzy Decision Tree | 94.1% | 10-fold CV | - |
| ANFIS (27 rules) | 90.1% | Holdout CV | - |
| ANN (RGB) | - | 70/15/15 split | - |

### ⚠️ Resultados Limitados
- **CVNN (HSV)**: 77.56% (erro 0.704)  
- **Semi-supervised**: Sem métricas definidas  

## 🔑 Conclusões
- **Robôs**: PSO e CNN são os mais eficazes (>98%)  
- **Pele**: Fuzzy Decision Tree lidera (94.1%)  
- **Validação**: 10-fold CV é padrão ouro na maioria dos estudos  
