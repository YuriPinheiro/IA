# Limpeza das variáveis 
rm(list = ls())

# Troca de diretório de trabalho 
setwd("C:\\Users\\PICHAU\\Documents\\Computação aplicada\\Projeto 2")

# 1 Leitura dos dados 
dados = read.csv("sensor_readings_24.data",header = F, na.strings = "?")

# 2 Ajuste prévio dos dados
# 2.1 Renomeada coluna rótulo
names(dados)[25] = "CLASS"

# 2.2 Renomeado nomes dos rótulos 
dados$CLASS = factor(dados$CLASS, labels = c("Frente", "ADir", "LEsq", "LDir"))

# 3 Divisão do conjunto de dados em 10 partições 
library(caret)
rotuloIndice = which("CLASS" == names(dados))
folds = createFolds(dados[,rotuloIndice], k=10)

# 3.1 Análise da separação por partição 
summary(folds)

library(ggplot2)
library(gridExtra)

p = list()

for (i in 1:10) {
  dadosAnalise = as.data.frame(dados[folds[[i]],])
  p[[i]] = ggplot(dadosAnalise,aes(CLASS,fill=CLASS)) + geom_bar() + 
    geom_text(aes(label = sprintf("%0.1f%%",stat(prop)*100),group=1), stat = "count", vjust = 1, 
              colour = "white") +
    theme(axis.title.x=element_blank(),legend.position="none")
  # p[[i]] = ggplot(dadosAnalise, aes(CLASS, fill=CLASS)) + geom_bar()
}

do.call(grid.arrange, c(p, ncol=2))

# 4 Configuração experimental 10-Fold cross validation 

# 4.1 criado base do experimento para todos subconjuntos 
desempenho.parametrico = data.frame(Acuracia=double(),Sensibilidade=double(),
                                    Especificidade=double(),Precisao=double(),F1=double(),
                                    stringsAsFactors=FALSE)

desempenho.nao.parametrico = data.frame(Acuracia=double(),Sensibilidade=double(),
                                        Especificidade=double(),Precisao=double(),F1=double(),
                                        stringsAsFactors=FALSE)

# 4.2 Iteração de treinamento e teste em cada subconjunto
for(i in 1:10){
  dadosTeste = as.data.frame(dados[folds[[i]],])
  dadosTreinamento = as.data.frame(dados[unlist(folds[-i]),])
  
  # remoção de outliers 
  # conforme visto no projeto 1, identificamos que removendo os outliers podemos inviabilizar a base, portanto não será aplicado a remoção
  
  rotuloIndice = which("CLASS" == names(dadosTreinamento))
  
  # normalização 
  escorezParams = preProcess(dadosTreinamento[,-rotuloIndice], method=c("center", "scale"))
  escorezNorm = predict(escorezParams, dadosTreinamento[,-rotuloIndice])
  dadosTreinamento = data.frame(c(escorezNorm, dadosTreinamento[rotuloIndice]))
  
  escorezNorm = predict(escorezParams, dadosTeste[,-rotuloIndice])
  dadosTeste = data.frame(c(escorezNorm, dadosTeste[rotuloIndice]))
  
  # remoção de caracteristicas correlacionas 
  dadosCorrelacao = cor(dadosTreinamento[,-rotuloIndice])
  
  indicesCorrelacaoForte = findCorrelation(dadosCorrelacao, cutoff=0.95, verbose=T)
  
  if(length(indicesCorrelacaoForte) > 0){
    dadosTreinamento = dadosTreinamento[,-indicesCorrelacaoForte]
    dadosTeste = dadosTeste[,-indicesCorrelacaoForte]
  }
  
  # extração de características 
  pca = prcomp(dadosTreinamento[,-rotuloIndice]) 
  
  numeroComponente = min(which(summary(pca)$importance[3,]>0.95))
  dadosReduzidos = predict(pca, dadosTreinamento[,-rotuloIndice])[,1:numeroComponente]
  dadosTreinamento = data.frame(dadosReduzidos,dadosTreinamento[rotuloIndice])
  
  dadosReduzidos = predict(pca, dadosTeste[,-rotuloIndice])[,1:numeroComponente]
  dadosTeste = data.frame(dadosReduzidos,dadosTeste[rotuloIndice])
  
  rotuloIndice = which("CLASS" == names(dadosTreinamento))
  
  # treinamento 
  library(doParallel)
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl)
  
  # treinamento paramétrico 
  set.seed(6)
  paramsTreinamento = trainControl(method = "none", classProbs = TRUE)
  qda.modelo = train(dadosTreinamento[,-rotuloIndice],    #dados
                     dadosTreinamento[,rotuloIndice],     #rótulos 
                     trControl = paramsTreinamento, 
                     method="qda")
  
  # treinamento não-paramétrico 
  set.seed(66)
  paramsTreinamento = trainControl(classProbs = TRUE)
  rf.modelo = train(dadosTreinamento[,-rotuloIndice],   #dados
                     dadosTreinamento[,rotuloIndice],    #rótulos
                     trControl = paramsTreinamento, 
                     tuneLength = 10, 
                     method = "rf")
  
  stopCluster(cl)
  
  # teste 
  
  # teste paramétrico 
  qda.predicao = predict(qda.modelo, dadosTeste[,-rotuloIndice])
  res = confusionMatrix(qda.predicao, dadosTeste[,rotuloIndice], positive = "Frente")
  desempenho.parametrico[i,1] = res$overall[1]
  desempenho.parametrico[i,2] = res$byClass[1]
  desempenho.parametrico[i,3] = res$byClass[2]
  desempenho.parametrico[i,4] = res$byClass[5]
  desempenho.parametrico[i,5] = res$byClass[7]
  
  # teste não-paramétrico 
  rf.predicao = predict(rf.modelo, dadosTeste[,-rotuloIndice])
  res = confusionMatrix(rf.predicao, dadosTeste[,rotuloIndice], positive = "Frente")
  desempenho.nao.parametrico[i,1] = res$overall[1]
  desempenho.nao.parametrico[i,2] = res$byClass[1]
  desempenho.nao.parametrico[i,3] = res$byClass[2]
  desempenho.nao.parametrico[i,4] = res$byClass[5]
  desempenho.nao.parametrico[i,5] = res$byClass[7]
}

# 5 Apresentação do desempenho - Matriz de confusão

print("QDA - Paramétrico")
print(sprintf("A acurácia média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,1])*100, sd(desempenho.parametrico[,1])*100))
print(sprintf("A sensibilidade média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,2])*100, sd(desempenho.parametrico[,2])*100))
print(sprintf("A especificidade média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,3])*100, sd(desempenho.parametrico[,3])*100))
print(sprintf("A precisão média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,4])*100, sd(desempenho.parametrico[,4])*100))
print(sprintf("A F1 média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,5])*100, sd(desempenho.parametrico[,5])*100))

print("Random forest - Não Paramétrico")
print(sprintf("A acurácia média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,1])*100, sd(desempenho.nao.parametrico[,1])*100))
print(sprintf("A sensibilidade média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,2])*100, sd(desempenho.nao.parametrico[,2])*100))
print(sprintf("A especificidade média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,3])*100, sd(desempenho.nao.parametrico[,3])*100))
print(sprintf("A precisão média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,4])*100, sd(desempenho.nao.parametrico[,4])*100))
print(sprintf("A F1 média foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,5])*100, sd(desempenho.nao.parametrico[,5])*100))

# 6 Apresentação do desempenho - Curva ROC
# Por possuir 4 classes a curva roc não se aplica.
