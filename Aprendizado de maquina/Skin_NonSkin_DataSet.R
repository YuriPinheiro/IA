# Limpeza das vari�veis 
rm(list = ls())

# Troca de diret�rio de trabalho
setwd("C:\\Users\\PICHAU\\Documents\\Computa��o aplicada\\Projeto 2")

# 1 Leitura dos dados 
dados = read.table("Skin_NonSkin.txt", header = F, na.strings = "?", sep = "")

# 2 Ajuste pr�vio dos dados
# 2.1 Cria��o do cabe�alho para todas as colunas
names(dados) = c("B","G","R","CLASS")

# 2.2 Ajustado r�tulos para "Pele" e "N�o.Pele" com factor
dados$CLASS = factor(dados$CLASS, labels=c("Pele","N�o.Pele"))

# 3 Divis�o do conjunto de dados

# 3.1 Divis�o do conjunto que ser� usado para treinamento e teste
library(caret)
indicesTreinamento = createDataPartition(dados$CLASS, 
                                         p = 1/10, list = FALSE)
dados = as.data.frame(dados[indicesTreinamento,])

# 3.2 Divis�o do conjunto de dados em 10 parti��es
rotuloIndice = which("CLASS" == names(dados))
folds = createFolds(dados[,rotuloIndice], k=10)

# 3.3 An�lise da separa��o por parti��o 
# summary(folds)
# 
# library(ggplot2)
# library(gridExtra)
# 
# p = list()
# 
# for (i in 1:10) {
#   dadosAnalise = as.data.frame(dados[folds[[i]],])
#   p[[i]] = ggplot(dadosAnalise,aes(CLASS,fill=CLASS)) + geom_bar() + 
#     geom_text(aes(label = sprintf("%0.1f%%",stat(prop)*100),group=1), stat = "count", vjust = 1.5, 
#               colour = "white") +
#     theme(axis.title.x=element_blank(),legend.position="none")
#   # p[[i]] = ggplot(dadosAnalise, aes(CLASS, fill=CLASS)) + geom_bar()
# }
# 
# do.call(grid.arrange, c(p, ncol=2))

# 4 Configura��o experimental 10-Fold cross validation

# 4.1 criado base do experimento para todos subconjuntos 
desempenho.parametrico = data.frame(Acuracia=double(),Sensibilidade=double(),
                                          Especificidade=double(),Precisao=double(),F1=double(),
                                          stringsAsFactors=FALSE)

desempenho.nao.parametrico = data.frame(Acuracia=double(),Sensibilidade=double(),
                                              Especificidade=double(),Precisao=double(),F1=double(),
                                              stringsAsFactors=FALSE)
parametrico.rocs = c()
nao.parametrico.rocs = c()

# 4.2 Itera��o de treinamento e teste em cada subconjunto
for (i in 1:10) {
  dadosTeste = as.data.frame(dados[folds[[i]],])
  dadosTreinamento = as.data.frame(dados[unlist(folds[-i]),])
  
  # remo��o dos outliers
  # n�o tem necessidade de remover os outliers, visto que no projeto 1 identificamos que n�o temos outliers
  
  rotuloIndice = which("CLASS" == names(dadosTreinamento))
  
  # normaliza��o 
  escorezParams = preProcess(dadosTreinamento[,-rotuloIndice], method=c("center", "scale"))
  escorezNorm = predict(escorezParams, dadosTreinamento[,-rotuloIndice])
  dadosTreinamento = data.frame(c(escorezNorm, dadosTreinamento[rotuloIndice]))
  
  escorezNorm = predict(escorezParams, dadosTeste[,-rotuloIndice])
  dadosTeste = data.frame(c(escorezNorm, dadosTeste[rotuloIndice]))
  
  # remo��o de caracteristicas correlacionas 
  dadosCorrelacao = cor(dadosTreinamento[,-rotuloIndice])
  
  indicesCorrelacaoForte = findCorrelation(dadosCorrelacao, cutoff=0.95, verbose=T)
  
  if(length(indicesCorrelacaoForte) > 0){
    dadosTreinamento = dadosTreinamento[,-indicesCorrelacaoForte]
    dadosTeste = dadosTeste[,-indicesCorrelacaoForte]
  }
  
  rotuloIndice = which("CLASS" == names(dadosTreinamento))
  
  # extra��o de caracter�sticas 
  pca = prcomp(dadosTreinamento[,-rotuloIndice]) 
  
  numeroComponente = min(which(summary(pca)$importance[3,]>0.95))
  dadosReduzidos = predict(pca, dadosTreinamento[,-rotuloIndice])[,1:numeroComponente]
  dadosTreinamento = data.frame(dadosReduzidos,dadosTreinamento[rotuloIndice])
  
  dadosReduzidos = predict(pca, dadosTeste[,-rotuloIndice])[,1:numeroComponente]
  dadosTeste = data.frame(dadosReduzidos,dadosTeste[rotuloIndice])
  
  rotuloIndice = which("CLASS" == names(dadosTreinamento))
  
  # treinamento 
  library(doParallel)
  cl <- makePSOCKcluster(6)
  registerDoParallel(cl)
  
  # treinamento param�trico 
  set.seed(6)
  paramsTreinamento = trainControl(method = "none", classProbs = TRUE)
  lda.modelo = train(dadosTreinamento[,-rotuloIndice],    #dados
                     dadosTreinamento[,rotuloIndice],     #r�tulos 
                     trControl = paramsTreinamento, 
                     method="lda")
  
  # treinamento n�o-param�trico 
  set.seed(66)
  paramsTreinamento = trainControl(classProbs = TRUE)
  svmPoly.modelo = train(dadosTreinamento[,-rotuloIndice],   #dados
                     dadosTreinamento[,rotuloIndice],    #r�tulos
                     trControl = paramsTreinamento, 
                     tuneLength = 4, 
                     method = "svmPoly")
  
   stopCluster(cl)
  
  # teste 
  
   library(ROCR)
   
  # teste param�trico 
  lda.predicao = predict(lda.modelo, dadosTeste[,-rotuloIndice])
  res = confusionMatrix(lda.predicao, dadosTeste[,rotuloIndice], positive = "Pele")
  desempenho.parametrico[i,1] = res$overall[1]
  desempenho.parametrico[i,2] = res$byClass[1]
  desempenho.parametrico[i,3] = res$byClass[2]
  desempenho.parametrico[i,4] = res$byClass[5]
  desempenho.parametrico[i,5] = res$byClass[7]
  
  lda.probs = predict(lda.modelo, dadosTeste[,-rotuloIndice],type="prob")
  lda.predictions = prediction(lda.probs[,1], dadosTeste[,rotuloIndice])
  lda.roc = performance(lda.predictions ,"tpr","fpr")
  parametrico.rocs[[i]] = lda.roc
  
  # teste n�o-param�trico 
  svmPoly.predicao = predict(svmPoly.modelo, dadosTeste[,-rotuloIndice])
  res = confusionMatrix(svmPoly.predicao, dadosTeste[,rotuloIndice], positive = "Pele")
  desempenho.nao.parametrico[i,1] = res$overall[1]
  desempenho.nao.parametrico[i,2] = res$byClass[1]
  desempenho.nao.parametrico[i,3] = res$byClass[2]
  desempenho.nao.parametrico[i,4] = res$byClass[5]
  desempenho.nao.parametrico[i,5] = res$byClass[7]
  
  svmPoly.probs = predict(svmPoly.modelo, dadosTeste[,-rotuloIndice],type="prob")
  svmPoly.predictions = prediction(svmPoly.probs[,1], dadosTeste[,rotuloIndice])
  svmPoly.roc = performance(svmPoly.predictions ,"tpr","fpr")
  nao.parametrico.rocs[[i]] = svmPoly.roc
}

# 5 Apresenta��o do desempenho - Matriz de confus�o

print("LDA - Param�trico")
print(sprintf("A acur�cia m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,1])*100, sd(desempenho.parametrico[,1])*100))
print(sprintf("A sensibilidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,2])*100, sd(desempenho.parametrico[,2])*100))
print(sprintf("A especificidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,3])*100, sd(desempenho.parametrico[,3])*100))
print(sprintf("A precis�o m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,4])*100, sd(desempenho.parametrico[,4])*100))
print(sprintf("A F1 m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,5])*100, sd(desempenho.parametrico[,5])*100))

print("SVM Polinomial - N�o Param�trico")
print(sprintf("A acur�cia m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,1])*100, sd(desempenho.nao.parametrico[,1])*100))
print(sprintf("A sensibilidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,2])*100, sd(desempenho.nao.parametrico[,2])*100))
print(sprintf("A especificidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,3])*100, sd(desempenho.nao.parametrico[,3])*100))
print(sprintf("A precis�o m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,4])*100, sd(desempenho.nao.parametrico[,4])*100))
print(sprintf("A F1 m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,5])*100, sd(desempenho.nao.parametrico[,5])*100))

# 6 Apresenta��o do desempenho - Curva ROC

colors = c("red", "blue", "green", "red", "blue", "green", "red", "blue", "green", "red")

library(ROCR)

for (i in 1:10) {
  if(i == 1){
    plot(parametrico.rocs[[i]], col=colors[[i]], main = 'ROC')  
  }
  plot(parametrico.rocs[[i]], col=colors[[i]], main = 'ROC', add=T)
}

for (i in 1:10) {
  if(i == 1){
    plot(nao.parametrico.rocs[[i]], col=colors[[i]], main = 'ROC')  
  }
  plot(nao.parametrico.rocs[[i]], col=colors[[i]], main = 'ROC', add=T)
}

