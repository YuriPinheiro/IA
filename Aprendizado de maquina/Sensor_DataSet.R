# Limpeza das vari�veis 
rm(list = ls())

# Troca de diret�rio de trabalho 
setwd("C:\\Users\\PICHAU\\Documents\\Computa��o aplicada\\Projeto 2")

# 1 Leitura dos dados 
dados = read.csv("sensor_readings_24.data",header = F, na.strings = "?")

# 2 Ajuste pr�vio dos dados
# 2.1 Renomeada coluna r�tulo
names(dados)[25] = "CLASS"

# 2.2 Renomeado nomes dos r�tulos 
dados$CLASS = factor(dados$CLASS, labels = c("Frente", "ADir", "LEsq", "LDir"))

# 3 Divis�o do conjunto de dados em 10 parti��es 
library(caret)
rotuloIndice = which("CLASS" == names(dados))
folds = createFolds(dados[,rotuloIndice], k=10)

# 3.1 An�lise da separa��o por parti��o 
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

# 4 Configura��o experimental 10-Fold cross validation 

# 4.1 criado base do experimento para todos subconjuntos 
desempenho.parametrico = data.frame(Acuracia=double(),Sensibilidade=double(),
                                    Especificidade=double(),Precisao=double(),F1=double(),
                                    stringsAsFactors=FALSE)

desempenho.nao.parametrico = data.frame(Acuracia=double(),Sensibilidade=double(),
                                        Especificidade=double(),Precisao=double(),F1=double(),
                                        stringsAsFactors=FALSE)

# 4.2 Itera��o de treinamento e teste em cada subconjunto
for(i in 1:10){
  dadosTeste = as.data.frame(dados[folds[[i]],])
  dadosTreinamento = as.data.frame(dados[unlist(folds[-i]),])
  
  # remo��o de outliers 
  # conforme visto no projeto 1, identificamos que removendo os outliers podemos inviabilizar a base, portanto n�o ser� aplicado a remo��o
  
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
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl)
  
  # treinamento param�trico 
  set.seed(6)
  paramsTreinamento = trainControl(method = "none", classProbs = TRUE)
  qda.modelo = train(dadosTreinamento[,-rotuloIndice],    #dados
                     dadosTreinamento[,rotuloIndice],     #r�tulos 
                     trControl = paramsTreinamento, 
                     method="qda")
  
  # treinamento n�o-param�trico 
  set.seed(66)
  paramsTreinamento = trainControl(classProbs = TRUE)
  rf.modelo = train(dadosTreinamento[,-rotuloIndice],   #dados
                     dadosTreinamento[,rotuloIndice],    #r�tulos
                     trControl = paramsTreinamento, 
                     tuneLength = 10, 
                     method = "rf")
  
  stopCluster(cl)
  
  # teste 
  
  # teste param�trico 
  qda.predicao = predict(qda.modelo, dadosTeste[,-rotuloIndice])
  res = confusionMatrix(qda.predicao, dadosTeste[,rotuloIndice], positive = "Frente")
  desempenho.parametrico[i,1] = res$overall[1]
  desempenho.parametrico[i,2] = res$byClass[1]
  desempenho.parametrico[i,3] = res$byClass[2]
  desempenho.parametrico[i,4] = res$byClass[5]
  desempenho.parametrico[i,5] = res$byClass[7]
  
  # teste n�o-param�trico 
  rf.predicao = predict(rf.modelo, dadosTeste[,-rotuloIndice])
  res = confusionMatrix(rf.predicao, dadosTeste[,rotuloIndice], positive = "Frente")
  desempenho.nao.parametrico[i,1] = res$overall[1]
  desempenho.nao.parametrico[i,2] = res$byClass[1]
  desempenho.nao.parametrico[i,3] = res$byClass[2]
  desempenho.nao.parametrico[i,4] = res$byClass[5]
  desempenho.nao.parametrico[i,5] = res$byClass[7]
}

# 5 Apresenta��o do desempenho - Matriz de confus�o

print("QDA - Param�trico")
print(sprintf("A acur�cia m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,1])*100, sd(desempenho.parametrico[,1])*100))
print(sprintf("A sensibilidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,2])*100, sd(desempenho.parametrico[,2])*100))
print(sprintf("A especificidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,3])*100, sd(desempenho.parametrico[,3])*100))
print(sprintf("A precis�o m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,4])*100, sd(desempenho.parametrico[,4])*100))
print(sprintf("A F1 m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.parametrico[,5])*100, sd(desempenho.parametrico[,5])*100))

print("Random forest - N�o Param�trico")
print(sprintf("A acur�cia m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,1])*100, sd(desempenho.nao.parametrico[,1])*100))
print(sprintf("A sensibilidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,2])*100, sd(desempenho.nao.parametrico[,2])*100))
print(sprintf("A especificidade m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,3])*100, sd(desempenho.nao.parametrico[,3])*100))
print(sprintf("A precis�o m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,4])*100, sd(desempenho.nao.parametrico[,4])*100))
print(sprintf("A F1 m�dia foi de %.2f%% (+/- %.2f%%)", mean(desempenho.nao.parametrico[,5])*100, sd(desempenho.nao.parametrico[,5])*100))

# 6 Apresenta��o do desempenho - Curva ROC
# Por possuir 4 classes a curva roc n�o se aplica.
