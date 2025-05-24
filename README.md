# Projeto de Previsão de Tarifas de Táxi em Nova Iorque

## Descrição do Problema

Este projeto, desenvolvido no âmbito da unidade curricular de Ciência de Dados da Universidade da Madeira, tem como objetivo construir um modelo de Machine Learning capaz de prever o preço das viagens de táxi em Nova Iorque. A previsão é baseada em diversas variáveis disponíveis no *dataset* "New York City Taxi Trips", como a distância da viagem, o horário do dia, o número de passageiros e as condições de tráfego. [cite: 41, 42]

## Objetivo

O principal objetivo do projeto é estimar o valor das tarifas de táxi em Nova Iorque. A tarefa foi abordada sob duas perspetivas: [cite: 43]

1.  **Regressão**: Construção de um modelo que preveja com precisão o valor exato da tarifa para uma dada viagem, com base nas características da mesma. Esta é uma tarefa de previsão de variável-alvo contínua. [cite: 44, 45]
2.  **Classificação**: Reformulação do problema para classificar as viagens em faixas de preços predefinidas: [cite: 45]
    * **Classe 1**: Viagens curtas, de baixo custo (`< $10`). [cite: 46]
    * **Classe 2**: Viagens de média distância, com tarifa moderada (`$10 - $30`). [cite: 46]
    * **Classe 3**: Viagens longas, com tarifa mais alta (`$30 - $60`). [cite: 47]
    * **Classe 4**: Tarifas *premium* (`> $60`). [cite: 47]

## Descrição dos Dados

O *dataset* utilizado, **New York City Yellow Taxi Trip Records**, contém informações detalhadas sobre viagens de táxi na cidade de Nova Iorque, incluindo dados sobre tempo, distância, localizações de embarque e desembarque, e informações de pagamento.

As principais colunas do *dataset* incluem:

* **VendorID**: Código que identifica o fornecedor do sistema de processamento eletrónico de pagamentos (TPEP).
    * 1 = Creative Mobile Technologies, LLC
    * 2 = VeriFone Inc.
* **tpep_pickup_datetime**: Data e hora em que o taxímetro foi ativado.
* **tpep_dropoff_datetime**: Data e hora em que o taxímetro foi desativado.
* **Passenger_count**: Número de passageiros no veículo (valor inserido pelo motorista).
* **Trip_distance**: Distância percorrida na viagem (em milhas) reportada pelo taxímetro.
* **PULocationID**: Código da zona TLC onde o taxímetro foi ativado (embarque).
* **DOLocationID**: Código da zona TLC onde o taxímetro foi desativado (desembarque).
* **RateCodeID**: Código da tarifa aplicada no final da viagem.
    * 1 = Tarifa padrão
    * 2 = JFK
    * 3 = Newark
    * 4 = Nassau ou Westchester
    * 5 = Tarifa negociada
    * 6 = Viagem em grupo
* **Store_and_fwd_flag**: Indica se o registo da viagem foi armazenado antes de ser enviado para o fornecedor, devido à falta de conexão do veículo com o servidor.
    * Y = Viagem armazenada antes do envio
    * N = Viagem enviada em tempo real
* **Payment_type**: Código que indica a forma de pagamento utilizada pelo passageiro.
    * 1 = Cartão de crédito
    * 2 = Dinheiro
    * 3 = Sem cobrança
    * 4 = Disputa
    * 5 = Desconhecido
    * 6 = Viagem cancelada
* **Fare_amount**: Valor da tarifa baseado no tempo e na distância percorrida.
* **Extra**: Cobranças adicionais, como sobretaxas noturnas ou de horário de pico.
* **MTA_tax**: Taxa de `$0,50` do MTA aplicada automaticamente com base na tarifa do taxímetro.
* **Improvement_surcharge**: Taxa de melhoria de `$0,30` aplicada desde 2015.
* **Tip_amount**: Valor da gorjeta (preenchido automaticamente para pagamentos com cartão; gorjetas em dinheiro não são registadas).
* **Tolls_amount**: Valor total de portagens pagas durante a viagem.
* **Total_amount**: Valor total cobrado do passageiro (não inclui gorjetas pagas em dinheiro).

## Procedimento (Ciclo de Vida da Análise de Dados)

O projeto seguiu as seis fases do ciclo de vida da análise de dados.

### 1. Formulação do Problema (Fase 1) [cite: 1]

Nesta fase inicial, o problema que o *dataset* aborda foi claramente definido. Os objetivos da análise de dados foram especificados. [cite: 1, 2]

### 2. Análise e Limpeza de Dados (Fase 2) [cite: 2]

A fase de pré-processamento envolveu a limpeza e transformação dos dados para os tornar adequados para a modelagem. As etapas realizadas incluem:

* **Pré-processamento**: O *dataset*, a sua fonte e quaisquer passos de pré-processamento tomados foram descritos. Incluiu-se a limpeza de dados e a normalização/padronização. [cite: 3]
* **Análise Exploratória de Dados (EDA)**: Estatísticas descritivas e visualizações foram conduzidas para compreender os dados. Métodos estatísticos padrão (como histogramas) foram usados para identificar padrões, *outliers* e correlações. [cite: 4]
* **Redução de Dimensão**: Métodos de redução de dimensão foram usados para identificar padrões nos dados, incluindo pelo menos um método linear e um não-linear (por exemplo, PCA e UMAP). [cite: 5]
* **Insights Iniciais**: Quaisquer *insights* iniciais obtidos da EDA foram discutidos. [cite: 6]
* **Engenharia de Features**: Criação de novas *features* relevantes para o modelo de previsão. Algumas das *features* criadas incluem:
    * `pickup_hour`: Hora do dia de início da viagem.
    * `dropoff_day_of_month`: Dia do mês de término da viagem.
    * `trip_distance_month`: Distância da viagem por mês.
    * `pickup_seconds`: Tempo de início da viagem em segundos.
    * `dropoff_seconds`: Tempo de fim da viagem em segundos.
    * Estatísticas agregadas como média de `extra`, `MTA_tax`, `tolls_amount` e percentis de `trip_distance` e `fare_amount`.
* **Testes de Hipóteses**: Hipóteses nulas e alternativas foram formuladas. Testes estatísticos apropriados foram escolhidos. Os testes de hipóteses foram realizados e os resultados interpretados. [cite: 7]

### 3. Seleção de Modelos (Fase 3)

Nesta fase, realizou-se a engenharia de *features*, produzindo pelo menos 10 novas *features*. A seleção do modelo foi iniciada, examinando o conjunto de modelos que seriam adequados para o problema em questão. [cite: 8] O método de validação de modelo mais adequado foi avaliado. [cite: 9] Todas as análises foram justificadas. O Random Forest foi identificado como um modelo apropriado devido à presença de relações não lineares, a sua capacidade de computar a importância das *features* e a sua robustez a *outliers* e ruído.

### 4. Construção do Modelo (Fase 4) [cite: 10, 11]

Esta fase envolveu a implementação de modelos de Machine Learning em Python: [cite: 10]

* **Algoritmo KNN**: O algoritmo KNN foi implementado de raiz utilizando apenas arrays NumPy com uma estrutura de dados adequada. [cite: 11] Uma explicação e documentação detalhadas do algoritmo produzido foram fornecidas. [cite: 12] O algoritmo implementado foi aplicado ao *dataset* e a sua performance avaliada. [cite: 13] Os resultados foram discutidos. [cite: 13]
* **Aprendizagem Supervisionada (Scikit-learn)**: Pelo menos dois modelos de aprendizagem supervisionada (além do kNN) foram testados utilizando a biblioteca `sklearn`. [cite: 14] Os resultados foram discutidos. [cite: 14]
* **Modelo Ensemble**: Dois modelos de *ensemble* adequados foram escolhidos, um com *bagging* e um com *boosting*, e aplicados ao problema. [cite: 15] Os resultados foram discutidos. [cite: 16]
* **Modelo de Deep Learning**: Uma arquitetura de *deep learning* foi escolhida (com implementação camada por camada ou *transfer learning*) e o modelo de *deep learning* foi implementado utilizando TensorFlow ou PyTorch. [cite: 16] Os resultados foram discutidos. [cite: 17]
* **Clustering**: Pelo menos dois algoritmos de *clustering* foram aplicados, variando o número de *clusters* para avaliar a presença de padrões nos dados. [cite: 17] Os resultados foram discutidos. [cite: 18]

### 5. Comparação e Avaliação do Modelo (Fase 5) [cite: 18]

Nesta fase, a performance dos modelos foi comparada e avaliada:

* **Comparação de Performance**: A performance dos modelos foi comparada numa tabela com representação adequada. [cite: 18]
* **Métricas de Avaliação**: Métricas de avaliação apropriadas foram usadas. [cite: 19]
* **Análise de Pontos Fortes e Fracos**: Os pontos fortes e fracos de cada abordagem foram discutidos. [cite: 19]
* **Resumo e Insights**: As principais descobertas e *insights* de todo o projeto foram sumarizados. [cite: 20]
* **Recomendações**: Recomendações para futuras melhorias ou ações foram fornecidas. [cite: 20]

### 6. Operacionalização (Fase 6) [cite: 21]

A fase final do projeto focou na preparação para a implementação dos modelos num ambiente de produção:

* **Relatório Final e Documento Técnico**: O relatório final e o documento técnico descrevendo o código foram produzidos. [cite: 21]
* **Apresentação**: A apresentação foi preparada. [cite: 21]
* **Plano de Implementação**: Foi discutido como implementar os modelos num ambiente de produção (*deployment plan*). [cite: 22]

## Entregas (*Deliverables*)

* **Mid-journey Report (Parte 1)**: Relatório detalhado com justificação de todas as ações e discussão dos *insights* fornecidos pelos dados. [cite: 23]
* **Final Report (Parte 2)**: Um relatório abrangente documentando cada fase, incluindo trechos de código, visualizações e interpretações. O documento deve também resumir as lições aprendidas durante o projeto, incluindo desafios enfrentados, descobertas inesperadas e *insights* obtidos. [cite: 24]
* **Executive Summary (Parte 2)**: Uma versão condensada do relatório final (duas páginas), adequada para *stakeholders* não técnicos, destacando os principais achados, *insights* e recomendações concisamente. [cite: 25]
* **Código**: [cite: 26]
    * Código R e Python, claramente comentado para entendimento com documentação apropriada. [cite: 26] Para Python, usar estilo procedural (usando funções para a maioria dos cálculos) ou programação orientada a objetos. [cite: 27]
    * **Data Exploration and Preprocessing Notebook (Parte 1)**: Um Jupyter notebook ou equivalente detalhando as etapas de exploração e pré-processamento aplicadas aos dados brutos. [cite: 28] Isso deve incluir limpeza de dados, engenharia de *features* e quaisquer transformações realizadas no *dataset*. [cite: 29]
    * **Modeling Notebook (Parte 2)**: Um notebook que descreve o processo de modelagem, incluindo a escolha de algoritmos, ajuste de hiperparâmetros, resultados de validação cruzada e métricas de desempenho do modelo final. [cite: 30] Deve ser acompanhado por visualizações ou gráficos ilustrando o comportamento do modelo. [cite: 31]
    * **Codebase (Parte 2)**: A base de código contendo *scripts* ou *notebooks* usados ao longo do projeto, para garantir que a análise e os processos de modelagem sejam reproduzíveis e possam ser facilmente partilhados ou transferidos para outros membros da equipa. [cite: 32]
    * **Trained Models (Parte 2)**: Os modelos finais treinados devem ser guardados e documentados para permitir possível implementação futura num ambiente de produção ou para análise adicional. [cite: 33] Os modelos devem ser fornecidos numa pasta compactada (*zip folder*). [cite: 33]
    * **Visualization Dashboard (Opcional, Parte 2)**: Um *dashboard* interativo ou ferramenta de visualização permitindo que os *stakeholders* explorem visualmente os resultados. [cite: 34]
    * **Documentation (Partes 1 e 2)**: Documentação abrangente para o código, modelos e quaisquer outros componentes relevantes para facilitar futura manutenção, colaboração e transferência de conhecimento. [cite: 35]
* **Final Presentation and Defense**: Cada grupo deve preparar uma apresentação concisa de 15 minutos resumindo os principais achados e metodologias. [cite: 36] Seguido por uma defesa de 15 minutos. [cite: 37]

## Notas Importantes

* Todos os relatórios devem ser feitos usando um editor LaTeX (Overleaf é recomendado) utilizando o estilo IEEE de coluna dupla ou estilo de coluna única. [cite: 38]
* Todo o código deve ser mantido num repositório privado no GitHub para controlo de versão. [cite: 39]
* Cada grupo é composto por dois estudantes (podendo ser apenas um), e cada membro do grupo deve conhecer a implementação e o código desenvolvido em detalhe para a defesa. [cite: 40]

## Dados e Modelos

* **Dataset**: [New York City Taxi Trips 2019 (Kaggle)](https://www.kaggle.com/datasets/dhruvildave/new-york-city-taxi-trips-2019/data) [cite: 41]
* **Dataset Processado**: [Link para o dataset processado](https://drive.google.com/file/d/1VdCwn-FpwIJ4Ox7nWTWjZCORiRnxpy9Q/view?usp=drive_link)
* **Modelos Treinados**: [Link para os modelos treinados](https://drive.google.com/file/d/1dTNM6WOI9Bmg2-yV_7upejPfIOrfZsiY/view?usp=sharing)
    * *Os modelos devem ser colocados na pasta `out/models`.*