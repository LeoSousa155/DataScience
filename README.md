# Descrição do problema

No âmbito da unidade curricular de Ciência de Dados, foi proposto o desenvolvimento de um modelo de machine learning utilizando o dataset "New York City Taxi Trips". O objetivo é construir um modelo capaz de prever o preço das viagens de táxi com base em diversas variáveis disponíveis, como distância da viagem, horário do dia, número de passageiros e possíveis condições de tráfego.

# Objetivo
O principal objetivo deste projeto é desenvolver um modelo que consiga estimar o valor das tarifas de táxi em Nova York. A tarefa será abordada sob duas perspectivas:

1. Regressão: Criar um modelo que consiga prever com precisão o valor exato da tarifa para uma determinada viagem.
2. Classificação: Reformular o problema para classificar as viagens em faixas de preços predefinidas:

    - Classe 1: Viagens curtas e de baixo custo (`< $10`)

    - Classe 2: Viagens de média distância e preço moderado (`$10 - $30`)

    - Classe 3: Viagens longas com tarifas mais elevadas (`$30 - $60`)

    - Classe 4: Tarifas premium (`> $60`)

# Descrição dos Dados

O dataset **New York City Yellow Taxi Trip Records** contém informações detalhadas sobre viagens de táxi na cidade de Nova York, incluindo dados sobre tempo, distância, localizações de embarque e desembarque, além de informações de pagamento.

O dataset inclui as seguintes colunas:

- **VendorID**: Código que identifica o provedor do sistema de processamento eletrônico de pagamentos (TPEP).
  - 1 = Creative Mobile Technologies, LLC
  - 2 = VeriFone Inc.

- **tpep_pickup_datetime**: Data e hora em que o taxímetro foi ativado.
- **tpep_dropoff_datetime**: Data e hora em que o taxímetro foi desativado.
- **Passenger_count**: Número de passageiros no veículo (valor inserido pelo motorista).
- **Trip_distance**: Distância percorrida na viagem (em milhas) reportada pelo taxímetro.
- **PULocationID**: Código da zona TLC onde o taxímetro foi ativado (embarque).
- **DOLocationID**: Código da zona TLC onde o taxímetro foi desativado (desembarque).
- **RateCodeID**: Código da tarifa aplicada ao final da viagem.
  - 1 = Tarifa padrão
  - 2 = JFK
  - 3 = Newark
  - 4 = Nassau ou Westchester
  - 5 = Tarifa negociada
  - 6 = Viagem em grupo

- **Store_and_fwd_flag**: Indica se o registro da viagem foi armazenado antes de ser enviado ao provedor, devido à falta de conexão do veículo com o servidor.
  - Y = Viagem armazenada antes do envio
  - N = Viagem enviada em tempo real

- **Payment_type**: Código que indica a forma de pagamento utilizada pelo passageiro.
  - 1 = Cartão de crédito
  - 2 = Dinheiro
  - 3 = Sem cobrança
  - 4 = Disputa
  - 5 = Desconhecido
  - 6 = Viagem cancelada

- **Fare_amount**: Valor da tarifa baseado no tempo e na distância percorrida.
- **Extra**: Cobranças adicionais, como sobretaxas noturnas ou de horário de pico.
- **MTA_tax**: Taxa de `$0,50` do MTA aplicada automaticamente com base na tarifa do taxímetro.
- **Improvement_surcharge**: Taxa de melhoria de `$0,30` aplicada desde 2015.
- **Tip_amount**: Valor da gorjeta (preenchido automaticamente para pagamentos com cartão; gorjetas em dinheiro não são registradas).
- **Tolls_amount**: Valor total de pedágios pagos durante a viagem.
- **Total_amount**: Valor total cobrado do passageiro (não inclui gorjetas pagas em dinheiro).

# Procedimento

## 1.ª FASE - DISCOVERY

Nesta fase inicial, o foco foi enquadrar o problema de negócio como um desafio de análise de dados. O dataset fornecido contém informações detalhadas sobre viagens de táxi em Nova Iorque ao longo de um ano. O objetivo principal é construir um modelo capaz de prever o valor da tarifa (`fare_amount`) com base nas diversas variáveis associadas a cada viagem, como horário de início e fim, distância percorrida e número de passageiros.

## 2.ª FASE - DATA PRE-PROCESSING

A fase de pré-processamento de dados envolveu a limpeza e transformação dos dados para torná-los adequados para modelagem. Foram realizadas as seguintes etapas:

- **Limpeza de dados:** Identificação e tratamento de valores ausentes, outliers e inconsistências nos dados.
- **Engenharia de features:** Criação de novas features que podem ser relevantes para o modelo de previsão. Algumas das novas features criadas incluem:
    - `pickup_hour`: Hora do dia em que a viagem começou.
    - `dropoff_day_of_month`: Dia do mês em que a viagem terminou.
    - `trip_distance_month`: Distância da viagem por mês.
    - `pickup_seconds`: Tempo de início da viagem em segundos.
    - `dropoff_seconds`: Tempo de fim da viagem em segundos.
    - Estatísticas agregadas como média de `extra`, `MTA_tax`, `tolls_amount` e percentis de `trip_distance` e `fare_amount`.

A importância das features foi avaliada após a criação das novas features, conforme ilustrado na Figura 6 do relatório. Histogramas e boxplots foram utilizados para visualizar o comportamento das features antes e depois da criação das novas variáveis (Figura 8).

## 3.ª FASE - MODEL PLANNING

Após a análise e tratamento dos dados, concluiu-se que o modelo mais apropriado para este problema é o **Random Forest**. Esta escolha foi motivada pelas seguintes razões:

- Presença de relações não lineares entre as diferentes features.
- Capacidade do Random Forest de computar a importância das features, o que se mostrou valioso.
- Robustez do modelo a outliers e ruído, que são características presentes no dataset.

A fase de model planning seguiu o ciclo de ciência de dados, abrangendo a limpeza, o pré-processamento e a criação de novas features relevantes. Testes de hipóteses foram realizados para avaliar a significância de cada classe.

## Resultado
Como resultado da nossa análise dos dados conseguimos criar um conjunto de dados que irão ser utilizados nas próximas fases para treinar os nossos modelos

Link: https://drive.google.com/file/d/1VdCwn-FpwIJ4Ox7nWTWjZCORiRnxpy9Q/view?usp=drive_link