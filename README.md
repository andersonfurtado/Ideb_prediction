# Predição do Índice de Desenvolvimento da Educação Básica (Ideb) usando o Azure Machine Learning

# Índice
<!--ts-->
- [Diagrama de Arquitetura](#ArchitectureDiagram)
- [Configuração e instalação do projeto](#Configuração-e-instalação-do-projeto)
- [Conjunto de dados](#conjunto de dados)
  * [Visão geral](#visão geral)
  * [Tarefa](#tarefa)
  * [Acesso](#acesso)
- [ML automatizado](#automated-ml)
  * [Visão geral das definições de AutoML](#visão geral das definições de autoML)
  * [Resultados](#resultados)
  * Widget [RunDetails](#rundetails-widget)
  * [Melhor modelo](#melhor-modelo)
- [Ajuste de hiperparâmetro](#ajuste-de-hiperparâmetro)
  * Visão geral das definições de afinação de hiperparâmetros](#visão geral das definições de afinação de hiperparâmetros)
  * [Resultados](#resultados)
  * Widget [RunDetails](#rundetails-widget)
  * [Melhor modelo](#melhor-modelo)
- [Implementação do modelo](#model-deployment)
  * Visão geral do modelo implementado](#visão geral do modelo implementado)
  * [Ponto final](#pontofinal)
  * [Consulta do ponto final](#endpoint-query)  
- [Sugestões para melhorar](#sugestões-para-melhorar)
- [Referências](#referências)

<!--te-->

O projeto trata da previsão do Índice de Desenvolvimento da Educação Básica (Ideb) das escolas de ensino médio no Brasil. O Ideb é calculado a partir de dados sobre aprovação escolar, obtidos no Censo Escolar, e médias de desempenho no Sistema de Avaliação da Educação Básica (Saeb) no Brasil. O Ideb agrega ao foco pedagógico das avaliações em larga escala a possibilidade de resultados sintéticos, facilmente assimiláveis, e que permitem o estabelecimento de metas de qualidade educacional para os sistemas de ensino.

Neste projeto, considerei um problema de regressão, ou seja, um processo em que um modelo aprende a prever um valor contínuo de saída para um dado dado de entrada. Para este projeto, executei essencialmente dois processos:

<ol>
  <li>First I applied AutoML where several models are trained to fit the training data. Then I chose and saved the best model, that is, the model with the best score.
  </li><br>
  <li> Second, using HyperDrive, I adjusted the hyperparameters and applied the Random Forest Regressor which consists of a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  </li><br>
</ol>
Os modelos Hyperdrive e Automl foram treinados e implantados usando um ponto de extremidade no Azure Machine Learning.

A resolução deste problema se justifica porque indicadores educacionais como o Ideb são desejáveis por permitirem o monitoramento do sistema educacional do país. Sua importância, em termos de diagnóstico e orientação de ações políticas voltadas para a melhoria do sistema educacional, está em:
- detetar escolas e/ou redes de ensino cujos alunos apresentam baixo desempenho em termos de rendimento e proficiência;
- acompanhar a evolução temporal do desempenho dos alunos nessas escolas e/ou redes de ensino.

## Diagrama de Arquitetura

Contém o `Diagrama de Arquitetura` deste projeto:

![Architecture Diagram](./Images/ArchitectureDiagram.png)

## Configuração e instalação do projeto
Os ficheiros de arranque necessários para executar este projeto são os seguintes:
- **automl.ipynb**: Jupyter Notebook para executar o experimento autoML
- **hyperparameter_tuning.ipynb**: Jupyter Notebook para executar a experiência Hyperdrive
- **train.py**: Script utilizado no Hyperdrive
- **score.py**: Script usado para implantar o modelo
- **ideb_dataset.csv**: O conjunto de dados elaborado após a divulgação do Índice de Desenvolvimento da Educação Básica (Ideb) pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep).


## Conjunto de dados

### Visão Geral
Os dados foram obtidos a partir do Censo Escolar 2019 e divulgados em 28 de setembro de 2023 pelo Inep, que podem ser encontrados em <http://download.inep.gov.br/educacao_basica/portal_ideb/planilhas_para_download/2019/divulgacao_ensino_medio_municipios_2019.zip>.
<https://download.inep.gov.br/educacao_basica/portal_ideb/planilhas_para_download/2021/divulgacao_ensino_medio_municipios_2021.zip>.

### Tarefa
Considerando que o Ideb foi desenvolvido para ser um indicador que sintetiza informações de desempenho em exames padronizados com informações de desempenho escolar (taxa média
de aprovação dos alunos na etapa de ensino), as características utilizadas para predizer o Índice de Desenvolvimento da Educação Básica foram:
- **Taxa de Aprovação Total (2019)**: Na base de dados denominada "TAprov2019 Total"
- **Taxa de Aprovação da 1ª Série (2019)**: Na base de dados denominada "TAprov2019_1_serie"
- **Taxa de aprovação da 2ª série (2019)**: Na base de dados denominada "TAprov2019_2_serie"
- **Taxa de Aprovação da 2ª Série (2019)**: Na base de dados denominada "TAprov2019_3_serie"
- **Taxa de aprovação da 4ª série (2019)**: Na base de dados denominada "TAprov2019_4_serie"
- **Indicador_de_Aprovação**: Na base de dados denominada "Indicador de Rendimento"
- **Grade SAEB Matemática (2019)**: Na base de dados denominada "SAEB2019_Matematica"
- **Grade SAEB Língua Portuguesa (2019)**: Na base de dados denominada "SAEB2019_Lingua Portuguesa"
- **Score Médio Padronizado do SAEB (N)**: Na base de dados denominada "SAEB2019_Nota Média Padronizada"

### Acesso
Inicialmente, explorei o repositório de bases de dados do Ideb no site do Inep para obter um conjunto de dados para treinar os modelos. Assim que decidi trabalhar com o conjunto de dados do Ideb 2019 por ensino médio, encontrei o link para baixar os dados. O link foi então passado para o método from_delimited_files da classe Tabular do objeto da classe Dataset.

## ML automatizado
Para configurar a execução do ML automatizado, precisamos especificar com que tipo de tarefa estamos lidando, a métrica primária, os conjuntos de dados de treinamento e validação (que estão no formato TabularDataset) e o nome da coluna de destino. A caraterização é definida como "auto", o que significa que a etapa de caraterização deve ser feita automaticamente. 

### Resultados
O melhor modelo geral foi o modelo `VotingEnsemble`, com um R2 Score de 0,99787. O restante dos modelos estava entre 0,75571 e 0,99786, exceto o VotingEnsemble.

A execução do AutoML tem os seguintes parâmetros:

* *tarefa*: Para ajudar a gerenciar as execuções secundárias e quando elas podem ser executadas, recomendamos a criação de um cluster dedicado por experimento. Neste projeto, `4` foi o número de execuções/iterações secundárias simultâneas. 

* *max_concurrent_iterations*: Esta é a métrica que é optimizada durante o treino do algoritmo do modelo. Por exemplo, Precisão, Área sob a curva (AUC), Pontuação R2, etc. Neste projeto, o `R2 Score` foi utilizado como uma métrica de sucesso.
* *primary_metric*: Esta é a métrica que é optimizada durante o treino do algoritmo do modelo. Por exemplo, a exatidão, a área sob a curva (AUC), a pontuação R2, etc. Neste projeto, o `R2 Score` foi utilizado como uma métrica de sucesso.

* *experiment_timeout_minutes*: Este é o tempo máximo durante o qual o AutoML pode utilizar diferentes modelos para treinar no conjunto de dados. Neste projeto, o tempo máximo de expiração em horas foi `0.5`.

* *training_data*: Este é o dado de treinamento sobre o qual todos os diferentes modelos são treinados. Neste projeto, foi o `ideb_dataset`.

* *label_column_name*: Esta é a variável que é prevista pelo modelo. Neste projeto, foi o `IDEB_2019`.

*n_cross_validations*: Este é o n do processo de validação cruzada. A validação cruzada é o processo em que diferentes pontos de dados são colocados no conjunto de dados de treino e de teste (reamostragem) para garantir que o modelo não se ajusta demasiado a determinados valores. Neste projeto, o n foi `3`.

* *featurização*: Isto ajuda certos algoritmos que são sensíveis a características em diferentes escalas. Por exemplo, é possível ativar mais caraterização, como a imputação de valores em falta, codificação e transformações. Neste projeto, a caraterização foi `auto` (especifica que, como parte do pré-processamento, os guardrails de dados e os passos de caraterização devem ser feitos automaticamente). 

Contém o widget `RunDetails` da implementação do Jupyter notebook no caso da execução do AutoML:

![automl_rundetails](./Images/automl_rundetails.png)

As seguintes capturas de ecrã mostram a iteração

![automl_iteration](./Images/automl_iteration.png)

As imagens seguintes mostram a identificação da melhor execução:

![automl_bestmodel](./Images/automl_bestmodel.png)

Para uma visão geral completa do código, remeto para o notebook jypter automl.ipynb.
Talvez possamos melhorar a pontuação do erro médio absoluto
- Ativar o registo na aplicação Web implementada
- Converter o modelo para o formato ONNX
- Aumentar o número de iterações
- Avaliar outros modelos de regressão.

## Ajuste de hiperparâmetros
O objetivo deste projeto é prever o Ideb por escola no nível médio brasileiro. De acordo com os dados, a variável alvo "IDEB_2019" é uma variável flutuante, ou seja, uma variável contínua que varia entre 1,0, 1,6, 2,5, 3,5, 7,5 etc., portanto, trata-se de um problema de regressão. Nesse caso, então, utiliza-se modelos de regressão como a regressão linear, a regressão de floresta aleatória ou qualquer outro modelo de regressão. Considerando que no modelo de classificação, eu preciso converter o recurso alvo - "IDEB_2019" em um recurso categórico com 1 ou 0, não sendo adequado para o propósito deste projeto, é evidente que este é um `problema de regressão`.

Uma vez que a tarefa era um problema de regressão, o modelo utilizou o `Random Forest Regressor`, uma vez que: (i) É um dos algoritmos de aprendizagem mais precisos disponíveis. Para muitos conjuntos de dados, produz um classificador altamente preciso; (ii) funciona eficientemente em grandes bases de dados; (iii) pode lidar com milhares de variáveis de entrada sem eliminação de variáveis; (iv) gera uma estimativa interna não enviesada do erro de generalização à medida que a construção da floresta progride; (v) tem um método eficaz para estimar dados em falta e mantém a precisão quando uma grande proporção dos dados está em falta.

Para o experimento de ajuste de hiperparâmetros, feito via HyperDrive, os tipos de parâmetros e seus intervalos agrupam os seguintes parâmetros:

* *primary_metric_name*: Esta é a métrica que é optimizada durante o treino do algoritmo do modelo. Por exemplo, a exatidão, a área sob a curva (AUC), etc. Neste projeto, a `R2 Score` é utilizada como uma métrica de sucesso.

* *primary_metric_goal*: Este é o parâmetro que diz ao Hyperdrive como otimizar o algoritmo usando o nome da métrica primária fornecida. O objetivo pode ser qualquer coisa, desde Maximizar até Minimizar o nome_da_métrica_primária. Neste projeto, é `PrimaryMetricGoal.MAXIMIZE`.

* *max_total_runs*: Este é o número máximo de execuções que o Hyperdrive irá executar usando diferentes hiperparâmetros. Neste projeto, o max_total_runs é `40`.

* *max_concurrent_runs*: Este é o número máximo de execuções que são executadas simultaneamente em diferentes threads. Neste projeto, o max_concurrent_runs é `8`.

* *hyperparameter_sampling*: Este é o Parameter Sampler que especifica as técnicas nas quais os hiperparâmetros são ajustados. Neste projeto, RandomParameterSampling foi utilizado para ajustar os hiperparâmetros '--max_depth' com `choice(range(1, 20)`, '--min_samples_split' com `choice(2, 5, 10, 15, 100)` e '--min_samples_leaf' com `choice(range(1, 10)`.

* *política*: Esta é a política de parada antecipada usada pelo Hyperdrive, que é usada para fornecer orientação sobre quantas iterações podem ser executadas antes que o modelo comece a se ajustar demais. Neste projeto, BanditPolicy foi usada com o argumento evaluation_interval de `2` e slack_factor de `0.1`. BanditPolciy termina qualquer execução cuja métrica primária seja menor que o fator de folga da melhor execução.

* *RunConfig*: Há vários métodos para configurar um trabalho de treinamento no Azure Machine Learning por meio do SDK. Este é o script de treinamento que será executado com os hiperparâmetros de amostra. Por exemplo, Estimators, ScriptRunConfig e o RunConfiguration de nível inferior. Neste projeto, o `ScriptRunConfig` (define os recursos por trabalho (um ou vários nós) e o destino de computação a ser usado).

Ele contém a implementação RunDetails no caso da execução do Hyperdrive.

### Resultados

O melhor modelo da execução do HyperDrive foi o VotingEnsemble com `0,99787 R2 Score`. Aqui contém a implementação `RunDetails` do melhor modelo treinado com seus parâmetros:
![hyperdrive_rundetails](./Images/hyperdrive_rundetails.png)

Isto mostra que o melhor modelo gerado pelo Hyperdrive com o seu ID de execução:
![hyperdrive_bestrun](./Images/hyperdrive_bestrun.png)

Algumas ideias sobre como talvez melhorar o melhor modelo do HyperDrive:
- Testar diferentes métodos de amostragem
- Especificar diferentes tipos de distribuições de hiperparâmetros
- Explorar outra gama de valores definidos para cada hiperparâmetro
- Explorar outra política de terminação antecipada
- Aumentar o número de iterações

Para uma visão geral completa do código, consulte o notebook jupyter automl.ipynb.

## Implementação do modelo
Resumindo os resultados, o melhor modelo gerado pelo AutoML foi 38 modelos dentre os quais o `VotingEnsemblehad` teve o melhor desempenho com 0.99787 R2 Score. Por outro lado, o Hyperdrive gerou 40 modelos e o melhor modelo ao executar o HyperDrive foi com 0.9978595055572106 R2 Score. Então, eu implementei o melhor modelo - que por acaso era o modelo VotingEnsemble para executar o AutoML, como um endpoint.

Os modelos `103` gerados pelo AutoML, dentre os quais o `VotingEnsemble` teve o melhor desempenho com 0.99787 R2 Score. Por outro lado, o Hyperdrive gerou 48 iterações com o Modelo de Regressão Logística com diferentes ajustes dos hiperparâmetros min_samples_leaf, min_samples_split e max_depth e obteve um R2 Score de 0,9978595055572106 com min_samples_leaf como 3, min_samples_split como 5 e max_depth como 15. Portanto, o modelo AutoML foi implantado.

Ele contém o `Endpont` ativo:

![endpoint](./Images/endpoint.png)

O melhor modelo é implantado seguindo os seguintes passos:

* *Registrar o modelo*: Para além do conteúdo do próprio ficheiro do modelo, o seu modelo registado também armazenará metadados do modelo - descrição do modelo, etiquetas e informações da estrutura - que serão úteis ao gerir e implementar modelos no seu espaço de trabalho;
*Preparar uma configuração de inferência*: Uma configuração de inferência descreve como configurar o serviço Web que contém seu modelo. Ela é usada mais tarde, quando você implanta o modelo;
*Preparar um script de entrada (usado score.py)*: O script de entrada recebe dados enviados para um serviço Web implantado e os passa para o modelo. Em seguida, ele pega a resposta retornada pelo modelo e a retorna para o cliente. O script de entrada utilizado como entrada para InferenceConfig foi "score.py", que é o melhor modelo de trabalho gerado a partir do Hyperdrive;
*Escolha um alvo de computação*: O destino de computação que você usa para hospedar seu modelo afetará o custo e a disponibilidade do ponto de extremidade implantado. O destino de computação escolhido foi uma Instância de Contêiner do Azure que incluía o script de pontuação;
* Implantar o modelo no destino de computação: Os serviços Web utilizam um ou mais modelos, carregam-nos num ambiente e executam-nos num dos vários destinos de implementação suportados; 
*Teste o serviço Web resultante*: Após a implantação bem-sucedida, um endpoing REST com um url de pontuação foi gerado para ser usado para previsões, conforme mostrado abaixo:
![REST_endpoint](./Images/REST_endpoint.png)

Depois de implantar o modelo como um serviço Web, foi criado um ponto de extremidade da API REST. É possível enviar dados para este ponto final e receber a previsão devolvida pelo modelo. Este exemplo demonstra como usei o Python para chamar o serviço Web criado:

```
import requests
import json

# Ponto final de pontuação
scoring_uri = service.scoring_uri


# Se o serviço for autenticado, defina a chave ou o token
#chave = '<sua chave ou token>'

# Dois conjuntos de dados para pontuar, de modo a obtermos dois resultados
data = {"data":
        [
          {
           "TAprov2019_Total": 99,9, 
           "TAprov2019_1_serie": 99.2, 
           "TAprov2019_2_serie": 59.1, 
           "TAprov2019_3_serie": 60.5, 
           "TAprov2019_4_serie": 70.5, 
           "Indicador_Rendimento": 0.99, 
           "SAEB2019_Matematica": 365.38, 
           "SAEB2019_Lingua Portuguesa": 351.54, 
           "SAEB2019_Nota Media Padronizada": 7.055853
          },
      ]
    }
# Converter para string JSON
input_data = json.dumps(data)

# Definir o tipo de conteúdo
headers = {'Content-Type': 'application/json'}
# Se a autenticação estiver activada, definir o cabeçalho de autorização
#headers['Authorization'] = f'Bearer {key}'

# Efetuar o pedido e apresentar a resposta
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)
```
O resultado retornado é semelhante ao seguinte:

```
"{\"resultado\": [6.903481911249511]}"
```
Ou seja, uma escola de ensino médio com as características descritas em `data {}` tem o resultado de Ideb 6.903481911249511.

No final, eu excluo a implantação da ACI e também o cluster de computação.

## Sugestões para melhorar
Algumas áreas de melhoria para experiências futuras são:

- Testar diferentes métodos de amostragem
- Especificar diferentes tipos de distribuições de hiperparâmetros
- Habilitar o registro em log no aplicativo Web implantado
- Alterar o espaço de pesquisa
- Implantar o modelo no Edge usando o Azure IoT Edge
- Explorar outro intervalo de valores definidos para cada hiperparâmetro
- Explorar outra política de encerramento antecipado
- Usar mais dados é a maneira mais simples e melhor possível de evitar o ajuste excessivo
- Converter o modelo para o formato ONNX
- Aumentar o número de iterações
- Definir a caraterização como automática
- Utilizar a regressão baseada em redes neuronais

A implementação destas melhorias em trabalhos futuros permitirá melhorar a precisão do modelo e obter novos conhecimentos para a empresa. Além disso, é possível desenvolver pipelines que possibilitem a reutilização deste modelo e melhorias contínuas.

## Referência
<ol>
  <li>[Inep 2020] Nota Técnica cocepção IDEB. August 2020. https://download.inep.gov.br/educacao_basica/portal_ideb/o_que_e_o_ideb/Nota_Tecnica_n1_concepcaoIDEB.pdf 
  </li><br>
  <li>[Microsoft 2020] BanditPolicy class - Azure Machine Learning Python | Microsoft Docs. https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py
  </li><br>
  <li>[Microsoft 2020] MedianStoppingPolicy class - Azure Machine Learning Python | Microsoft Docs. https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.medianstoppingpolicy?view=azure-ml-py
  </li><br>
  <li>[Microsoft 2020] Tune hyperparameters for your model with Azure Machine Learning | Microsoft Docs. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-search-space
  </li><br>
  <li>[Microsoft 2020] RandomParameterSampling class - Azure Machine Learning Python | Microsoft Docs. https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py
  </li><br>
  <li>[Microsoft 2020] Hyperparameter tuning a model with Azure Machine Learning | Microsoft Docs. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?view=azure-ml-py
   </li><br>
  <li>[Microsoft 2020] Consume an Azure Machine Learning model deployed as a web service | Microsoft Docs. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?view=azure-ml-py&tabs=python#call-the-service-python
   </li><br>
  <li>[Microsoft 2020] What is automated machine learning (AutoML)? | Microsoft Docs. https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azure-ml-py
  </li><br>
  <li>[2007-2020, scikit-learn deveopers] sklearn.model_selection.train_test_split — scikit-learn 0.23.2 documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  </li><br>
  <li>[2007-2020, scikit-learn deveopers]sklearn.linear_model.RandomForestRegression — scikit-learn 0.23.2 documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
  </li><br>
  <li>[GitHub 2021] Training of Python scikit-learn models on Azure. https://github.com/microsoft/MLHyperparameterTuning
  </li><br>
<ol>
