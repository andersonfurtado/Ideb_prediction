# Predição do Índice de Desenvolvimento da Educação Básica (Ideb) utilizando a Automação de Machine Larning do Azure

# Tabela de Conteúdo
<!--ts-->
- [Diagrama de Arquitetura](#ArchitectureDiagram)
- [[Configuração e instalação do projeto](#Project-Set-Up-and-Installation)
- [Dataset](#dataset)
  * [Visão geral](#overview)
  * [Tarefas](#task)
  * [Acesso](#access)
- [ML Automatizado](#automated-ml)
  * [Visão geral das configurações do AutoML](#overview-of-automl-settings)
  * [Resultados](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Melhor Modelo AutoML](#best-model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Síntese das definições dos hiperparâmetros](#overview-of-hyperparameter-tuning-settings)
  * [Resultados](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Melhor MOodelo](#best-model)
- [Implementação do Modelo](#model-deployment)
  * [Síntese da Implementação do Modelo](#overview-of-deployed-model)
  * [Endpoint](#endpoint)
  * [Endpoint Query](#endpoint-query)  
- [Sugestões de Melhoria](#suggestions-to-improve)
- [Referências](#references)
 
<!--te-->  

O projeto trata da previsão do Índice de Desenvolvimento da Educação Básica (Ideb) das escolas de ensino médio no Brasil. O Ideb é calculado a partir de dados sobre aprovação escolar, obtidos no Censo Escolar, e médias de desempenho no Sistema de Avaliação da Educação Básica (Saeb) no Brasil. O Ideb agrega ao foco pedagógico das avaliações em larga escala a possibilidade de resultados sintéticos, facilmente assimiláveis, e que permitem o estabelecimento de metas de qualidade educacional para os sistemas de ensino.

Neste projeto, considerei um problema de regressão, ou seja, um processo em que um modelo aprende a prever um valor contínuo de saída para um dado dado de entrada. Para este projeto, executei essencialmente dois processos:
<ol>
  <li>Primeiro, apliquei o AutoML, onde vários modelos são treinados para se ajustarem aos dados de treino. Depois escolhi e guardei o melhor modelo, ou seja, o modelo com a melhor pontuação.
  </li><br>
  <li> Em segundo lugar, utilizando o HyperDrive, ajustei os hiperparâmetros e apliquei o Random Forest Regressor, que consiste num meta-estimador que ajusta uma série de árvores de decisão de classificação em várias subamostras do conjunto de dados e utiliza o cálculo da média para melhorar a precisão da previsão e controlar o sobreajuste.
  </li><br>
</ol>
Modelos Hyperdrive and Automl foram treinados e implantados usando um ponto de extremidade no Azure Machine Learning.


A resolução desse problema se justifica porque indicadores educacionais como o Ideb são desejáveis por permitirem o monitoramento do sistema educacional do país. Sua importância, em termos de diagnóstico e orientação de ações políticas voltadas para a melhoria do sistema educacional, está em:
- detetar escolas e/ou redes de ensino cujos alunos apresentam baixo desempenho em termos de rendimento e proficiência;
- acompanhar a evolução temporal do desempenho dos alunos nessas escolas e/ou redes de ensino.

## Architecture Diagram

Contém o `Diagrama de Arquitetura` deste projeto:

![Architecture Diagram](./Images/ArchitectureDiagram.png)

## Project Set Up and Installation
Os arquivos necessários para executar este projeto são os seguintes:
- **automl.ipynb**: Jupyter Notebook para executar o experimento autoML
- **hyperparameter_tuning.ipynb**: Jupyter Notebook para realizar o experimento Hyperdrive
- **train.py**: Script utilizado no Hyperdrive
- **score.py**: Script utilizado para implantar o modelo
- **ideb_dataset.csv**: O conjunto de dados foi elaborado após a divulgação do Índice de Desenvolvimento da Educação Básica (Ideb) pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep).


## Dataset

### Overview
Os dados foram obtidos a partir do Censo Escolar 2019 e divulgados em 28 de setembro de 2023 pelo Inep, que pode ser consultado em<http://download.inep.gov.br/educacao_basica/portal_ideb/planilhas_para_download/2019/divulgacao_ensino_medio_municipios_2019.zip>.

### Task
Considerando que o Ideb foi desenvolvido para ser um indicador que sintetiza informações de desempenho em exames padronizados com informações de desempenho escolar (taxa média de aprovação dos alunos na etapa de ensino), as características utilizadas para predizer o Índice de Desenvolvimento da Educação Básica foram
aprovação dos alunos na etapa de ensino), as características utilizadas para predizer o Índice de Desenvolvimento da Educação Básica foram:
- **Total Approval Rate (2019)**: Na base de dados denominada "TAprov2019 Total"
- **1st Series Approval Rate (2019)**: Na base de dados denominada "TAprov2019_1_serie"
- **2nd Series Approval Rate (2019)**: Na base de dados denominada "TAprov2019_2_serie"
- **2nd Series Approval Rate (2019)**: Na base de dados denominada "TAprov2019_3_serie"
- **4th Grade Approval Rate (2019)**: Na base de dados denominada "TAprov2019_4_serie"
- **Approval_Indicator**: Na base de dados denominada "Indicador de Rendimento"
- **Grade SAEB Mathematics (2019)**: Na base de dados denominada "SAEB2019_Matematica"
- **Grade SAEB Language Portuguese (2019)**: Na base de dados denominada "SAEB2019_Lingua Portuguesa"
- **SAEB Standardized Average Score (N)**: Na base de dados denominada "SAEB2019_Nota Media Padronizada"

### Access
Inicialmente, explorei o repositório da base de dados do Ideb no site do Inep para obter um conjunto de dados para treinar os modelos. Assim que decidi trabalhar com o conjunto de dados do Ideb 2019 por ensino médio, encontrei o link para baixar os dados. O link foi então passado para o método from_delimited_files da classe Tabular objeto da classe Dataset objeto.

## Automated ML
Para configurar a execução do Automated ML, precisamos de especificar o tipo de tarefa com que estamos a lidar, a métrica primária, os conjuntos de dados de treino e validação (que estão na forma TabularDataset) e o nome da coluna de destino. A caraterização é definida como "auto", o que significa que a etapa de caraterização deve ser feita automaticamente. 

### Results
O melhor modelo geral foi o modelo `VotingEnsemble`, com um R2 Score de 0,99787. Os demais modelos ficaram entre 0,75571 e 0,99786, exceto o VotingEnsemble.

A execução do AutoML tem os seguintes parâmetros:

* *task*: Para ajudar a gerenciar as execuções secundárias e quando elas podem ser executadas, recomendamos que você crie um cluster dedicado por experimento. Neste projeto, `4` foi o número de execuções/iterações secundárias simultâneas. 
 
* *max_concurrent_iterations*: Esta é a métrica que é optimizada durante a formação do algoritmo do modelo. Por exemplo, a exatidão, a área sob a curva (AUC), a pontuação R2, etc. Neste projeto, o `R2 Score` foi utilizado como uma métrica de sucesso.
* *primary_metric*: Esta é a métrica que é optimizada durante o treino do algoritmo do modelo. Por exemplo, a exatidão, a área sob a curva (AUC), a pontuação R2, etc. Neste projeto, o `R2 Score` foi utilizado como uma métrica de sucesso.

* *experiment_timeout_minutes*: Este é o tempo máximo durante o qual o AutoML pode utilizar diferentes modelos para treinar no conjunto de dados. Neste projeto, o tempo máximo de espera em horas foi `0.5`.

* *training_data*: Estes são os dados de treinamento sobre os quais todos os diferentes modelos são treinados. Neste projeto, foi o `ideb_dataset`.

* *label_column_name*: Esta é a variável que é prevista pelo modelo. Neste projeto, foi o `IDEB_2019`.

* *n_cross_validations*: Este é o n do processo de validação cruzada. A validação cruzada é o processo em que diferentes pontos de dados são colocados no conjunto de dados de treino e de teste (resampling) para garantir que o modelo não se ajusta demasiado a determinados valores. Neste projeto, o n era `3`.

* *featurization*: Isto ajuda certos algoritmos que são sensíveis a características em diferentes escalas. Por exemplo, pode ativar mais caraterização, como imputação de valores em falta, codificação e transformações. Neste projeto, a caraterização foi `auto` (especifica que, como parte do pré-processamento, os guardrails de dados e os passos de caraterização devem ser feitos automaticamente). 

Contém o widget `RunDetails` da implementação do bloco de notas Jupyter no caso da execução do AutoML:

![automl_rundetails](./Images/automl_rundetails.png)

As seguintes capturas de ecrã mostram a iteração

![automl_iteration](./Images/automl_iteration.png)

Os seguintes screenshots mostram o melhor ID de execução:

![automl_bestmodel](./Images/automl_bestmodel.png)

Para uma visão geral completa do código, remeto para o notebook jypter automl.ipynb.
Talvez possamos melhorar a pontuação do erro absoluto médio ao:
- Ativar o registo na aplicação Web implementada
- Converter o modelo para o formato ONNX
- Aumentar o número de iterações
- Avaliar outros modelos de regressão.

## Hyperparameter Tuning
O objetivo deste projeto é prever o Ideb por escola no nível médio brasileiro. De acordo com os dados, a variável alvo "IDEB_2019" é uma variável flutuante, ou seja, uma variável contínua que varia entre 1,0, 1,6, 2,5, 3,5, 7,5 etc., portanto, trata-se de um problema de regressão. Nesse caso, então, utiliza-se modelos de regressão como a regressão linear, a regressão de floresta aleatória ou qualquer outro modelo de regressão. Considerando que no modelo de classificação, eu preciso converter o recurso alvo - "IDEB_2019" em um recurso categórico com 1 ou 0, não sendo adequado para o propósito deste projeto, é evidente que este é um `problema de regressão`.

Uma vez que a tarefa era um problema de regressão, o modelo utilizou o `Random Forest Regressor`, uma vez que: (i) É um dos algoritmos de aprendizagem mais precisos disponíveis. Para muitos conjuntos de dados, produz um classificador altamente preciso; (ii) funciona eficientemente em grandes bases de dados; (iii) pode lidar com milhares de variáveis de entrada sem eliminação de variáveis; (iv) gera uma estimativa interna não enviesada do erro de generalização à medida que a construção da floresta progride; (v) tem um método eficaz para estimar dados em falta e mantém a precisão quando uma grande proporção dos dados está em falta.

Para a experiência de ajuste dos hiperparâmetros, efectuada através do HyperDrive, os tipos de parâmetros e os seus intervalos agrupam os seguintes parâmetros:

* *primary_metric_name*:  Esta é a métrica que é optimizada durante a formação do algoritmo do modelo. Por exemplo, a exatidão, a área sob a curva (AUC), etc. Neste projeto, a `R2 Score` é utilizada como uma métrica de sucesso.

* *primary_metric_goal*: Este é o parâmetro que diz ao Hyperdrive como otimizar o algoritmo utilizando o nome_primário_métrico fornecido. O objetivo pode ser qualquer coisa desde Maximizar até Minimizar o nome_da_métrica_primária. Neste projeto, é `PrimaryMetricGoal.MAXIMIZE`.

* *max_total_runs*: Este é o número máximo de execuções que o Hyperdrive executará usando diferentes hiperparâmetros. Neste projeto, o max_total_runs é `40`.

* *max_concurrent_runs*: Este é o número máximo de execuções que são executadas simultaneamente em diferentes threads. Neste projeto, o max_concurrent_runs é `8`.

* *hyperparameter_sampling*: ste é o Parameter Sampler que especifica as técnicas nas quais os hiperparâmetros são ajustados. Neste projeto, o RandomParameterSampling foi utilizado para afinar os hiperparâmetros '--max_depth' com `choice(range(1, 20)`, '--min_samples_split' com `choice(2, 5, 10, 15, 100)` e '--min_samples_leaf' com `choice(range(1, 10)`.

* *policy*: Esta é a política de parada antecipada usada pelo Hyperdrive, que é usada para fornecer orientação sobre quantas iterações podem ser executadas antes que o modelo comece a se sobreajustar. Neste projeto, BanditPolicy foi utilizada com o argumento evaluation_interval de `2` e slack_factor de `0.1`. BanditPolciy termina qualquer execução cuja métrica primária seja menor que o fator de folga da melhor execução.

* *RunConfig*: Existem vários métodos para configurar um trabalho de treino no Azure Machine Learning através do SDK. Este é o script de treino que será executado com os hiperparâmetros de amostra. Por exemplo, Estimators, ScriptRunConfig e o RunConfiguration de nível inferior. Neste projeto, o `ScriptRunConfig` (define os recursos por trabalho (um ou vários nós) e o destino de computação a ser usado).

Contém a implementação RunDetails no caso da execução Hyperdrive.

### Results

O melhor modelo da execução do HyperDrive foi o VotingEnsemble com `0.99787 R2 Score`. Este contém a implementação `RunDetails` do melhor modelo treinado com seus parâmetros:
![hyperdrive_rundetails](./Images/hyperdrive_rundetails.png)

Isto mostra que o melhor modelo gerado pelo Hyperdrive com o seu ID de execução:
![hyperdrive_bestrun](./Images/hyperdrive_bestrun.png)

Algumas ideias sobre como talvez melhorar o melhor modelo de HyperDrive:
- Testar diferentes métodos de amostragem
- Especificar diferentes tipos de distribuições de hiperparâmetros
- Explorar outra gama de valores definidos para cada hiperparâmetro
- Explorar outra política de terminação antecipada
- Aumentar o número de iterações

Para uma visão geral completa do código, consulte o notebook jypter automl.ipynb.

## Model Deployment
Resumindo os resultados, o melhor modelo gerado pelo AutoML foi 38 modelos dentre os quais o `VotingEnsemblehad` teve o melhor desempenho com 0.99787 R2 Score. Por outro lado, o Hyperdrive gerou 40 modelos e o melhor modelo ao executar o HyperDrive foi com 0.9978595055572106 R2 Score. Então, eu implementei o melhor modelo - que por acaso era o modelo VotingEnsemble para executar o AutoML, como um endpoint.

Os modelos `103` gerados pelo AutoML, dentre os quais o `VotingEnsemble` teve o melhor desempenho com 0.99787 R2 Score. Por outro lado, o Hyperdrive gerou 48 iterações com o Modelo de Regressão Logística com diferentes ajustes dos hiperparâmetros min_samples_leaf, min_samples_split e max_depth e obteve um R2 Score de 0,9978595055572106 com min_samples_leaf como 3, min_samples_split como 5 e max_depth como 15. Portanto, o modelo Hyperdrive foi implantado.

Este contém o `Endpont` ativo:

![endpoint](./Images/endpoint.png)

O melhor modelo é implementado seguindo estes passos:

* *Register the model*: Para além do conteúdo do próprio ficheiro de modelo, o seu modelo registado também armazena metadados do modelo - descrição do modelo, etiquetas e informações da estrutura - que serão úteis ao gerir e implementar modelos no seu espaço de trabalho;
* *Prepare an inference configuration*: Uma configuração de inferência descreve como configurar o serviço Web que contém o seu modelo. É utilizada mais tarde, quando o modelo é implementado;
* *Prepare an entry script (used score.py)*: O script de entrada recebe os dados submetidos a um serviço Web implementado e passa-os para o modelo. Em seguida, pega na resposta devolvida pelo modelo e devolve-a ao cliente. O entry_script utilizado como entrada para InferenceConfig foi "score.py", que é o melhor modelo de trabalho gerado a partir do Hyperdrive;
* *Choose a compute target*: The compute target que utilizar para alojar o seu modelo afectará o custo e a disponibilidade do seu ponto final implementado. O destino de computação escolhido foi uma Instância de Contêiner do Azure que incluía o script de pontuação;
* *Deploy the model to the compute target*: Os serviços Web pegam num ou mais modelos, carregam-nos num ambiente e executam-nos num dos vários alvos de implementação suportados; 
* *Test the resulting web service*: Após a implantação bem-sucedida, foi gerado um endpoing REST com um url de pontuação para ser usado para previsões, conforme mostrado abaixo: ![REST_endpoint](./Images/REST_endpoint.png).

Depois de implementar o modelo como um serviço Web, foi criado um ponto de extremidade da API REST. Pode enviar dados para este ponto final e receber a previsão devolvida pelo modelo. Este exemplo demonstra como utilizei o Python para chamar o serviço Web criado:


```
import requests
import json

# scoring endpoint
scoring_uri = service.scoring_uri


# If the service is authenticated, set the key or token
#key = '<your key or token>'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
           "TAprov2019_Total": 99.9, 
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
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)
```
O resultado obtido é semelhante ao seguinte:

```
"{\"result\": [6.903481911249511]}"
```
Ou seja, uma escola de ensino médio com as características descritas em `data {}` tem o resultado do Ideb 6.903481911249511.

No final, eu excluo a implantação da ACI, bem como o cluster de computação.

## Suggestions to Improve
Algumas áreas de melhoria para experiências futuras são

- Testar diferentes métodos de amostragem
- Especificar diferentes tipos de distribuições de hiperparâmetros
- Ativar o registo na aplicação Web implementada
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

## Reference
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
