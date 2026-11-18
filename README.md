# NAO Multi-YOLO Benchmark

Benchmark de 25 variantes das famílias YOLOv8, YOLOv9, YOLOv10, YOLOv11 e YOLOv12 usando imagens capturadas pela câmera do robô humanoide NAO e inferência executada em um computador externo.

O código realiza a captura por meio do NAOqi, transfere a imagem ao computador de processamento e executa os modelos YOLO **sequencialmente**. Não há execução paralela dos 25 modelos.

## Plataforma experimental

Configuração do computador informado para o experimento:

- CPU: Intel Core i9-12900
- RAM: 64 GB DDR4
- GPU: NVIDIA GeForce RTX 4070
- VRAM: 12 GB

O NAO é utilizado para aquisição das imagens. A inferência dos modelos é executada no computador externo.

O script também registra, em cada arquivo JSON gerado, informações que podem ser detectadas no ambiente de execução, como sistema operacional, versão do Python, versões das bibliotecas, disponibilidade/CUDA do PyTorch e GPUs detectadas. Essas informações são coletadas no momento da nova execução; elas não devem ser interpretadas como versões históricas do experimento quando não aparecem nos registros antigos.

## Modelos avaliados

O dicionário `YOLO_MODELS` em `main.py` contém 25 pesos:

- YOLOv8: n, s, m, l e x;
- YOLOv9: t, s, m, c e e;
- YOLOv10: n, s, m, l e x;
- YOLOv11: n, s, m, l e x;
- YOLOv12: n, s, m, l e x.

Os pesos são carregados pela biblioteca Ultralytics. Na primeira execução, os arquivos que ainda não estiverem disponíveis localmente poderão ser baixados automaticamente pela biblioteca.

## Protocolo implementado

As configurações relevantes estão no início de `main.py` e nas funções de captura e inferência:

- conexão NAOqi: IP padrão `172.15.1.29`, porta `9559` (ajuste o IP para a rede utilizada);
- câmera: índice `0`, resolução VGA (`640 × 480`) e espaço de cor RGB;
- taxa solicitada na assinatura da câmera: 5 FPS;
- intervalo entre iterações: 5 segundos;
- limiar de confiança: 0,5;
- execução dos modelos: sequencial, um modelo por vez;
- inferência por modelo: três chamadas consecutivas sobre o mesmo quadro;
- tempo reportado: média e desvio-padrão dos três tempos medidos;
- cálculo de FPS: inverso do tempo médio;
- métricas de detecção: extraídas da primeira saída das três chamadas.

O código não descarta separadamente a primeira chamada como aquecimento (*warm-up*). Portanto, os tempos devem ser interpretados conforme esse protocolo, sem presumir uma etapa de aquecimento não implementada.

## Métricas coletadas

Para cada modelo e iteração, o programa registra:

- tempo médio de inferência, desvio-padrão e FPS;
- quantidade de detecções e número de classes distintas;
- estatísticas dos escores de confiança: média, desvio-padrão, mediana, mínimo, máximo e quartis;
- área das caixas delimitadoras e contagem de objetos pequenos, médios e grandes;
- amostras de utilização de CPU, memória RAM e GPU.

A utilização de CPU e RAM é obtida por métricas do sistema operacional. A carga de GPU é consultada por meio do GPUtil. Portanto, esses percentuais são **medidas em nível de sistema**, não medições isoladas do processo Python ou do modelo.

Os escores de confiança são valores emitidos pelo detector. Sem imagens anotadas com *ground truth*, eles não representam acurácia, precisão, revocação ou mAP.

## Instalação

Instale as dependências Python:

```bash
pip install -r requirements.txt
```

O módulo `qi` é fornecido pelo SDK/ambiente NAOqi e deve ser instalado/configurado separadamente, conforme a versão do robô e do computador utilizado. Ele não é tratado como uma dependência comum do PyPI neste projeto.

Para inferência na GPU NVIDIA, instale uma versão do PyTorch com suporte CUDA compatível com o driver e o ambiente da máquina antes de executar o benchmark. O pacote Ultralytics utiliza o PyTorch instalado.

As dependências estão sem versões fixadas porque os registros históricos disponíveis no repositório não documentam integralmente as versões exatas utilizadas naquele experimento. As novas execuções passam a registrar as versões detectáveis no JSON de cada iteração.

## Configuração e execução

1. Verifique se o NAO está ligado e se o serviço NAOqi está acessível pela rede.
2. Confirme o IP e a porta em `connect_to_nao()`.
3. Instale o SDK NAOqi e as dependências Python.
4. Execute:

```bash
python main.py
```

O programa continua executando até ser interrompido com `Ctrl+C`. Se a captura principal e o método alternativo falharem, a iteração é descartada: o programa não substitui uma imagem real por uma imagem sintética.

## Arquivos gerados

Os arquivos são gravados em `nao_yolo_benchmark_results/`:

```text
nao_yolo_benchmark_results/
├── annotated_images/
│   └── imagens anotadas por modelo e iteração
├── figures/
│   └── performance_summary_*.png
├── benchmark_<iteracao>_<timestamp>.json
├── metrics_summary.csv
├── resources_summary.csv
├── class_performance.csv
└── final_report_<timestamp>.txt
```

Os CSVs são atualizados de forma acumulada durante a sessão. Cada JSON contém os resultados detalhados da iteração, metadados do ambiente e dados por modelo. O relatório final consolida estatísticas descritivas, ranking de FPS, escores médios de confiança e utilização média de recursos.

O arquivo `results.md` preserva o registro textual das quatro iterações documentadas no experimento original. Ele é um log histórico, não um arquivo gerado automaticamente pelo script.

## Testes de integridade do código

Para verificar a sintaxe e algumas condições estruturais do pipeline de resultados, execute:

```bash
python -m unittest discover -s tests -v
```

Esses testes não executam o NAO, não carregam os pesos YOLO e não repetem o benchmark.

## Limitações

- O benchmark avalia o desempenho de inferência e os escores emitidos pelos modelos no cenário capturado; não calcula métricas de detecção baseadas em *ground truth*.
- A utilização de recursos é monitorada em nível de sistema.
- O código executa os modelos sequencialmente.
- O script atual não implementa alternância automática entre cenários de iluminação, densidade ou distância.
- O loop principal é contínuo e deve ser interrompido pelo operador ou adaptado para um número fixo de iterações.

## Autor

Vitor Amadeu Souza

Projeto de pesquisa sobre benchmarking de modelos YOLO em um sistema robótico humanoide NAO.
