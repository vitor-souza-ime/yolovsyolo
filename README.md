# NAO Multi-YOLO Benchmark

Este projeto implementa um **benchmark avançado de diferentes versões do YOLO (v8, v9, v10, v11 e v12)** no robô humanoide **NAO**, com captura de imagens em tempo real, análise estatística e geração de relatórios visuais.

## 🚀 Funcionalidades
- Conexão direta com o robô **NAO** via NAOqi.
- Captura de imagens da câmera frontal do NAO, com fallback robusto.
- Execução paralela de múltiplos modelos YOLO (`YOLOv8` até `YOLOv12`).
- Coleta de métricas de performance:
  - Tempo médio de processamento e FPS.
  - Quantidade e classes únicas de objetos detectados.
  - Estatísticas de confiança (média, desvio padrão, quartis).
  - Distribuição de tamanhos dos objetos.
- Monitoramento de recursos do sistema (**CPU, RAM e GPU**).
- Análise detalhada por classe (COCO dataset).
- Geração de gráficos avançados com **Seaborn/Matplotlib**.

## 📊 Exemplos de gráficos gerados
- Distribuição de FPS por modelo.
- Correlação entre métricas (heatmap).
- Evolução do FPS ao longo das iterações.
- Comparação de uso de CPU/GPU/Memory.
- Heatmap de desempenho por classe.
- Distribuição de tamanhos de objetos detectados.

## 📦 Dependências
Instale os pacotes necessários com:
```bash
pip install ultralytics opencv-python pillow numpy pandas matplotlib seaborn psutil gputil scipy naoqi
````

## 🤖 Configuração do NAO

1. Certifique-se de que o **NAOqi** está rodando no robô.
2. Ajuste o IP e a porta no arquivo principal:

   ```python
   connect_to_nao(ip="172.15.1.29", port=9559)
   ```

## ▶️ Execução

1. Conecte-se ao NAO.
2. Configure os modelos YOLO disponíveis no dicionário `YOLO_MODELS`.
3. Execute o script principal:

   ```bash
   python nao_yolo_benchmark.py
   ```
4. Ao final, os relatórios e gráficos serão salvos em:

   ```
   nao_yolo_benchmark_results/
   ```

## 📁 Estrutura de Saída

```
nao_yolo_benchmark_results/
│── comprehensive_analysis_YYYYMMDD_HHMMSS.png   # Relatório visual completo
│── metrics.json                                 # Métricas coletadas
│── annotated_images/                            # Imagens anotadas por modelo
```

## 📖 Cenários de Teste

O benchmark permite avaliar robustez em diferentes condições:

* **normal**: iluminação normal
* **low\_light**: pouca luz
* **high\_contrast**: contraste alto
* **crowded**: muitos objetos
* **distant**: objetos distantes

## 🧑‍💻 Autor

Projeto desenvolvido por **Vitor Amadeu Souza** no contexto de pesquisa sobre **benchmarking de modelos YOLO aplicados à robótica humanoide**.
