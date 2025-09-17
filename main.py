def create_performance_plots():
    """Cria gráficos de análise de performance avançados"""
    if not METRICS_DATA:
        print("❌ Não há dados para plotar")
        return
    
    df = pd.DataFrame(METRICS_DATA)
    
    # Configuração dos plots
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # Layout dos subplots (4x3 grid)
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. FPS por modelo (Box plot para mostrar variabilidade)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df, x='model', y='fps', ax=ax1)
    ax1.set_title('FPS Distribution by Model', fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_ylabel('FPS')
    
    # 2. Tempo de processamento vs. Objetos detectados (com tamanho = confiança)
    ax2 = fig.add_subplot(gs[0, 1])
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax2.scatter(model_data['processing_time'], model_data['objects_count'], 
                   s=model_data['confidence_mean']*100, alpha=0.6, label=model)
    ax2.set_xlabel('Processing Time (seconds)')
    ax2.set_ylabel('Objects Detected')
    ax2.set_title('Speed vs Detection Rate\n(bubble size = avg confidence)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Distribuição de confiança por modelo
    ax3 = fig.add_subplot(gs[0, 2])
    df.boxplot(column='confidence_mean', by='model', ax=ax3)
    ax3.set_title('Confidence Distribution by Model')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Average Confidence')
    plt.suptitle('')  # Remove o título automático do pandas
    
    # 4. Uso de recursos - CPU
    ax4 = fig.add_subplot(gs[1, 0])
    if RESOURCE_DATA:
        resource_df = pd.DataFrame(RESOURCE_DATA)
        sns.barplot(data=resource_df, x='model', y='cpu_avg', ax=ax4)
        ax4.set_title('Average CPU Usage by Model')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.set_ylabel('CPU Usage (%)')
    
    # 5. Uso de recursos - Memória
    ax5 = fig.add_subplot(gs[1, 1])
    if RESOURCE_DATA:
        sns.barplot(data=resource_df, x='model', y='memory_avg', ax=ax5)
        ax5.set_title('Average Memory Usage by Model')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
        ax5.set_ylabel('Memory Usage (%)')
    
    # 6. Classes únicas detectadas por modelo
    ax6 = fig.add_subplot(gs[1, 2])
    avg_unique_classes = df.groupby('model')['unique_classes'].mean().sort_values(ascending=False)
    avg_unique_classes.plot(kind='bar', ax=ax6, color='lightcoral')
    ax6.set_title('Average Unique Classes Detected')
    ax6.set_ylabel('Number of Classes')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Análise temporal - FPS ao longo das iterações
    ax7 = fig.add_subplot(gs[2, 0])
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax7.plot(model_data['iteration'], model_data['fps'], 
                marker='o', alpha=0.7, label=model, linewidth=2)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('FPS')
    ax7.set_title('FPS Evolution Over Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Distribuição de tamanhos de objetos
    ax8 = fig.add_subplot(gs[2, 1])
    size_data = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        avg_small = model_data['small_objects'].mean()
        avg_medium = model_data['medium_objects'].mean()
        avg_large = model_data['large_objects'].mean()
        size_data.append([model, 'Small', avg_small])
        size_data.append([model, 'Medium', avg_medium])
        size_data.append([model, 'Large', avg_large])
    
    size_df = pd.DataFrame(size_data, columns=['Model', 'Size', 'Count'])
    sns.barplot(data=size_df, x='Model', y='Count', hue='Size', ax=ax8)
    ax8.set_title('Object Size Distribution by Model')
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45)
    ax8.set_ylabel('Average Object Count')
    
    # 9. Correlação entre métricas
    ax9 = fig.add_subplot(gs[2, 2])
    correlation_cols = ['fps', 'objects_count', 'confidence_mean', 'unique_classes']
    corr_matrix = df[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9)
    ax9.set_title('Metrics Correlation Matrix')
    
    # 10. Performance por classe (top classes)
    ax10 = fig.add_subplot(gs[3, :])
    if CLASS_PERFORMANCE_DATA:
        class_df = pd.DataFrame(CLASS_PERFORMANCE_DATA)
        # Agrupa por modelo e classe, calculando médias
        class_summary = class_df.groupby(['model', 'class']).agg({
            'count': 'mean',
            'avg_confidence': 'mean'
        }).reset_index()
        
        # Pega apenas as top classes
        top_classes = class_summary.groupby('class')['count'].sum().sort_values(ascending=False).head(6).index
        class_summary_filtered = class_summary[class_summary['class'].isin(top_classes)]
        
        # Pivota para criar heatmap
        pivot_data = class_summary_filtered.pivot(index='model', columns='class', values='count')
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='Blues', ax=ax10)
        ax10.set_title('Average Detection Count by Model and Class')
        ax10.set_xlabel('Object Class')
        ax10.set_ylabel('Model')
    
    plt.suptitle('YOLO Multi-Version Performance Analysis - Comprehensive Report', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Salva o gráfico
    plot_filename = os.path.join(OUTPUT_DIR, f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico abrangente salvo: {plot_filename}")
    
    # Cria gráfico adicional para análise estatística
    create_statistical#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import qi
import time
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import threading
import GPUtil
from collections import defaultdict, Counter
from scipy import stats

# YOLO (Ultralytics)
from ultralytics import YOLO

# ---------------------------
# Configurações
# ---------------------------
INTERVAL_SECONDS = 5  # intervalo entre capturas
OUTPUT_DIR = "nao_yolo_benchmark_results"
CONFIDENCE_THRESHOLD = 0.5

# Versões do YOLO para comparação

YOLO_MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv8l": "yolov8l.pt",
    "YOLOv8x": "yolov8x.pt",
    "YOLOv9t": "yolov9t.pt",
    "YOLOv9s": "yolov9s.pt",
    "YOLOv9m": "yolov9m.pt",
    "YOLOv9l": "yolov9l.pt",
    "YOLOv9x": "yolov9x.pt",
    "YOLOv10n": "yolov10n.pt",
    "YOLOv10s": "yolov10s.pt",
    "YOLOv10m": "yolov10m.pt",
    "YOLOv10l": "yolov10l.pt",
    "YOLOv10x": "yolov10x.pt",
    "YOLOv11n": "yolo11n.pt",
    "YOLOv11s": "yolo11s.pt",
    "YOLOv11m": "yolo11m.pt",
    "YOLOv11l": "yolo11l.pt",
    "YOLOv11x": "yolo11x.pt",
    "YOLOv12n": "yolo12n.pt",
    "YOLOv12s": "yolo12s.pt",
    "YOLOv12m": "yolo12m.pt",
    "YOLOv12l": "yolo12l.pt",
    "YOLOv12x": "yolo12x.pt"
}

# Métricas para análise avançada
METRICS_DATA = []
RESOURCE_DATA = []
CLASS_PERFORMANCE_DATA = []
SCENARIO_DATA = []

# Cenários de teste para análise de robustez
TEST_SCENARIOS = {
    "normal": "Iluminação normal",
    "low_light": "Pouca luz",
    "high_contrast": "Alto contraste",
    "crowded": "Muitos objetos",
    "distant": "Objetos distantes"
}

# Classes COCO mais comuns para análise detalhada
TARGET_CLASSES = ['person', 'chair', 'table', 'laptop', 'cell phone', 'book', 'bottle', 'cup']

# Monitor de recursos do sistema
class SystemMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        # Calcula estatísticas
        return {
            'cpu_avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_max': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_max': np.max(self.memory_usage) if self.memory_usage else 0,
            'gpu_avg': np.mean(self.gpu_usage) if self.gpu_usage else 0,
            'gpu_max': np.max(self.gpu_usage) if self.gpu_usage else 0
        }
    
    def _monitor_resources(self):
        while self.monitoring:
            # CPU usage
            self.cpu_usage.append(psutil.cpu_percent())
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)
            
            # GPU usage (se disponível)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_usage.append(gpus[0].load * 100)
                else:
                    self.gpu_usage.append(0)
            except:
                self.gpu_usage.append(0)
            
            time.sleep(0.1)  # Monitor a cada 100ms

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------------------------
# Funções utilitárias do NAO
# ---------------------------

def connect_to_nao(ip="172.15.1.29", port=9559):
    """Conecta ao robô NAO"""
    try:
        session = qi.Session()
        session.connect(f"tcp://{ip}:{port}")
        print(f"✅ Conectado ao NAO em {ip}:{port}")
        return session
    except Exception as e:
        print(f"❌ Erro ao conectar ao NAO: {e}")
        raise

def capture_image_from_nao(session):
    """Captura imagem da câmera do NAO com tratamento robusto de erros"""
    try:
        camera_service = session.service("ALVideoDevice")
        camera_id = 0  # Câmera frontal
        resolution = 2  # VGA (640x480)
        color_space = 11  # RGB
        fps = 5
        
        # Método mais robusto de subscrição
        try:
            # Primeiro tenta desinscrever qualquer cliente existente
            try:
                camera_service.unsubscribe("python_client")
            except:
                pass
            
            # Subscreve à câmera
            video_client = camera_service.subscribeCamera("python_client", camera_id, resolution, color_space, fps)
        except Exception as e:
            print(f"⚠️ Método subscribeCamera falhou: {e}")
            # Método alternativo
            try:
                video_client = camera_service.subscribe("python_client", resolution, color_space, fps)
            except Exception as e2:
                print(f"⚠️ Método subscribe também falhou: {e2}")
                raise Exception("Não foi possível subscrever à câmera")
        
        # Aguarda um pouco para estabilizar
        time.sleep(0.2)
        
        # Captura a imagem
        nao_image = camera_service.getImageRemote(video_client)
        
        if nao_image is None or len(nao_image) < 7:
            raise Exception("Dados de imagem inválidos")
        
        # Extrai dados da imagem
        width = nao_image[0]
        height = nao_image[1]
        channels = nao_image[2]
        image_data = nao_image[6]
        
        print(f"📷 Imagem capturada: {width}x{height}, {channels} canais")
        
        # Converte para array numpy
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        
        # Verifica se o tamanho está correto
        expected_size = width * height * channels
        if len(image_array) != expected_size:
            print(f"⚠️ Tamanho inesperado da imagem: {len(image_array)} vs {expected_size}")
            image_array = image_array[:expected_size]
        
        image_array = image_array.reshape((height, width, channels))
        
        # Converte para PIL Image
        image = Image.fromarray(image_array).convert("RGB")
        
        return image
        
    except Exception as e:
        print(f"❌ Erro na captura principal: {e}")
        print("🔄 Tentando método alternativo...")
        return capture_image_alternative(session)
    finally:
        # Limpa a subscrição
        try:
            if 'video_client' in locals():
                camera_service.unsubscribe(video_client)
        except:
            pass

def capture_image_alternative(session):
    """Método alternativo usando ALPhotoCapture"""
    try:
        print("📸 Usando método alternativo ALPhotoCapture...")
        photo_service = session.service("ALPhotoCapture")
        photo_service.setResolution(2)  # VGA
        photo_service.setPictureFormat("jpg")
        
        # Caminho temporário
        temp_path = "/tmp/nao_temp_image.jpg"
        
        # Remove arquivo anterior se existir
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Tira a foto
        photo_service.takePicture(temp_path)
        
        # Aguarda o arquivo ser criado
        time.sleep(0.5)
        
        if not os.path.exists(temp_path):
            raise Exception("Arquivo de imagem não foi criado")
        
        # Carrega a imagem
        image = Image.open(temp_path).convert("RGB")
        print("✅ Imagem capturada com método alternativo")
        
        return image
        
    except Exception as e:
        print(f"❌ Método alternativo também falhou: {e}")
        print("🔵 Usando imagem de fallback...")
        # Retorna uma imagem azul como fallback
        return Image.new('RGB', (640, 480), color='blue')

# ---------------------------
# Configuração e execução multi-YOLO
# ---------------------------

def setup_yolo_models():
    """Configura todos os modelos YOLO"""
    print("🔥 Configurando modelos YOLO...")
    
    models = {}
    for model_name, model_file in YOLO_MODELS.items():
        try:
            print(f"⏳ Carregando {model_name}...")
            model = YOLO(model_file)
            models[model_name] = model
            print(f"✅ {model_name} carregado com sucesso")
        except Exception as e:
            print(f"❌ Erro ao carregar {model_name}: {e}")
            models[model_name] = None
    
    # Remove modelos que falharam
    models = {k: v for k, v in models.items() if v is not None}
    print(f"✅ {len(models)} modelos YOLO carregados com sucesso")
    
    return models

def detect_objects_multi_yolo(models, image):
    """Detecta objetos usando múltiplas versões do YOLO com métricas avançadas"""
    results_dict = {}
    
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    print(f"🔍 Processando imagem com {len(models)} modelos YOLO...")
    
    for model_name, model in models.items():
        try:
            print(f"   🔍 {model_name}...")
            
            # Monitor de recursos
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
            # Medição precisa do tempo (múltiplas execuções para precisão)
            times = []
            results = None
            
            # Executa 3 vezes para média mais precisa do tempo
            for _ in range(3):
                start_time = time.perf_counter()
                current_results = model(image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                if results is None:  # Usa o primeiro resultado para análise
                    results = current_results
            
            # Para o monitor de recursos
            resource_stats = monitor.stop_monitoring()
            
            # Calcula estatísticas de tempo
            processing_time = np.mean(times)
            time_std = np.std(times)
            
            # Calcula FPS
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            objects = []
            class_counts = Counter()
            confidence_scores = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extrai informações da detecção
                        conf = float(box.conf.cpu().numpy()[0])
                        cls = int(box.cls.cpu().numpy()[0])
                        xyxy = box.xyxy.cpu().numpy()[0].tolist()
                        class_name = model.names[cls]
                        
                        objects.append({
                            "class": class_name,
                            "score": conf,
                            "box": xyxy
                        })
                        
                        class_counts[class_name] += 1
                        confidence_scores.append(conf)
            
            # Calcula métricas avançadas
            if confidence_scores:
                confidence_stats = {
                    'mean': np.mean(confidence_scores),
                    'std': np.std(confidence_scores),
                    'median': np.median(confidence_scores),
                    'min': np.min(confidence_scores),
                    'max': np.max(confidence_scores),
                    'q25': np.percentile(confidence_scores, 25),
                    'q75': np.percentile(confidence_scores, 75)
                }
            else:
                confidence_stats = {k: 0.0 for k in ['mean', 'std', 'median', 'min', 'max', 'q25', 'q75']}
            
            # Análise por classe (para classes-alvo)
            class_performance = {}
            for target_class in TARGET_CLASSES:
                class_detections = [obj for obj in objects if obj['class'] == target_class]
                if class_detections:
                    class_confidences = [obj['score'] for obj in class_detections]
                    class_performance[target_class] = {
                        'count': len(class_detections),
                        'avg_confidence': np.mean(class_confidences),
                        'max_confidence': np.max(class_confidences)
                    }
                else:
                    class_performance[target_class] = {
                        'count': 0,
                        'avg_confidence': 0.0,
                        'max_confidence': 0.0
                    }
            
            # Análise de distribuição de tamanhos dos objetos
            object_sizes = []
            for obj in objects:
                box = obj['box']
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height
                object_sizes.append(area)
            
            size_stats = {}
            if object_sizes:
                size_stats = {
                    'mean_area': np.mean(object_sizes),
                    'median_area': np.median(object_sizes),
                    'small_objects': sum(1 for size in object_sizes if size < 1000),  # < 1000 pixels²
                    'medium_objects': sum(1 for size in object_sizes if 1000 <= size < 10000),
                    'large_objects': sum(1 for size in object_sizes if size >= 10000)
                }
            else:
                size_stats = {
                    'mean_area': 0,
                    'median_area': 0,
                    'small_objects': 0,
                    'medium_objects': 0,
                    'large_objects': 0
                }
            
            results_dict[model_name] = {
                "objects": objects,
                "results": results,
                "processing_time": processing_time,
                "time_std": time_std,
                "fps": fps,
                "objects_count": len(objects),
                "confidence_stats": confidence_stats,
                "class_counts": dict(class_counts),
                "class_performance": class_performance,
                "size_stats": size_stats,
                "resource_usage": resource_stats,
                "unique_classes": len(class_counts)
            }
            
            print(f"      ⏱️ {processing_time:.3f}s (±{time_std:.3f}) - {fps:.1f} FPS - {len(objects)} objetos")
            
        except Exception as e:
            print(f"❌ Erro no modelo {model_name}: {e}")
            results_dict[model_name] = None
    
    return results_dict

def draw_multi_yolo_results(image, results_dict, iteration):
    """Desenha resultados de todos os modelos YOLO"""
    annotated_images = {}
    
    for model_name, result_data in results_dict.items():
        if result_data is None:
            continue
            
        try:
            # Converte PIL para numpy se necessário
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image.copy()
            
            # YOLO results já vem com método plot
            annotated_image = result_data["results"][0].plot()
            
            # Adiciona título
            cv2.putText(annotated_image, f"{model_name} (Iter {iteration})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Time: {result_data['processing_time']:.3f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(annotated_image, f"Objects: {result_data['objects_count']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            annotated_images[model_name] = annotated_image
            
        except Exception as e:
            print(f"❌ Erro ao desenhar resultados {model_name}: {e}")
            # Usa imagem original em caso de erro
            if isinstance(image, Image.Image):
                annotated_images[model_name] = np.array(image)
            else:
                annotated_images[model_name] = image
    
    return annotated_images

# ---------------------------
# Análise e visualização de métricas
# ---------------------------

def collect_metrics(results_dict, iteration, scenario="normal"):
    """Coleta métricas avançadas de todos os modelos para análise posterior"""
    global METRICS_DATA, RESOURCE_DATA, CLASS_PERFORMANCE_DATA
    
    timestamp = datetime.now().isoformat()
    
    for model_name, result_data in results_dict.items():
        if result_data is None:
            continue
        
        # Métricas principais
        metric_entry = {
            "iteration": iteration,
            "scenario": scenario,
            "model": model_name,
            "timestamp": timestamp,
            
            # Métricas de tempo e performance
            "processing_time": result_data["processing_time"],
            "time_std": result_data["time_std"],
            "fps": result_data["fps"],
            
            # Métricas de detecção
            "objects_count": result_data["objects_count"],
            "unique_classes": result_data["unique_classes"],
            
            # Métricas de confiança
            "confidence_mean": result_data["confidence_stats"]["mean"],
            "confidence_std": result_data["confidence_stats"]["std"],
            "confidence_median": result_data["confidence_stats"]["median"],
            "confidence_min": result_data["confidence_stats"]["min"],
            "confidence_max": result_data["confidence_stats"]["max"],
            "confidence_q25": result_data["confidence_stats"]["q25"],
            "confidence_q75": result_data["confidence_stats"]["q75"],
            
            # Métricas de tamanho de objetos
            "mean_object_area": result_data["size_stats"]["mean_area"],
            "median_object_area": result_data["size_stats"]["median_area"],
            "small_objects": result_data["size_stats"]["small_objects"],
            "medium_objects": result_data["size_stats"]["medium_objects"],
            "large_objects": result_data["size_stats"]["large_objects"],
        }
        
        METRICS_DATA.append(metric_entry)
        
        # Métricas de recursos do sistema
        resource_entry = {
            "iteration": iteration,
            "model": model_name,
            "timestamp": timestamp,
            "cpu_avg": result_data["resource_usage"]["cpu_avg"],
            "cpu_max": result_data["resource_usage"]["cpu_max"],
            "memory_avg": result_data["resource_usage"]["memory_avg"],
            "memory_max": result_data["resource_usage"]["memory_max"],
            "gpu_avg": result_data["resource_usage"]["gpu_avg"],
            "gpu_max": result_data["resource_usage"]["gpu_max"]
        }
        
        RESOURCE_DATA.append(resource_entry)
        
        # Métricas por classe
        for class_name, class_data in result_data["class_performance"].items():
            class_entry = {
                "iteration": iteration,
                "model": model_name,
                "class": class_name,
                "timestamp": timestamp,
                "count": class_data["count"],
                "avg_confidence": class_data["avg_confidence"],
                "max_confidence": class_data["max_confidence"]
            }
            
            CLASS_PERFORMANCE_DATA.append(class_entry)

def print_benchmark_results(results_dict, iteration):
    """Imprime resultados comparativos avançados no console"""
    print(f"\n{'='*100}")
    print(f"📊 YOLO MULTI-VERSION BENCHMARK - Iteração {iteration}")
    print(f"{'='*100}")
    
    # Ordena por tempo de processamento
    sorted_results = sorted(
        [(name, data) for name, data in results_dict.items() if data is not None],
        key=lambda x: x[1]["processing_time"]
    )
    
    print(f"\n🏃 RANKING POR VELOCIDADE (FPS):")
    for i, (model_name, data) in enumerate(sorted_results, 1):
        print(f"   {i}º {model_name}: {data['fps']:.1f} FPS ({data['processing_time']:.3f}s ±{data['time_std']:.3f})")
    
    # Ranking por detecções
    sorted_by_objects = sorted(sorted_results, key=lambda x: x[1]["objects_count"], reverse=True)
    
    print(f"\n🎯 RANKING POR DETECÇÕES:")
    for i, (model_name, data) in enumerate(sorted_by_objects, 1):
        unique_classes = data['unique_classes']
        print(f"   {i}º {model_name}: {data['objects_count']} objetos ({unique_classes} classes únicas)")
    
    # Ranking por confiança
    sorted_by_confidence = sorted(sorted_results, key=lambda x: x[1]["confidence_stats"]["mean"], reverse=True)
    
    print(f"\n🏆 RANKING POR CONFIANÇA MÉDIA:")
    for i, (model_name, data) in enumerate(sorted_by_confidence, 1):
        conf_stats = data['confidence_stats']
        print(f"   {i}º {model_name}: {conf_stats['mean']:.3f} (±{conf_stats['std']:.3f}, max: {conf_stats['max']:.3f})")
    
    # Ranking por uso de recursos
    sorted_by_efficiency = sorted(sorted_results, 
                                key=lambda x: x[1]["resource_usage"]["cpu_avg"] + x[1]["resource_usage"]["memory_avg"])
    
    print(f"\n⚡ RANKING POR EFICIÊNCIA (menor uso de recursos):")
    for i, (model_name, data) in enumerate(sorted_by_efficiency, 1):
        resources = data['resource_usage']
        print(f"   {i}º {model_name}: CPU {resources['cpu_avg']:.1f}% | RAM {resources['memory_avg']:.1f}% | GPU {resources['gpu_avg']:.1f}%")
    
    # Análise por classe (top 3 classes mais detectadas)
    print(f"\n📋 ANÁLISE POR CLASSE (Top 3 mais detectadas):")
    for model_name, data in sorted_results[:3]:  # Top 3 modelos mais rápidos
        print(f"\n   {model_name}:")
        class_counts = data.get('class_counts', {})
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for class_name, count in top_classes:
            class_perf = data['class_performance'].get(class_name, {})
            avg_conf = class_perf.get('avg_confidence', 0)
            print(f"     • {class_name}: {count}x (conf. média: {avg_conf:.3f})")
    
    # Análise de distribuição de tamanhos
    print(f"\n📏 ANÁLISE DE TAMANHO DOS OBJETOS:")
    for model_name, data in sorted_results[:3]:
        size_stats = data['size_stats']
        print(f"   {model_name}: P={size_stats['small_objects']} | M={size_stats['medium_objects']} | G={size_stats['large_objects']}")
        print(f"     (área média: {size_stats['mean_area']:.0f} pixels²)")

def create_performance_plots():
    """Cria gráficos de análise de performance"""
    if not METRICS_DATA:
        print("❌ Não há dados para plotar")
        return
    
    df = pd.DataFrame(METRICS_DATA)
    
    # Configuração dos plots
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLO Multi-Version Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Tempo de processamento médio por modelo
    avg_time = df.groupby('model')['processing_time'].mean().sort_values()
    avg_time.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Average Processing Time by Model')
    axes[0,0].set_ylabel('Time (seconds)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Número médio de objetos detectados por modelo
    avg_objects = df.groupby('model')['objects_count'].mean().sort_values(ascending=False)
    avg_objects.plot(kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('Average Objects Detected by Model')
    axes[0,1].set_ylabel('Number of Objects')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Confiança média por modelo
    avg_confidence = df.groupby('model')['avg_confidence'].mean().sort_values(ascending=False)
    avg_confidence.plot(kind='bar', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Average Confidence by Model')
    axes[1,0].set_ylabel('Confidence Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot: Tempo vs Objetos detectados
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        axes[1,1].scatter(model_data['processing_time'], model_data['objects_count'], 
                         label=model, alpha=0.7)
    
    axes[1,1].set_xlabel('Processing Time (seconds)')
    axes[1,1].set_ylabel('Objects Detected')
    axes[1,1].set_title('Processing Time vs Objects Detected')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Salva o gráfico
    plot_filename = os.path.join(OUTPUT_DIR, f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {plot_filename}")
    
    # Mostra o gráfico
    plt.show()

def save_benchmark_results(annotated_images, results_dict, iteration):
    """Salva todos os resultados do benchmark"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar imagens anotadas
        for model_name, image in annotated_images.items():
            filename = os.path.join(OUTPUT_DIR, f"{model_name}_{iteration}_{timestamp}.jpg")
            cv2.imwrite(filename, image)
        
        # Salvar dados JSON detalhados
        json_filename = os.path.join(OUTPUT_DIR, f"benchmark_{iteration}_{timestamp}.json")
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "models": {}
        }
        
        for model_name, data in results_dict.items():
            if data is not None:
                benchmark_data["models"][model_name] = {
                    "processing_time": data["processing_time"],
                    "objects_count": data["objects_count"],
                    "max_confidence": data["max_confidence"],
                    "avg_confidence": data["avg_confidence"],
                    "objects": data["objects"]
                }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        # Salvar métricas em CSV
        if METRICS_DATA:
            csv_filename = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
            df = pd.DataFrame(METRICS_DATA)
            df.to_csv(csv_filename, index=False)
            print(f"📈 Métricas CSV: {csv_filename}")
        
        print(f"💾 Arquivos da iteração {iteration} salvos:")
        print(f"   📸 {len(annotated_images)} imagens anotadas")
        print(f"   📊 Dados: {json_filename}")
        
        return benchmark_data
        
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
        return None

def generate_final_report():
    """Gera relatório final com estatísticas consolidadas"""
    if not METRICS_DATA:
        return
    
    df = pd.DataFrame(METRICS_DATA)
    
    # Estatísticas por modelo
    stats = df.groupby('model').agg({
        'processing_time': ['mean', 'std', 'min', 'max'],
        'objects_count': ['mean', 'std', 'min', 'max'],
        'avg_confidence': ['mean', 'std', 'min', 'max'],
        'max_confidence': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Salva relatório
    report_filename = os.path.join(OUTPUT_DIR, f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLO MULTI-VERSION BENCHMARK - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total iterations: {df['iteration'].max()}\n")
        f.write(f"Models tested: {', '.join(df['model'].unique())}\n")
        f.write(f"Analysis period: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")
        
        f.write("PERFORMANCE STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(str(stats))
        f.write("\n\n")
        
        # Rankings
        f.write("FINAL RANKINGS:\n")
        f.write("-" * 40 + "\n")
        
        f.write("\n1. FASTEST MODELS (avg processing time):\n")
        fastest = df.groupby('model')['processing_time'].mean().sort_values()
        for i, (model, time_avg) in enumerate(fastest.items(), 1):
            f.write(f"   {i}º {model}: {time_avg:.3f}s\n")
        
        f.write("\n2. BEST DETECTION RATE (avg objects detected):\n")
        best_detection = df.groupby('model')['objects_count'].mean().sort_values(ascending=False)
        for i, (model, count_avg) in enumerate(best_detection.items(), 1):
            f.write(f"   {i}º {model}: {count_avg:.1f} objects\n")
        
        f.write("\n3. HIGHEST CONFIDENCE (avg confidence):\n")
        best_confidence = df.groupby('model')['avg_confidence'].mean().sort_values(ascending=False)
        for i, (model, conf_avg) in enumerate(best_confidence.items(), 1):
            f.write(f"   {i}º {model}: {conf_avg:.3f}\n")
    
    print(f"📋 Relatório final salvo: {report_filename}")

# ---------------------------
# Main loop
# ---------------------------

def main():
    print("=" * 80)
    print("🔥 MULTI-VERSION YOLO BENCHMARK - NAO ROBOT")
    print("🎯 Comparing YOLOv8, v9, v10, v11, v12")
    print("=" * 80)
    
    # Conecta ao NAO
    try:
        session = connect_to_nao()
    except Exception as e:
        print(f"❌ Falha na conexão com o NAO: {e}")
        print("💡 Verifique se o NAO está ligado e conectado na rede")
        return
    
    # Configura os modelos YOLO
    try:
        yolo_models = setup_yolo_models()
        if not yolo_models:
            print("❌ Nenhum modelo YOLO foi carregado com sucesso")
            return
    except Exception as e:
        print(f"❌ Falha na configuração dos modelos: {e}")
        return
    
    print(f"\n🚀 Iniciando benchmark com {len(yolo_models)} modelos")
    print(f"📊 Modelos: {', '.join(yolo_models.keys())}")
    print(f"⏱️ Intervalo entre capturas: {INTERVAL_SECONDS}s")
    print(f"🎯 Limiar de confiança: {CONFIDENCE_THRESHOLD}")
    print("Pressione Ctrl+C para parar e gerar relatório final\n")
    
    iteration = 1
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"📸 BENCHMARK - Iteração {iteration}")
            print(f"{'='*60}")
            
            try:
                # Captura imagem do NAO
                print("📷 Capturando imagem...")
                image = capture_image_from_nao(session)
                
                if image is None:
                    print("❌ Falha ao capturar imagem")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ Muitos erros consecutivos ({consecutive_errors}). Encerrando...")
                        break
                    time.sleep(INTERVAL_SECONDS)
                    continue
                
                # Executa detecção em todos os modelos YOLO
                print(f"\n🔥 Executando detecção em {len(yolo_models)} modelos...")
                results_dict = detect_objects_multi_yolo(yolo_models, image)
                
                # Coleta métricas
                collect_metrics(results_dict, iteration)
                
                # Desenha resultados
                annotated_images = draw_multi_yolo_results(image, results_dict, iteration)
                
                # Imprime resultados no console
                print_benchmark_results(results_dict, iteration)
                
                # Salva resultados
                save_benchmark_results(annotated_images, results_dict, iteration)
                
                # Reset contador de erros
                consecutive_errors = 0
                iteration += 1
                
                # A cada 5 iterações, gera gráficos atualizados
                if iteration % 5 == 0:
                    print("\n📊 Gerando gráficos de análise...")
                    create_performance_plots()
                
                # Aguarda próxima iteração
                print(f"\n⏳ Aguardando {INTERVAL_SECONDS}s para próxima iteração...")
                time.sleep(INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Erro na iteração {iteration}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Muitos erros consecutivos ({consecutive_errors}). Encerrando...")
                    break
                time.sleep(INTERVAL_SECONDS / 2)
    
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico no benchmark: {e}")
    finally:
        # Gera relatórios finais
        print("\n📊 Gerando análises finais...")
        try:
            create_performance_plots()
            generate_final_report()
        except Exception as e:
            print(f"❌ Erro ao gerar relatórios finais: {e}")
        
        # Limpeza
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("🧹 Limpeza concluída")
        print("👋 Benchmark finalizado")

if __name__ == "__main__":
    main()
