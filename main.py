#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import qi
import time
import os
from datetime import datetime
import platform
import sys
import math
from importlib import metadata
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import threading
import GPUtil
from collections import Counter

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
    "YOLOv9c": "yolov9c.pt",
    "YOLOv9e": "yolov9e.pt",
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
        print(f"Falha no método alternativo de captura: {e}")
        raise RuntimeError(
            "Não foi possível obter uma imagem real do NAO; a iteração será descartada."
        ) from e

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
    """Gera gráficos consolidados usando os nomes reais das métricas coletadas."""
    if not METRICS_DATA:
        print("Não há dados para gerar gráficos.")
        return

    df = pd.DataFrame(METRICS_DATA)
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("NAO Multi-YOLO Benchmark — Performance Summary", fontsize=16)

    fps_summary = df.groupby("model")["fps"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    fps_summary["std"] = fps_summary["std"].fillna(0)
    axes[0, 0].bar(fps_summary.index, fps_summary["mean"], yerr=fps_summary["std"], capsize=2)
    axes[0, 0].set_title("Mean FPS by model (± SD)")
    axes[0, 0].set_ylabel("Frames per second")
    axes[0, 0].tick_params(axis="x", labelrotation=60, labelsize=7)
    axes[0, 0].grid(axis="y", alpha=0.25)

    time_summary = df.groupby("model")["processing_time"].mean().sort_values()
    axes[0, 1].bar(time_summary.index, time_summary.values)
    axes[0, 1].set_title("Mean inference time by model")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].tick_params(axis="x", labelrotation=60, labelsize=7)
    axes[0, 1].grid(axis="y", alpha=0.25)

    confidence_summary = df.groupby("model")["confidence_mean"].mean().sort_values(ascending=False)
    axes[0, 2].bar(confidence_summary.index, confidence_summary.values)
    axes[0, 2].set_title("Mean detection confidence score")
    axes[0, 2].set_ylabel("Confidence score")
    axes[0, 2].tick_params(axis="x", labelrotation=60, labelsize=7)
    axes[0, 2].grid(axis="y", alpha=0.25)

    objects_summary = df.groupby("model")["objects_count"].mean().sort_values(ascending=False)
    axes[1, 0].bar(objects_summary.index, objects_summary.values)
    axes[1, 0].set_title("Mean number of detections per frame")
    axes[1, 0].set_ylabel("Detections")
    axes[1, 0].tick_params(axis="x", labelrotation=60, labelsize=7)
    axes[1, 0].grid(axis="y", alpha=0.25)

    if RESOURCE_DATA:
        resource_df = pd.DataFrame(RESOURCE_DATA)
        resource_summary = resource_df.groupby("model")[["cpu_avg", "memory_avg", "gpu_avg"]].mean()
        resource_summary = resource_summary.sort_index()
        resource_summary.plot(kind="bar", ax=axes[1, 1], width=0.85)
        axes[1, 1].set_title("Mean system resource utilization")
        axes[1, 1].set_ylabel("Utilization (%)")
        axes[1, 1].set_xlabel("")
        axes[1, 1].legend(["CPU", "RAM", "GPU"], fontsize=8)
        axes[1, 1].tick_params(axis="x", labelrotation=60, labelsize=7)
        axes[1, 1].grid(axis="y", alpha=0.25)
    else:
        axes[1, 1].text(0.5, 0.5, "No resource data available", ha="center", va="center")
        axes[1, 1].set_axis_off()

    ordered_models = list(fps_summary.index)
    fps_samples = [df.loc[df["model"] == model, "fps"].dropna().values for model in ordered_models]
    axes[1, 2].boxplot(fps_samples, labels=ordered_models, showmeans=True)
    axes[1, 2].set_title("FPS distribution across iterations")
    axes[1, 2].set_ylabel("Frames per second")
    axes[1, 2].tick_params(axis="x", labelrotation=60, labelsize=7)
    axes[1, 2].grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(
        figures_dir,
        f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico consolidado salvo: {plot_filename}")


def _json_safe(value):
    """Converte valores NumPy e estruturas aninhadas para tipos aceitos pelo JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _distribution_version(package_name):
    """Retorna a versão instalada ou None quando não há metadados disponíveis."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _runtime_metadata():
    """Registra informações observáveis do ambiente em que o benchmark foi executado."""
    try:
        gpu_info = [
            {"name": str(gpu.name), "memory_total_mb": float(gpu.memoryTotal)}
            for gpu in GPUtil.getGPUs()
        ]
    except Exception:
        gpu_info = []

    try:
        import torch
        torch_info = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else []
        }
    except Exception:
        torch_info = {}

    return {
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "system_ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_devices": gpu_info,
        "software": {
            "ultralytics": _distribution_version("ultralytics"),
            "torch": torch_info,
            "opencv": _distribution_version("opencv-python") or _distribution_version("opencv-python-headless"),
            "numpy": _distribution_version("numpy"),
            "pandas": _distribution_version("pandas"),
            "matplotlib": _distribution_version("matplotlib"),
            "psutil": _distribution_version("psutil"),
            "GPUtil": _distribution_version("GPUtil")
        }
    }


def save_benchmark_results(annotated_images, results_dict, iteration):
    """Salva imagens, JSON detalhado e CSVs acumulados sem acessar campos inexistentes."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        annotated_dir = os.path.join(OUTPUT_DIR, "annotated_images")
        os.makedirs(annotated_dir, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        for model_name, image in annotated_images.items():
            filename = os.path.join(annotated_dir, f"{model_name}_{iteration}_{timestamp}.jpg")
            if not cv2.imwrite(filename, image):
                print(f"Aviso: não foi possível salvar a imagem de {model_name}: {filename}")

        model_records = {}
        for model_name, data in results_dict.items():
            if data is None:
                continue
            model_records[model_name] = {
                "model_file": YOLO_MODELS.get(model_name),
                "processing_time_s": data.get("processing_time"),
                "processing_time_std_s": data.get("time_std"),
                "fps": data.get("fps"),
                "objects_count": data.get("objects_count"),
                "unique_classes": data.get("unique_classes"),
                "confidence_stats": data.get("confidence_stats", {}),
                "class_counts": data.get("class_counts", {}),
                "class_performance": data.get("class_performance", {}),
                "size_stats": data.get("size_stats", {}),
                "resource_usage": data.get("resource_usage", {}),
                "objects": data.get("objects", [])
            }

        image_metadata = {}
        sample_image = next(iter(annotated_images.values()), None)
        if sample_image is not None and hasattr(sample_image, "shape"):
            image_metadata = {
                "width": int(sample_image.shape[1]),
                "height": int(sample_image.shape[0])
            }

        benchmark_data = {
            "schema_version": "1.0",
            "timestamp": now.isoformat(),
            "iteration": iteration,
            "image": image_metadata,
            "runtime": _runtime_metadata(),
            "models": model_records
        }

        json_filename = os.path.join(OUTPUT_DIR, f"benchmark_{iteration}_{timestamp}.json")
        with open(json_filename, "w", encoding="utf-8") as output_file:
            json.dump(_json_safe(benchmark_data), output_file, indent=2, ensure_ascii=False, allow_nan=False)

        if METRICS_DATA:
            metrics_filename = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
            pd.DataFrame(METRICS_DATA).to_csv(metrics_filename, index=False)
            print(f"CSV de métricas salvo: {metrics_filename}")

        if RESOURCE_DATA:
            resources_filename = os.path.join(OUTPUT_DIR, "resources_summary.csv")
            pd.DataFrame(RESOURCE_DATA).to_csv(resources_filename, index=False)
            print(f"CSV de recursos salvo: {resources_filename}")

        if CLASS_PERFORMANCE_DATA:
            classes_filename = os.path.join(OUTPUT_DIR, "class_performance.csv")
            pd.DataFrame(CLASS_PERFORMANCE_DATA).to_csv(classes_filename, index=False)
            print(f"CSV por classe salvo: {classes_filename}")

        print(f"Arquivos da iteração {iteration} salvos:")
        print(f"  Imagens anotadas: {len(annotated_images)}")
        print(f"  Dados JSON: {json_filename}")
        return benchmark_data

    except Exception as error:
        print(f"Erro ao salvar resultados da iteração {iteration}: {error}")
        return None


def generate_final_report():
    """Gera relatório consolidado com as métricas efetivamente coletadas."""
    if not METRICS_DATA:
        print("Não há métricas para gerar o relatório final.")
        return

    df = pd.DataFrame(METRICS_DATA)
    aggregation = {
        "processing_time": ["mean", "std", "min", "max"],
        "fps": ["mean", "std", "min", "max"],
        "objects_count": ["mean", "std", "min", "max"],
        "confidence_mean": ["mean", "std", "min", "max"],
        "confidence_max": ["mean", "std", "min", "max"]
    }
    stats = df.groupby("model").agg(aggregation).round(4)

    report_filename = os.path.join(
        OUTPUT_DIR,
        f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(report_filename, "w", encoding="utf-8") as report_file:
        report_file.write("=" * 80 + "\n")
        report_file.write("NAO MULTI-YOLO BENCHMARK — FINAL REPORT\n")
        report_file.write("=" * 80 + "\n\n")
        report_file.write(f"Iterations recorded: {df['iteration'].nunique()}\n")
        report_file.write(f"Models recorded: {df['model'].nunique()}\n")
        report_file.write(f"Analysis period: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")

        report_file.write("PER-MODEL DESCRIPTIVE STATISTICS\n")
        report_file.write("-" * 80 + "\n")
        report_file.write(stats.to_string())
        report_file.write("\n\n")

        fastest = df.groupby("model")["fps"].mean().sort_values(ascending=False)
        report_file.write("RANKING BY MEAN FPS\n")
        report_file.write("-" * 80 + "\n")
        for rank, (model_name, mean_fps) in enumerate(fastest.items(), 1):
            report_file.write(f"{rank:2d}. {model_name}: {mean_fps:.2f} FPS\n")

        confidence = df.groupby("model")["confidence_mean"].mean().sort_values(ascending=False)
        report_file.write("\nRANKING BY MEAN DETECTION CONFIDENCE SCORE\n")
        report_file.write("-" * 80 + "\n")
        for rank, (model_name, mean_score) in enumerate(confidence.items(), 1):
            report_file.write(f"{rank:2d}. {model_name}: {mean_score:.4f}\n")

        if RESOURCE_DATA:
            resource_df = pd.DataFrame(RESOURCE_DATA)
            resource_stats = resource_df.groupby("model")[
                ["cpu_avg", "memory_avg", "gpu_avg"]
            ].mean().round(2)
            report_file.write("\nMEAN SYSTEM RESOURCE UTILIZATION (%)\n")
            report_file.write("-" * 80 + "\n")
            report_file.write(resource_stats.to_string())
            report_file.write("\n")

        report_file.write(
            "\nNOTE: Detection confidence scores are model output scores; "
            "they are not accuracy, precision, recall, or mAP measurements.\n"
        )
        report_file.write(
            "CPU, RAM, and GPU percentages are system-level samples collected "
            "during each model run, not isolated per-process measurements.\n"
        )

    print(f"Relatório final salvo: {report_filename}")


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
