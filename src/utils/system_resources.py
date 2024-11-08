import psutil
import platform
import cpuinfo
import torch
import os
import GPUtil
from typing import Dict, List
import pandas as pd

class SystemResourceChecker:
    """Check and report system resources available for face detection"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.cpu_info = self._get_cpu_info()
        self.memory_info = self._get_memory_info()
        self.gpu_info = self._get_gpu_info()
        
    def _get_system_info(self) -> Dict:
        """Get basic system information"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'machine': platform.machine()
        }
    
    def _get_cpu_info(self) -> Dict:
        """Get detailed CPU information"""
        cpu_info = cpuinfo.get_cpu_info()
        return {
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'cpu_brand': cpu_info.get('brand_raw', 'Unknown'),
            'cpu_arch': cpu_info.get('arch', 'Unknown'),
            'cpu_bits': cpu_info.get('bits', 'Unknown')
        }
    
    def _get_memory_info(self) -> Dict:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_memory_gb': round(memory.total / (1024**3), 2),
            'available_memory_gb': round(memory.available / (1024**3), 2),
            'memory_percent_used': memory.percent,
            'swap_memory_gb': round(psutil.swap_memory().total / (1024**3), 2)
        }
    
    def _get_gpu_info(self) -> List[Dict]:
        """Get information about available GPUs"""
        gpu_info = []
        
        # Check CUDA availability through PyTorch
        torch_gpu_available = torch.cuda.is_available()
        if torch_gpu_available:
            torch_gpu_count = torch.cuda.device_count()
            for i in range(torch_gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'total_memory_gb': round(props.total_memory / (1024**3), 2),
                    'multi_processor_count': props.multi_processor_count
                })
        
        # Try to get additional GPU info through GPUtil
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                # Update existing info if we found it through PyTorch, or add new
                matching_info = next((g for g in gpu_info if gpu.name in g['name']), None)
                if matching_info:
                    matching_info.update({
                        'load_percent': round(gpu.load * 100, 2),
                        'memory_used_percent': round(gpu.memoryUtil * 100, 2),
                        'temperature': gpu.temperature
                    })
                else:
                    gpu_info.append({
                        'name': gpu.name,
                        'total_memory_gb': round(gpu.memoryTotal / 1024, 2),
                        'memory_used_percent': round(gpu.memoryUtil * 100, 2),
                        'load_percent': round(gpu.load * 100, 2),
                        'temperature': gpu.temperature
                    })
        except:
            pass  # GPUtil might not be available or have access
            
        return gpu_info
    
    def print_summary(self):
        """Print a formatted summary of system resources"""
        print("\nSystem Resource Summary")
        print("======================")
        
        print("\nSystem Information:")
        print(f"Platform: {self.system_info['platform']}")
        print(f"Python Version: {self.system_info['python_version']}")
        
        print("\nCPU Information:")
        print(f"Processor: {self.cpu_info['cpu_brand']}")
        print(f"Architecture: {self.cpu_info['cpu_arch']} ({self.cpu_info['cpu_bits']} bit)")
        print(f"Physical cores: {self.cpu_info['cpu_count_physical']}")
        print(f"Logical cores: {self.cpu_info['cpu_count_logical']}")
        if self.cpu_info['cpu_freq_max']:
            print(f"Max Frequency: {self.cpu_info['cpu_freq_max']/1000:.2f} GHz")
        
        print("\nMemory Information:")
        print(f"Total RAM: {self.memory_info['total_memory_gb']:.2f} GB")
        print(f"Available RAM: {self.memory_info['available_memory_gb']:.2f} GB")
        print(f"Memory Usage: {self.memory_info['memory_percent_used']}%")
        print(f"Swap Space: {self.memory_info['swap_memory_gb']:.2f} GB")
        
        print("\nGPU Information:")
        if self.gpu_info:
            for i, gpu in enumerate(self.gpu_info):
                print(f"\nGPU {i+1}: {gpu['name']}")
                for key, value in gpu.items():
                    if key != 'name':
                        print(f"  {key}: {value}")
        else:
            print("No GPU detected")
            
    def get_recommendations(self) -> List[str]:
        """Get recommendations for optimal face detection performance"""
        recommendations = []
        
        # CPU recommendations
        if self.cpu_info['cpu_count_logical'] <= 2:
            recommendations.append("Limited CPU cores available. Consider reducing batch size and parallel processing.")
        else:
            recommended_workers = min(self.cpu_info['cpu_count_logical'] - 1, 4)
            recommendations.append(f"Recommended number of workers for CPU processing: {recommended_workers}")
            
        # Memory recommendations
        available_memory_gb = self.memory_info['available_memory_gb']
        if available_memory_gb < 4:
            recommendations.append("Low available memory. Consider reducing batch size and image resolution.")
        else:
            max_batch_size = int(available_memory_gb / 2)  # Rough estimate
            recommendations.append(f"Recommended maximum batch size based on memory: {max_batch_size}")
            
        # GPU recommendations
        if self.gpu_info:
            for gpu in self.gpu_info:
                if gpu.get('memory_used_percent', 0) > 80:
                    recommendations.append(f"GPU {gpu['name']} memory usage is high. Consider reducing batch size.")
                if gpu.get('compute_capability', '0.0') < '3.5':
                    recommendations.append(f"GPU {gpu['name']} might have limited deep learning support.")
        else:
            recommendations.append("No GPU detected. Processing will be CPU-only.")
            
        return recommendations

# Usage example
if __name__ == "__main__":
    checker = SystemResourceChecker()
    checker.print_summary()
    
    print("\nRecommendations:")
    print("===============")
    for rec in checker.get_recommendations():
        print(f"- {rec}")