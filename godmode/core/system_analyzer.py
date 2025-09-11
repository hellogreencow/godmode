"""
System Analysis and Optimization Module for GodMode.

This module performs comprehensive system analysis to detect hardware capabilities
and dynamically optimize GodMode's performance based on available resources.
"""

import os
import platform
import subprocess
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SystemType(Enum):
    """System architecture types."""
    MAC_APPLE_SILICON = "mac_apple_silicon"
    MAC_INTEL = "mac_intel"
    LINUX = "linux"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class GPUType(Enum):
    """GPU/Accelerator types available."""
    APPLE_NEURAL_ENGINE = "apple_neural_engine"
    MPS = "mps"  # Metal Performance Shaders
    CUDA = "cuda"
    ROCM = "rocm"  # AMD
    CPU_ONLY = "cpu_only"
    UNKNOWN = "unknown"


@dataclass
class HardwareCapabilities:
    """Comprehensive hardware capabilities assessment."""
    system_type: SystemType
    cpu_cores: int
    memory_gb: float
    gpu_type: GPUType
    gpu_memory_gb: Optional[float] = None
    storage_gb: float = 0.0
    is_laptop: bool = False
    battery_powered: bool = False

    # Performance scores (0-100)
    cpu_score: int = 50
    memory_score: int = 50
    gpu_score: int = 0
    storage_score: int = 50
    overall_score: int = 50


@dataclass
class OptimizationProfile:
    """Dynamic optimization settings based on system capabilities."""
    # Memory management
    max_working_memory_items: int
    memory_chunk_size: int
    enable_memory_compression: bool

    # Processing
    enable_parallel_processing: bool
    max_concurrent_tasks: int
    use_gpu_acceleration: bool

    # Model settings
    preferred_model_size: str  # "small", "medium", "large"
    enable_model_caching: bool
    batch_size: int

    # Reasoning complexity
    max_reasoning_depth: int
    enable_complex_reasoning: bool
    attention_mechanism: str

    # Performance monitoring
    enable_performance_monitoring: bool
    monitoring_interval: float


class SystemAnalyzer:
    """Advanced system analysis and optimization engine."""

    def __init__(self):
        self.capabilities: Optional[HardwareCapabilities] = None
        self.optimization_profile: Optional[OptimizationProfile] = None

    def analyze_system(self) -> HardwareCapabilities:
        """Perform comprehensive system analysis."""
        logger.info("🔍 Starting comprehensive system analysis...")

        system_type = self._detect_system_type()
        cpu_info = self._analyze_cpu()
        memory_info = self._analyze_memory()
        gpu_info = self._analyze_gpu()
        storage_info = self._analyze_storage()
        power_info = self._analyze_power_source()

        capabilities = HardwareCapabilities(
            system_type=system_type,
            cpu_cores=cpu_info['cores'],
            memory_gb=memory_info['total_gb'],
            gpu_type=gpu_info['type'],
            gpu_memory_gb=gpu_info.get('memory_gb'),
            storage_gb=storage_info['total_gb'],
            is_laptop=power_info['is_laptop'],
            battery_powered=power_info['battery_powered']
        )

        # Calculate performance scores
        capabilities.cpu_score = self._calculate_cpu_score(cpu_info)
        capabilities.memory_score = self._calculate_memory_score(memory_info)
        capabilities.gpu_score = self._calculate_gpu_score(gpu_info)
        capabilities.storage_score = self._calculate_storage_score(storage_info)
        capabilities.overall_score = self._calculate_overall_score(capabilities)

        self.capabilities = capabilities
        logger.info("✅ System analysis complete!")
        logger.info(f"📊 Overall system score: {capabilities.overall_score}/100")

        return capabilities

    def generate_optimization_profile(self) -> OptimizationProfile:
        """Generate optimal configuration based on system capabilities."""
        if not self.capabilities:
            self.analyze_system()

        logger.info("🎯 Generating optimization profile...")

        caps = self.capabilities

        # Memory-based optimizations
        if caps.memory_gb >= 32:
            max_working_memory = 12
            memory_chunk_size = 2048
            enable_compression = False
        elif caps.memory_gb >= 16:
            max_working_memory = 9
            memory_chunk_size = 1024
            enable_compression = True
        else:
            max_working_memory = 7
            memory_chunk_size = 512
            enable_compression = True

        # CPU-based optimizations
        if caps.cpu_cores >= 8:
            enable_parallel = True
            max_concurrent = min(caps.cpu_cores - 2, 8)
        elif caps.cpu_cores >= 4:
            enable_parallel = True
            max_concurrent = 3
        else:
            enable_parallel = False
            max_concurrent = 1

        # GPU-based optimizations
        if caps.gpu_type in [GPUType.MPS, GPUType.CUDA]:
            use_gpu = True
            if caps.gpu_memory_gb and caps.gpu_memory_gb >= 8:
                preferred_model = "large"
                batch_size = 16
            elif caps.gpu_memory_gb and caps.gpu_memory_gb >= 4:
                preferred_model = "medium"
                batch_size = 8
            else:
                preferred_model = "small"
                batch_size = 4
        else:
            use_gpu = False
            preferred_model = "small" if caps.memory_gb < 16 else "medium"
            batch_size = 1

        # Reasoning complexity based on overall performance
        if caps.overall_score >= 80:
            max_depth = 8
            enable_complex = True
            attention = "multi_head"
        elif caps.overall_score >= 60:
            max_depth = 6
            enable_complex = True
            attention = "hierarchical"
        else:
            max_depth = 4
            enable_complex = False
            attention = "self_attention"

        profile = OptimizationProfile(
            max_working_memory_items=max_working_memory,
            memory_chunk_size=memory_chunk_size,
            enable_memory_compression=enable_compression,
            enable_parallel_processing=enable_parallel,
            max_concurrent_tasks=max_concurrent,
            use_gpu_acceleration=use_gpu,
            preferred_model_size=preferred_model,
            enable_model_caching=caps.storage_score > 70,
            batch_size=batch_size,
            max_reasoning_depth=max_depth,
            enable_complex_reasoning=enable_complex,
            attention_mechanism=attention,
            enable_performance_monitoring=caps.overall_score > 50,
            monitoring_interval=1.0 if caps.overall_score > 70 else 2.0
        )

        self.optimization_profile = profile
        logger.info("✅ Optimization profile generated!")

        return profile

    def _detect_system_type(self) -> SystemType:
        """Detect the system architecture."""
        system = platform.system().lower()

        if system == "darwin":
            try:
                # Check for Apple Silicon
                result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                      capture_output=True, text=True)
                if 'Apple' in result.stdout:
                    return SystemType.MAC_APPLE_SILICON
                else:
                    return SystemType.MAC_INTEL
            except:
                return SystemType.MAC_INTEL
        elif system == "linux":
            return SystemType.LINUX
        elif system == "windows":
            return SystemType.WINDOWS
        else:
            return SystemType.UNKNOWN

    def _analyze_cpu(self) -> Dict[str, Any]:
        """Analyze CPU capabilities."""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(['sysctl', 'hw.ncpu'], capture_output=True, text=True)
                cores = int(result.stdout.split(':')[1].strip())
            else:
                cores = os.cpu_count() or 1

            return {
                'cores': cores,
                'architecture': platform.machine(),
                'processor': platform.processor() or "Unknown"
            }
        except Exception as e:
            logger.warning(f"CPU analysis failed: {e}")
            return {'cores': 1, 'architecture': 'unknown', 'processor': 'unknown'}

    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory capabilities."""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                total_bytes = int(result.stdout.split(':')[1].strip())
                total_gb = total_bytes / (1024**3)
            else:
                import psutil
                total_bytes = psutil.virtual_memory().total
                total_gb = total_bytes / (1024**3)

            return {
                'total_gb': round(total_gb, 1),
                'available_gb': round(total_gb * 0.8, 1)  # Conservative estimate
            }
        except Exception as e:
            logger.warning(f"Memory analysis failed: {e}")
            return {'total_gb': 8.0, 'available_gb': 6.0}

    def _analyze_gpu(self) -> Dict[str, Any]:
        """Analyze GPU/accelerator capabilities."""
        try:
            import torch

            gpu_info = {'type': GPUType.CPU_ONLY}

            if torch.cuda.is_available():
                gpu_info['type'] = GPUType.CUDA
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif torch.backends.mps.is_available():
                gpu_info['type'] = GPUType.MPS
                # Apple Silicon systems typically have Neural Engine
                gpu_info['neural_engine'] = True
                # Estimate MPS memory (typically unified with system RAM)
                memory_info = self._analyze_memory()
                gpu_info['memory_gb'] = memory_info['total_gb'] * 0.7  # Conservative estimate
            elif platform.system() == "Darwin":
                # Check for Apple Neural Engine
                gpu_info['type'] = GPUType.APPLE_NEURAL_ENGINE

            return gpu_info

        except Exception as e:
            logger.warning(f"GPU analysis failed: {e}")
            return {'type': GPUType.CPU_ONLY}

    def _analyze_storage(self) -> Dict[str, Any]:
        """Analyze storage capabilities."""
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 2:
                        total = parts[1]
                        # Convert to GB
                        if total.endswith('Gi'):
                            total_gb = float(total[:-2])
                        elif total.endswith('Ti'):
                            total_gb = float(total[:-2]) * 1024
                        else:
                            total_gb = 256.0  # Default assumption
                    else:
                        total_gb = 256.0
                else:
                    total_gb = 256.0
            else:
                import psutil
                total_gb = psutil.disk_usage('/').total / (1024**3)

            return {
                'total_gb': round(total_gb, 1),
                'type': 'SSD' if total_gb > 100 else 'Unknown'  # Assumption for modern systems
            }
        except Exception as e:
            logger.warning(f"Storage analysis failed: {e}")
            return {'total_gb': 256.0, 'type': 'Unknown'}

    def _analyze_power_source(self) -> Dict[str, Any]:
        """Analyze power source and mobility."""
        try:
            # Check if it's a laptop (has battery)
            if platform.system() == "Darwin":
                result = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True)
                is_laptop = 'Battery' in result.stdout or 'InternalBattery' in result.stdout
                battery_powered = is_laptop and 'AC Power' not in result.stdout
            else:
                # For other systems, make educated guesses based on system type
                system = platform.system().lower()
                is_laptop = 'laptop' in platform.node().lower() or system in ['darwin']
                battery_powered = is_laptop

            return {
                'is_laptop': is_laptop,
                'battery_powered': battery_powered
            }
        except Exception as e:
            logger.warning(f"Power analysis failed: {e}")
            return {'is_laptop': False, 'battery_powered': False}

    def _calculate_cpu_score(self, cpu_info: Dict[str, Any]) -> int:
        """Calculate CPU performance score."""
        cores = cpu_info['cores']

        if cores >= 12:
            return 90
        elif cores >= 8:
            return 75
        elif cores >= 4:
            return 60
        elif cores >= 2:
            return 40
        else:
            return 20

    def _calculate_memory_score(self, memory_info: Dict[str, Any]) -> int:
        """Calculate memory performance score."""
        gb = memory_info['total_gb']

        if gb >= 64:
            return 95
        elif gb >= 32:
            return 85
        elif gb >= 16:
            return 70
        elif gb >= 8:
            return 50
        else:
            return 20

    def _calculate_gpu_score(self, gpu_info: Dict[str, Any]) -> int:
        """Calculate GPU performance score."""
        gpu_type = gpu_info['type']

        if gpu_type == GPUType.CUDA:
            memory_gb = gpu_info.get('memory_gb', 0)
            if memory_gb >= 24:
                return 95
            elif memory_gb >= 12:
                return 85
            elif memory_gb >= 8:
                return 75
            else:
                return 60
        elif gpu_type == GPUType.MPS:
            return 80  # Apple Silicon MPS
        elif gpu_type == GPUType.APPLE_NEURAL_ENGINE:
            return 70
        else:
            return 0

    def _calculate_storage_score(self, storage_info: Dict[str, Any]) -> int:
        """Calculate storage performance score."""
        gb = storage_info['total_gb']

        if gb >= 1000:
            return 90
        elif gb >= 500:
            return 75
        elif gb >= 256:
            return 60
        elif gb >= 128:
            return 40
        else:
            return 20

    def _calculate_overall_score(self, capabilities: HardwareCapabilities) -> int:
        """Calculate overall system performance score."""
        weights = {
            'cpu': 0.25,
            'memory': 0.30,
            'gpu': 0.25,
            'storage': 0.20
        }

        overall = (
            capabilities.cpu_score * weights['cpu'] +
            capabilities.memory_score * weights['memory'] +
            capabilities.gpu_score * weights['gpu'] +
            capabilities.storage_score * weights['storage']
        )

        return int(overall)

    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        if not self.capabilities:
            self.analyze_system()

        if not self.optimization_profile:
            self.generate_optimization_profile()

        return {
            'hardware_capabilities': {
                'system_type': self.capabilities.system_type.value,
                'cpu_cores': self.capabilities.cpu_cores,
                'memory_gb': self.capabilities.memory_gb,
                'gpu_type': self.capabilities.gpu_type.value,
                'gpu_memory_gb': self.capabilities.gpu_memory_gb,
                'storage_gb': self.capabilities.storage_gb,
                'is_laptop': self.capabilities.is_laptop,
                'battery_powered': self.capabilities.battery_powered,
                'performance_scores': {
                    'cpu': self.capabilities.cpu_score,
                    'memory': self.capabilities.memory_score,
                    'gpu': self.capabilities.gpu_score,
                    'storage': self.capabilities.storage_score,
                    'overall': self.capabilities.overall_score
                }
            },
            'optimization_profile': {
                'memory_management': {
                    'max_working_memory_items': self.optimization_profile.max_working_memory_items,
                    'memory_chunk_size': self.optimization_profile.memory_chunk_size,
                    'enable_memory_compression': self.optimization_profile.enable_memory_compression
                },
                'processing': {
                    'enable_parallel_processing': self.optimization_profile.enable_parallel_processing,
                    'max_concurrent_tasks': self.optimization_profile.max_concurrent_tasks,
                    'use_gpu_acceleration': self.optimization_profile.use_gpu_acceleration
                },
                'model_settings': {
                    'preferred_model_size': self.optimization_profile.preferred_model_size,
                    'enable_model_caching': self.optimization_profile.enable_model_caching,
                    'batch_size': self.optimization_profile.batch_size
                },
                'reasoning': {
                    'max_reasoning_depth': self.optimization_profile.max_reasoning_depth,
                    'enable_complex_reasoning': self.optimization_profile.enable_complex_reasoning,
                    'attention_mechanism': self.optimization_profile.attention_mechanism
                },
                'monitoring': {
                    'enable_performance_monitoring': self.optimization_profile.enable_performance_monitoring,
                    'monitoring_interval': self.optimization_profile.monitoring_interval
                }
            },
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate system-specific recommendations."""
        if not self.capabilities:
            return []

        recommendations = []

        # Memory recommendations
        if self.capabilities.memory_gb < 16:
            recommendations.append("Consider upgrading to 16GB+ RAM for optimal performance")
        elif self.capabilities.memory_gb < 32:
            recommendations.append("32GB+ RAM recommended for complex reasoning tasks")

        # CPU recommendations
        if self.capabilities.cpu_cores < 4:
            recommendations.append("Multi-core CPU recommended for parallel processing")
        elif self.capabilities.cpu_cores < 8:
            recommendations.append("8+ CPU cores optimal for advanced reasoning")

        # GPU recommendations
        if self.capabilities.gpu_type == GPUType.CPU_ONLY:
            recommendations.append("GPU acceleration recommended for faster processing")
        elif self.capabilities.gpu_type == GPUType.MPS:
            recommendations.append("Apple Silicon detected - MPS acceleration enabled")
        elif self.capabilities.gpu_type == GPUType.CUDA:
            recommendations.append("NVIDIA GPU detected - CUDA acceleration enabled")

        # Storage recommendations
        if self.capabilities.storage_gb < 256:
            recommendations.append("256GB+ SSD recommended for model caching")
        elif self.capabilities.storage_gb < 512:
            recommendations.append("512GB+ SSD optimal for large model storage")

        # Performance recommendations
        if self.capabilities.overall_score >= 80:
            recommendations.append("System optimized for maximum performance")
        elif self.capabilities.overall_score >= 60:
            recommendations.append("System suitable for most reasoning tasks")
        else:
            recommendations.append("Consider system upgrades for better performance")

        return recommendations


# Global analyzer instance
system_analyzer = SystemAnalyzer()
