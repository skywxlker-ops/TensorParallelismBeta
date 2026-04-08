#!/usr/bin/env python3
"""
Advanced System Monitor with GPU-CPU Affinity Detection
Shows which CPU cores are used for GPU data transfers and PCIe communication.
Supports an arbitrary number of NVIDIA GPUs (all that NVML reports).
"""

import psutil
import time
import csv
import os
import signal
import argparse
import subprocess
import re
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import deque, defaultdict

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


# ---------------------------------------------------------------------------
# PCIe per-lane bandwidth table (MB/s, one direction).
# Multiply by actual negotiated link width to get the full-link ceiling.
#
#   Gen1:  2.5 GT/s  x 8b/10b     =  250 MB/s/lane
#   Gen2:  5.0 GT/s  x 8b/10b     =  500 MB/s/lane
#   Gen3:  8.0 GT/s  x 128b/130b ~=  985 MB/s/lane
#   Gen4: 16.0 GT/s  x 128b/130b ~= 1969 MB/s/lane
#   Gen5: 32.0 GT/s  x 128b/130b ~= 3938 MB/s/lane
#   Gen6: 64.0 GT/s  x PAM4/FLIT ~= 7877 MB/s/lane
# ---------------------------------------------------------------------------
PCIE_PER_LANE_MB_S: Dict[int, int] = {
    1:  250,
    2:  500,
    3:  985,
    4: 1969,
    5: 3938,
    6: 7877,
}


class AdvancedSystemMonitor:
    """Advanced monitor showing GPU-CPU affinity and PCIe topology."""

    def __init__(self, output_file: str = None, interval: float = 1.0,
                 duration: Optional[float] = None, show_affinity: bool = True):
        self.interval = interval
        self.duration = duration
        self.output_file = (output_file or
                            f"system_monitor_adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.show_affinity = show_affinity
        self.running = False
        self.samples = 0

        # System info
        self.num_cpus = psutil.cpu_count(logical=True)
        self.num_physical_cores = psutil.cpu_count(logical=False)

        # GPU info -- gpu_count must exist before _detect_cpu_gpu_topology
        self.gpu_initialized = False
        self.gpu_count = 0
        self.gpu_available = self._init_gpu()

        # CPU-GPU topology (NUMA affinity per GPU)
        self.cpu_gpu_topology = self._detect_cpu_gpu_topology()

        # ---------------------------------------------------------------
        # Per-GPU rolling history and affinity logs.
        #
        # gpu_history[g]   - deque(maxlen=10) of recent utilisation floats
        #                    for GPU g. Used for live display and the
        #                    magnitude-of-change correlation metric.
        #
        # affinity_log[g]  - full time-series list of
        #                    (gpu_util: float, cpu_per_core: List[float])
        #                    tuples for GPU g. Never trimmed so the
        #                    end-of-run analysis sees every sample.
        #
        # core_history     - per-CPU-core rolling deque used by the live
        #                    display (shared across all GPUs; we just need
        #                    recent CPU load, not per-GPU splits here).
        # ---------------------------------------------------------------
        self.core_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))

        # Indexed by GPU id
        self.gpu_history: Dict[int, deque] = {
            g: deque(maxlen=10) for g in range(self.gpu_count)
        }
        self.affinity_log: Dict[int, List[Tuple[float, List[float]]]] = {
            g: [] for g in range(self.gpu_count)
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Static header list -- built once at startup so column order is stable
        self._csv_headers = self._build_headers()

        # Create CSV with header row
        self._init_csv()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_gpu(self) -> bool:
        """Initialise NVML and discover how many GPUs are present."""
        if not HAS_PYNVML:
            return False
        try:
            pynvml.nvmlInit()
            self.gpu_initialized = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            return self.gpu_count > 0
        except Exception:
            return False

    def _detect_cpu_gpu_topology(self) -> Dict:
        """
        Detect which CPU cores are NUMA-local to each GPU by inspecting
        PCIe bus IDs via lspci.

        topology['pcie_buses'][gpu_id]  = PCI bus number (int)
        topology['numa_nodes'][gpu_id]  = NUMA node index (int)
        topology['gpu_cores'][gpu_id]   = list of CPU logical core indices
        """
        topology: Dict = {
            'gpu_cores':  {},
            'numa_nodes': {},
            'pcie_buses': {},
        }

        if not HAS_PYNVML or self.gpu_count == 0:
            return topology

        try:
            # Collect bus IDs and NUMA nodes for every GPU
            for i in range(self.gpu_count):
                handle   = pynvml.nvmlDeviceGetHandleByIndex(i)
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id   = pci_info.bus
                topology['pcie_buses'][i] = bus_id

                # Attempt NUMA detection via lspci
                try:
                    result = subprocess.run(
                        ['lspci', '-vv', '-s', f'{bus_id:02x}:00.0'],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.split('\n'):
                        if 'NUMA' in line:
                            m = re.search(r'NUMA\s+node\s+(\d+)', line)
                            if m:
                                topology['numa_nodes'][i] = int(m.group(1))
                except Exception:
                    pass

                # Fallback: assign NUMA node 0 if detection failed
                if i not in topology['numa_nodes']:
                    topology['numa_nodes'][i] = 0

            # Map NUMA nodes to CPU core ranges
            unique_numa    = set(topology['numa_nodes'].values())
            num_numa_nodes = max(len(unique_numa), 1)
            cores_per_numa = self.num_cpus // num_numa_nodes

            for gpu_id, numa_node in topology['numa_nodes'].items():
                start_core = numa_node * cores_per_numa
                end_core   = min(start_core + cores_per_numa, self.num_cpus)
                topology['gpu_cores'][gpu_id] = list(range(start_core, end_core))

        except Exception as e:
            print(f"[Topology] Detection error: {e}")

        return topology

    def _build_headers(self) -> List[str]:
        """
        Build the complete, ordered list of CSV column names.

        GPU columns follow the pattern  gpuN_<metric>  where N is the
        zero-based GPU index. System-wide GPU summary columns (totals /
        averages across all GPUs) are prefixed with  gpu_system_.
        """
        headers = [
            'timestamp', 'elapsed_seconds', 'sample_number',
            'cpu_system_percent',
        ]

        # Per-logical-core CPU utilisation
        for i in range(self.num_cpus):
            headers.append(f'cpu_core_{i}_percent')

        headers.extend([
            'cpu_frequency_mhz', 'cpu_ctx_switches', 'cpu_interrupts',
            'cpu_soft_interrupts', 'cpu_syscalls',
        ])

        # Memory
        headers.extend([
            'ram_total_gb', 'ram_used_gb', 'ram_available_gb', 'ram_percent',
            'ram_active_gb', 'ram_cached_gb', 'ram_buffers_gb',
            'ram_shared_gb', 'ram_sreclaimable_gb',
            'swap_total_gb', 'swap_used_gb', 'swap_free_gb', 'swap_percent',
            'swap_in', 'swap_out',
        ])

        # Disk I/O
        headers.extend([
            'disk_read_mb', 'disk_write_mb', 'disk_read_count',
            'disk_write_count', 'disk_read_time_ms', 'disk_write_time_ms',
        ])

        # Network
        headers.extend([
            'net_sent_mb', 'net_recv_mb', 'net_sent_packets',
            'net_recv_packets', 'net_errin', 'net_errout',
            'net_dropin', 'net_dropout',
        ])

        # GPU -- system-wide summary first, then per-GPU indexed columns
        if self.gpu_available:
            headers.extend([
                'gpu_count',
                'gpu_system_util_avg_percent',     # mean utilisation across all GPUs
                'gpu_system_power_total_w',         # sum of power draw
                'gpu_system_memory_used_gb',        # sum of VRAM used
                'gpu_system_memory_total_gb',       # sum of VRAM total
                'gpu_system_pcie_tx_total_mb_s',    # sum of PCIe TX across all GPUs
                'gpu_system_pcie_rx_total_mb_s',    # sum of PCIe RX across all GPUs
            ])

            # Per-GPU indexed columns
            per_gpu_fields = [
                'util_percent',
                'memory_used_gb',
                'memory_total_gb',
                'memory_percent',
                'temperature_c',
                'power_draw_w',
                'power_limit_w',
                'clock_graphics_mhz',
                'clock_memory_mhz',
                'pcie_tx_mb_s',
                'pcie_rx_mb_s',
                'pcie_link_ceiling_mb_s',
                'pcie_tx_util_percent',
                'pcie_rx_util_percent',
                'pcie_gen',
                'pcie_width',
                'pcie_max_gen',
                'pcie_max_width',
                'numa_node',
            ]
            for g in range(self.gpu_count):
                for field in per_gpu_fields:
                    headers.append(f'gpu{g}_{field}')

        # CPU temperatures
        headers.extend([
            'cpu_package_temp_c',
            'cpu_core_temp_avg_c',
            'cpu_core_temp_max_c',
        ])
        for i in range(min(self.num_cpus, 28)):
            headers.append(f'cpu_core_{i}_temp_c')

        # GPU-CPU affinity -- one set of columns per GPU
        if self.gpu_available:
            for g in range(self.gpu_count):
                headers.extend([
                    f'gpu{g}_active_cores_mask',
                    f'gpu{g}_data_transfer_cores',
                    f'gpu{g}_cpu_correlation',
                ])

        return headers

    def _init_csv(self):
        """Create the CSV file with its header row and print startup info."""
        with open(self.output_file, 'w', newline='') as f:
            csv.writer(f).writerow(self._csv_headers)

        print(f"[Monitor] Output : {os.path.abspath(self.output_file)}")
        print(f"[Monitor] CPUs   : {self.num_cpus} logical "
              f"({self.num_physical_cores} physical)")
        print(f"[Monitor] GPUs   : {self.gpu_count} detected")

        if self.cpu_gpu_topology['gpu_cores']:
            print("[Monitor] GPU-CPU NUMA affinity:")
            for gpu_id, cores in self.cpu_gpu_topology['gpu_cores'].items():
                numa = self.cpu_gpu_topology['numa_nodes'].get(gpu_id, '?')
                print(f"  GPU {gpu_id}: NUMA node {numa}, "
                      f"CPU cores {cores[0]}-{cores[-1]}")

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def _get_detailed_cpu_metrics(self) -> Dict:
        """Collect per-core utilisation, frequency, stats and temperatures."""
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        freq  = psutil.cpu_freq()
        stats = psutil.cpu_stats()

        metrics: Dict = {
            'cpu_system_percent':  sum(cpu_per_core) / len(cpu_per_core),
            'cpu_per_core':        cpu_per_core,
            'cpu_frequency_mhz':   freq.current if freq else 0,
            'cpu_ctx_switches':    stats.ctx_switches,
            'cpu_interrupts':      stats.interrupts,
            'cpu_soft_interrupts': stats.soft_interrupts,
            'cpu_syscalls':        getattr(stats, 'syscalls', 0),
        }

        # CPU temperature (Linux coretemp sensor)
        pkg_temp       = 0.0
        core_temps:    List[float]       = []
        per_core_temps: Dict[int, float] = {}

        try:
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                for entry in temps['coretemp']:
                    if 'Package' in entry.label:
                        pkg_temp = float(entry.current)
                    elif entry.label.startswith('Core'):
                        try:
                            core_num = int(entry.label.replace('Core', '').strip())
                            per_core_temps[core_num] = float(entry.current)
                            core_temps.append(float(entry.current))
                        except ValueError:
                            pass
        except Exception:
            pass

        metrics['cpu_package_temp_c']  = pkg_temp
        metrics['cpu_core_temp_avg_c'] = (sum(core_temps) / len(core_temps)
                                           if core_temps else 0.0)
        metrics['cpu_core_temp_max_c'] = max(core_temps) if core_temps else 0.0

        # Sensor core IDs may be non-sequential; map them to 0-based indices
        temp_values = list(per_core_temps.values())
        for i in range(min(self.num_cpus, 28)):
            metrics[f'cpu_core_{i}_temp_c'] = (float(temp_values[i])
                                                if i < len(temp_values) else 0.0)

        return metrics

    def _get_detailed_memory_metrics(self) -> Dict:
        """Collect virtual memory and swap metrics."""
        mem  = psutil.virtual_memory()
        swap = psutil.swap_memory()

        meminfo: Dict[str, int] = {}
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        # /proc/meminfo values are in kB
                        meminfo[key.strip()] = int(val.strip().split()[0]) * 1024
        except Exception:
            pass

        cached_bytes = meminfo.get('Cached', getattr(mem, 'cached', 0))

        return {
            'ram_total_gb':        mem.total / 1e9,
            'ram_used_gb':         mem.used  / 1e9,
            'ram_available_gb':    mem.available / 1e9,
            'ram_percent':         mem.percent,
            'ram_active_gb':       meminfo.get('Active', 0) / 1e9,
            'ram_cached_gb':       cached_bytes / 1e9,
            'ram_buffers_gb':      meminfo.get('Buffers', 0) / 1e9,
            'ram_shared_gb':       meminfo.get('Shmem', 0) / 1e9,
            'ram_sreclaimable_gb': meminfo.get('SReclaimable', 0) / 1e9,
            'swap_total_gb':       swap.total / 1e9,
            'swap_used_gb':        swap.used  / 1e9,
            'swap_free_gb':        swap.free  / 1e9,
            'swap_percent':        swap.percent,
            'swap_in':             swap.sin,
            'swap_out':            swap.sout,
        }

    def _get_detailed_disk_metrics(self) -> Dict:
        """Collect cumulative disk I/O counters (since boot)."""
        io = psutil.disk_io_counters()
        if io is None:
            return {k: 0 for k in [
                'disk_read_mb', 'disk_write_mb', 'disk_read_count',
                'disk_write_count', 'disk_read_time_ms', 'disk_write_time_ms',
            ]}
        return {
            'disk_read_mb':       io.read_bytes  / 1e6,
            'disk_write_mb':      io.write_bytes / 1e6,
            'disk_read_count':    io.read_count,
            'disk_write_count':   io.write_count,
            'disk_read_time_ms':  io.read_time,
            'disk_write_time_ms': io.write_time,
        }

    def _get_detailed_network_metrics(self) -> Dict:
        """Collect cumulative network I/O counters (since boot)."""
        net = psutil.net_io_counters()
        return {
            'net_sent_mb':      net.bytes_sent / 1e6,
            'net_recv_mb':      net.bytes_recv / 1e6,
            'net_sent_packets': net.packets_sent,
            'net_recv_packets': net.packets_recv,
            'net_errin':        net.errin,
            'net_errout':       net.errout,
            'net_dropin':       net.dropin,
            'net_dropout':      net.dropout,
        }

    def _get_single_gpu_metrics(self, gpu_id: int) -> Dict:
        """
        Collect all NVML metrics for one GPU and return them under
        gpu{gpu_id}_<field> keys, ready to merge directly into the CSV row.

        PCIe throughput design
        ----------------------
        nvmlDeviceGetPcieThroughput() returns KB/s averaged over a ~20 ms
        driver-internal window.  The full-link one-directional ceiling is:

            PCIE_PER_LANE_MB_S[gen] x actual_link_width

        Link width is read BEFORE the throughput call so the ceiling is
        always consistent with what the hardware reports at that instant.

        We do NOT cap or discard values that exceed the ceiling -- anomalous
        readings stay in the CSV where they are visible for diagnosis.
        A warning is printed to stderr with the GPU index clearly labelled.
        """
        prefix  = f'gpu{gpu_id}_'
        metrics: Dict = {}

        def put(key: str, value) -> None:
            metrics[prefix + key] = value

        # Return a zeroed-out row for this GPU if NVML can't open the handle,
        # so the CSV schema stays intact regardless of per-GPU failures.
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception as e:
            print(f"\n[Monitor][GPU {gpu_id}] Handle error: {e}")
            for field in [
                'util_percent', 'memory_used_gb', 'memory_total_gb',
                'memory_percent', 'temperature_c', 'power_draw_w',
                'power_limit_w', 'clock_graphics_mhz', 'clock_memory_mhz',
                'pcie_tx_mb_s', 'pcie_rx_mb_s', 'pcie_link_ceiling_mb_s',
                'pcie_tx_util_percent', 'pcie_rx_util_percent',
                'pcie_gen', 'pcie_width', 'pcie_max_gen', 'pcie_max_width',
                'numa_node',
            ]:
                put(field, 0)
            return metrics

        # -- Utilisation -----------------------------------------------
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            put('util_percent',   util.gpu)
            put('memory_percent', util.memory)
        except Exception:
            put('util_percent',   0)
            put('memory_percent', 0)

        # -- VRAM ------------------------------------------------------
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            put('memory_used_gb',  mem.used  / 1e9)
            put('memory_total_gb', mem.total / 1e9)
        except Exception:
            put('memory_used_gb',  0)
            put('memory_total_gb', 0)

        # -- Temperature -----------------------------------------------
        try:
            put('temperature_c',
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            put('temperature_c', 0)

        # -- Power (NVML returns milliwatts) ---------------------------
        try:
            put('power_draw_w',
                pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        except Exception:
            put('power_draw_w', 0)
        try:
            put('power_limit_w',
                pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
        except Exception:
            put('power_limit_w', 0)

        # -- Clocks ----------------------------------------------------
        try:
            put('clock_graphics_mhz',
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
            put('clock_memory_mhz',
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        except Exception:
            put('clock_graphics_mhz', 0)
            put('clock_memory_mhz',   0)

        # -- PCIe link topology (read first -- needed for BW ceiling) --
        try:
            pcie_gen   = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            put('pcie_gen',       pcie_gen)
            put('pcie_width',     pcie_width)
            put('pcie_max_gen',   pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
            put('pcie_max_width', pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
        except Exception:
            pcie_gen   = 3   # conservative fallback for ceiling calculation
            pcie_width = 16
            put('pcie_gen',       0)
            put('pcie_width',     0)
            put('pcie_max_gen',   0)
            put('pcie_max_width', 0)

        # -- PCIe throughput -------------------------------------------
        # nvmlDeviceGetPcieThroughput -> KB/s  (divide by 1024 -> MB/s)
        # Full-link ceiling = per_lane_MB_s x actual_width
        try:
            tx_kbs = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            rx_kbs = pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)

            tx_mb_s = tx_kbs / 1024.0
            rx_mb_s = rx_kbs / 1024.0

            per_lane          = PCIE_PER_LANE_MB_S.get(pcie_gen, 985)
            link_ceiling_mb_s = per_lane * max(pcie_width, 1)

            # Warn per direction, per GPU; raw value is always kept in CSV
            for label, val in (('TX', tx_mb_s), ('RX', rx_mb_s)):
                if link_ceiling_mb_s and val > link_ceiling_mb_s * 1.05:
                    print(
                        f"\n[Monitor][GPU {gpu_id}][PCIe] {label} {val:.1f} MB/s "
                        f"exceeds Gen{pcie_gen} x{pcie_width} ceiling "
                        f"{link_ceiling_mb_s:.0f} MB/s -- "
                        f"possible NVML sampling artefact; raw value kept."
                    )

            put('pcie_tx_mb_s',           round(tx_mb_s, 2))
            put('pcie_rx_mb_s',           round(rx_mb_s, 2))
            put('pcie_link_ceiling_mb_s', round(link_ceiling_mb_s, 0))
            put('pcie_tx_util_percent',   round(
                min(tx_mb_s / link_ceiling_mb_s * 100, 100.0), 2)
                if link_ceiling_mb_s else 0.0)
            put('pcie_rx_util_percent',   round(
                min(rx_mb_s / link_ceiling_mb_s * 100, 100.0), 2)
                if link_ceiling_mb_s else 0.0)
        except Exception:
            put('pcie_tx_mb_s',           0)
            put('pcie_rx_mb_s',           0)
            put('pcie_link_ceiling_mb_s', 0)
            put('pcie_tx_util_percent',   0)
            put('pcie_rx_util_percent',   0)

        # -- NUMA affinity (from topology map built at startup) --------
        put('numa_node',
            self.cpu_gpu_topology['numa_nodes'].get(gpu_id, 0))

        return metrics

    def _get_all_gpu_metrics(self) -> Dict:
        """
        Query every GPU, assemble per-GPU indexed columns, then compute
        system-wide summary columns (averages / sums).

        Returns a flat dict ready to .update() into the CSV row.
        """
        if not self.gpu_available:
            return {}

        all_metrics: Dict = {'gpu_count': self.gpu_count}
        per_gpu: List[Dict] = []

        for g in range(self.gpu_count):
            gm = self._get_single_gpu_metrics(g)
            per_gpu.append(gm)
            all_metrics.update(gm)

        # System-wide summary columns
        def gpu_val(g: int, field: str) -> float:
            return float(per_gpu[g].get(f'gpu{g}_{field}', 0) or 0)

        n = self.gpu_count
        all_metrics['gpu_system_util_avg_percent']  = (
            sum(gpu_val(g, 'util_percent')   for g in range(n)) / n)
        all_metrics['gpu_system_power_total_w']      = (
            sum(gpu_val(g, 'power_draw_w')   for g in range(n)))
        all_metrics['gpu_system_memory_used_gb']     = (
            sum(gpu_val(g, 'memory_used_gb') for g in range(n)))
        all_metrics['gpu_system_memory_total_gb']    = (
            sum(gpu_val(g, 'memory_total_gb') for g in range(n)))
        all_metrics['gpu_system_pcie_tx_total_mb_s'] = (
            sum(gpu_val(g, 'pcie_tx_mb_s')   for g in range(n)))
        all_metrics['gpu_system_pcie_rx_total_mb_s'] = (
            sum(gpu_val(g, 'pcie_rx_mb_s')   for g in range(n)))

        return all_metrics

    # ------------------------------------------------------------------
    # Per-GPU affinity detection
    # ------------------------------------------------------------------

    def _detect_gpu_active_cores(
        self,
        cpu_per_core: List[float],
        gpu_id: int,
        gpu_util: float,
    ) -> Tuple[str, str, float]:
        """
        For a single GPU, update its rolling history and return
        (hex_mask, comma_cores, correlation).

        Design rationale
        ----------------
        Each GPU has its own rolling deque (self.gpu_history[gpu_id]) and
        its own affinity log (self.affinity_log[gpu_id]).  This lets the
        end-of-run summary report which CPU cores were active while *that*
        specific GPU was busy, which matters for multi-socket / cross-NUMA
        configs where GPU 0 and GPU 1 may own entirely different CPU cores.

        Correlation metric: magnitude of GPU utilisation change over the last
        5 samples.  This is a cheap heuristic -- a rising GPU tends to cause
        rising CPU activity on its DMA-handling cores shortly after.
        """
        history = self.gpu_history[gpu_id]
        history.append(gpu_util)
        self.affinity_log[gpu_id].append((gpu_util, list(cpu_per_core)))

        # Update shared per-core rolling history (used by live display)
        for i, val in enumerate(cpu_per_core):
            self.core_history[i].append(val)

        if len(history) < 3:
            return '0x0', '', 0.0

        # Identify the 4 CPU cores with highest utilisation while GPU is busy.
        # Only populate when GPU util > 50 % so idle noise doesn't corrupt
        # the affinity map.
        active_cores: List[int] = []
        if gpu_util > 50:
            sorted_cores = sorted(enumerate(cpu_per_core),
                                  key=lambda x: x[1], reverse=True)
            active_cores = [c[0] for c in sorted_cores[:4]]

        mask      = sum(1 << c for c in active_cores)
        mask_str  = f'0x{mask:08x}'
        cores_str = ','.join(map(str, active_cores))

        # Magnitude-of-change correlation
        correlation = 0.0
        if len(history) >= 5:
            gpu_delta   = history[-1] - history[-5]
            correlation = min(abs(gpu_delta), 100.0)

        return mask_str, cores_str, correlation

    # ------------------------------------------------------------------
    # CSV writing
    # ------------------------------------------------------------------

    def _write_sample(self, elapsed: float):
        """Collect all subsystem metrics and append one row to the CSV."""
        timestamp = datetime.now().isoformat()

        cpu_metrics  = self._get_detailed_cpu_metrics()
        mem_metrics  = self._get_detailed_memory_metrics()
        disk_metrics = self._get_detailed_disk_metrics()
        net_metrics  = self._get_detailed_network_metrics()
        gpu_metrics  = self._get_all_gpu_metrics()

        # Run per-GPU affinity detection and collect results
        affinity_fields: Dict = {}
        if self.gpu_available:
            for g in range(self.gpu_count):
                g_util = float(gpu_metrics.get(f'gpu{g}_util_percent', 0) or 0)
                mask_str, cores_str, correlation = self._detect_gpu_active_cores(
                    cpu_metrics['cpu_per_core'], g, g_util)
                affinity_fields[f'gpu{g}_active_cores_mask']   = mask_str
                affinity_fields[f'gpu{g}_data_transfer_cores'] = cores_str
                affinity_fields[f'gpu{g}_cpu_correlation']     = round(correlation, 2)

        # Assemble the full row dict
        row: Dict = {
            'timestamp':          timestamp,
            'elapsed_seconds':    round(elapsed, 3),
            'sample_number':      self.samples,
            'cpu_system_percent': cpu_metrics['cpu_system_percent'],
        }

        for i, val in enumerate(cpu_metrics['cpu_per_core']):
            row[f'cpu_core_{i}_percent'] = val

        row.update({
            'cpu_frequency_mhz':   cpu_metrics['cpu_frequency_mhz'],
            'cpu_ctx_switches':    cpu_metrics['cpu_ctx_switches'],
            'cpu_interrupts':      cpu_metrics['cpu_interrupts'],
            'cpu_soft_interrupts': cpu_metrics['cpu_soft_interrupts'],
            'cpu_syscalls':        cpu_metrics['cpu_syscalls'],
        })

        row.update(mem_metrics)
        row.update(disk_metrics)
        row.update(net_metrics)
        row.update(gpu_metrics)

        row.update({
            'cpu_package_temp_c':  cpu_metrics['cpu_package_temp_c'],
            'cpu_core_temp_avg_c': cpu_metrics['cpu_core_temp_avg_c'],
            'cpu_core_temp_max_c': cpu_metrics['cpu_core_temp_max_c'],
        })
        for i in range(min(self.num_cpus, 28)):
            row[f'cpu_core_{i}_temp_c'] = cpu_metrics[f'cpu_core_{i}_temp_c']

        row.update(affinity_fields)

        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_headers,
                                    extrasaction='ignore')
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Live display
    # ------------------------------------------------------------------

    def _print_live_stats(self):
        """
        Print one CPU summary line followed by one line per GPU.
        Uses ANSI cursor-up to overwrite in place after the first sample
        so the display doesn't scroll.
        """
        if not self.core_history[0]:
            return

        cpu_avg = (sum(self.core_history[i][-1]
                       for i in range(self.num_cpus)) / self.num_cpus)

        active_cores = sorted(
            [(i, self.core_history[i][-1]) for i in range(self.num_cpus)],
            key=lambda x: x[1], reverse=True
        )[:4]
        cores_str = ','.join(f'{c[0]}:{c[1]:.0f}%' for c in active_cores)

        # After the first printed block, jump back up to overwrite
        total_lines = self.gpu_count + 1   # 1 CPU line + N GPU lines
        if self.samples > 1:
            print(f'\x1b[{total_lines}A', end='')

        print(f'[{self.samples:5d}] CPU avg:{cpu_avg:5.1f}%  '
              f'top cores:[{cores_str}]' + ' ' * 10)

        for g in range(self.gpu_count):
            gh     = self.gpu_history[g]
            g_util = gh[-1] if gh else 0.0
            print(f'         GPU {g}: util={g_util:5.1f}%' + ' ' * 20)

        import sys
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Signal handling & main loop
    # ------------------------------------------------------------------

    def _signal_handler(self, signum, frame):
        self.running = False

    def run(self):
        self.running = True
        start_time   = time.time()
        last_print   = 0.0

        # Prime psutil -- first call always returns 0.0 per the psutil contract
        psutil.cpu_percent(interval=None, percpu=True)

        print('\n[Monitor] Started -- Press Ctrl+C to stop\n')

        while self.running:
            elapsed = time.time() - start_time
            if self.duration and elapsed >= self.duration:
                print(f'\n[Monitor] Duration reached ({self.duration}s)')
                break

            self._write_sample(elapsed)
            self.samples += 1

            if elapsed - last_print >= 1.0:
                self._print_live_stats()
                last_print = elapsed

            time.sleep(self.interval)

        self._cleanup()

    # ------------------------------------------------------------------
    # Summary / cleanup
    # ------------------------------------------------------------------

    def _cleanup(self):
        print('\n')
        print('=' * 70)
        print('MONITORING SUMMARY')
        print('=' * 70)
        print(f'Samples : {self.samples}')
        print(f'Output  : {os.path.abspath(self.output_file)}')

        if self.show_affinity and self.gpu_available:
            for g in range(self.gpu_count):
                log = self.affinity_log[g]
                if not log:
                    continue

                print(f'\nGPU {g} -- CPU Affinity Analysis:')

                high_gpu_samples = [
                    (gpu_u, cores) for gpu_u, cores in log if gpu_u > 50
                ]

                if not high_gpu_samples:
                    print('  No high-utilisation periods (>50%) recorded.')
                    continue

                print(f'  High-utilisation samples : {len(high_gpu_samples)}')

                # Average per-core activity across all high-GPU samples
                core_sums: Dict[int, float] = defaultdict(float)
                for _, cores in high_gpu_samples:
                    for core_idx, val in enumerate(cores):
                        core_sums[core_idx] += val

                avg_during_gpu = sorted(
                    [(core, total / len(high_gpu_samples))
                     for core, total in core_sums.items()],
                    key=lambda x: x[1], reverse=True
                )

                print(f'\n  Top 5 CPU cores while GPU {g} was busy:')
                for core, avg in avg_during_gpu[:5]:
                    bar = chr(9608) * int(avg / 5)
                    print(f'    Core {core:2d}: {avg:5.1f}%  {bar}')

                data_cores  = [c[0] for c in avg_during_gpu[:4]]
                numa_cores  = self.cpu_gpu_topology['gpu_cores'].get(g, [])
                print(f'\n  Empirical DMA cores  : {data_cores}')
                if numa_cores:
                    print(f'  NUMA-local cores     : {numa_cores[0]}-{numa_cores[-1]}')

        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        print('=' * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Advanced System Monitor with multi-GPU and CPU-affinity support'
    )
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('-d', '--duration', type=float, default=None,
                        help='Stop after N seconds (default: run until Ctrl+C)')
    args = parser.parse_args()

    monitor = AdvancedSystemMonitor(
        output_file=args.output,
        interval=args.interval,
        duration=args.duration,
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print('\n[Monitor] Interrupted')
        monitor._cleanup()


if __name__ == '__main__':
    main()