"""
DistributedClusterManager (agents/distributed_cluster.py)

Multi-Mac Distributed Edge Compute Connector.
Allows primary Mac to discover and distribute image diffusion batches,
video frame chunks, and model fine-tuning jobs across peer Macs on the local network (e.g. MacBook Air).
"""

import concurrent.futures
import logging
import os
import requests
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DistributedClusterManager:
    """
    Manages multi-Mac distributed compute nodes on local network.
    """

    def __init__(self, primary_node_url: str = "http://localhost:5005"):
        self.primary_node_url = primary_node_url
        # "unknown" until health_check() actually reaches the node. A hardcoded
        # "online" would report a healthy cluster before anything was contacted.
        self.nodes: List[Dict[str, Any]] = [
            {
                "node_id": "primary",
                "url": primary_node_url,
                "role": "primary",
                "status": "unknown",
                "device": None,
            }
        ]

    def register_node(self, node_id: str, url: str, role: str = "worker") -> Dict[str, Any]:
        """Register a new peer Mac worker node on the local network."""
        clean_url = url.rstrip("/")
        # Check if already registered
        for n in self.nodes:
            if n["url"] == clean_url or n["node_id"] == node_id:
                # Registration is a claim, not a health check; get_active_nodes
                # is what establishes reachability.
                n["status"] = "registered"
                n["url"] = clean_url
                return {"success": True, "message": f"Node '{node_id}' updated", "nodes": self.nodes}

        node_info = {
            "node_id": node_id,
            "url": clean_url,
            "role": role,
            "status": "registered",
            "device": None,
        }
        self.nodes.append(node_info)
        return {"success": True, "message": f"Registered node '{node_id}' at {clean_url}", "nodes": self.nodes}

    def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Health check all registered cluster nodes and return online nodes."""
        active = []
        for n in self.nodes:
            try:
                resp = requests.get(f"{n['url']}/", timeout=1.5)
                if resp.status_code == 200:
                    n["status"] = "online"
                    active.append(n)
                else:
                    n["status"] = "unreachable"
            except Exception:
                n["status"] = "offline"
        return active

    def distribute_batch_generation(
        self,
        prompts: List[str],
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Distribute a batch of text-to-image prompts across all available cluster nodes in parallel.
        """
        start_time = time.time()
        active_nodes = self.get_active_nodes()
        results = []

        def worker_job(prompt_idx_tuple):
            idx, prompt = prompt_idx_tuple
            target_node = active_nodes[idx % len(active_nodes)]
            node_url = target_node["url"]

            if target_node["role"] == "primary" or "localhost" in node_url or "127.0.0.1" in node_url:
                # Run locally via PyTorch MPS
                from .diffusion_engine import PyTorchDiffusionEngine
                engine = PyTorchDiffusionEngine()
                res = engine.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    raw_mode=True,
                )
                res["node_assigned"] = target_node["node_id"]
                return res
            else:
                # Dispatch HTTP request to remote MacBook Air node
                try:
                    payload = {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "raw_mode": True,
                    }
                    resp = requests.post(f"{node_url}/api/image/diffusion", json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        data["node_assigned"] = target_node["node_id"]
                        return data
                except Exception as e:
                    logger.warning(f"Remote cluster node '{target_node['node_id']}' failed: {e}")

                # Fallback to local execution
                from .diffusion_engine import PyTorchDiffusionEngine
                engine = PyTorchDiffusionEngine()
                res = engine.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    raw_mode=True,
                )
                res["node_assigned"] = f"{target_node['node_id']} (fallback to primary)"
                return res

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(2, len(active_nodes))) as executor:
            job_tuples = list(enumerate(prompts))
            futures = [executor.submit(worker_job, t) for t in job_tuples]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as ex:
                    results.append({"success": False, "error": str(ex)})

        return {
            "success": True,
            "total_prompts": len(prompts),
            "cluster_nodes_utilized": len(active_nodes),
            "elapsed_seconds": round(time.time() - start_time, 2),
            "batch_results": results,
        }
