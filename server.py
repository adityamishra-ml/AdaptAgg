# server.py - ENHANCED WITH WEIGHT SAVING
import flwr as fl
from flwr.common import Metrics, Parameters, NDArrays
from flwr.server import ServerConfig
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
FIXED_CLASS_ORDER = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']
SERVER_WEIGHTS_DIR = Path("server_weights")
SERVER_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# METRICS AGGREGATION
# ============================================================================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics from all clients using weighted averaging.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from each client
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics:
        return {}
    
    # Calculate total number of examples used for evaluation
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    if total_examples == 0:
        return {}
    
    # Calculate weighted average for each metric
    map50 = sum([m["map50"] * num_examples for num_examples, m in metrics]) / total_examples
    map50_95 = sum([m["map50-95"] * num_examples for num_examples, m in metrics]) / total_examples
    precision = sum([m["precision"] * num_examples for num_examples, m in metrics]) / total_examples
    recall = sum([m["recall"] * num_examples for num_examples, m in metrics]) / total_examples

    # Return aggregated metrics
    aggregated = {
        "map50": map50,
        "map50-95": map50_95,
        "precision": precision,
        "recall": recall
    }
    
    return aggregated

# ============================================================================
# CUSTOM STRATEGY WITH WEIGHT SAVING
# ============================================================================

class SaveWeightsFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that saves global model weights after each round.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = []  # Track metrics over rounds
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Aggregate model weights from clients and save the global model.
        """
        print(f"\n{'='*70}")
        print(f"🔄 SERVER: Round {server_round} - Aggregating {len(results)} client models...")
        
        # Call parent's aggregate_fit to perform FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # === SAVE AGGREGATED WEIGHTS ===
            self._save_global_weights(aggregated_parameters, server_round)
            print(f"✅ SERVER: Round {server_round} aggregation complete!")
        
        print(f"{'='*70}\n")
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List[Tuple],
    ) -> Tuple[Optional[float], Dict]:
        """
        Aggregate evaluation metrics and save them.
        """
        print(f"\n📊 SERVER: Round {server_round} - Aggregating evaluation metrics...")
        
        # Call parent's aggregate_evaluate
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if metrics_aggregated:
            # Save metrics
            self.metrics_history.append({
                "round": server_round,
                "loss": loss_aggregated,
                **metrics_aggregated
            })
            
            # Print aggregated metrics
            print(f"📈 SERVER: Round {server_round} Global Metrics:")
            print(f"   Loss: {loss_aggregated:.4f}")
            print(f"   mAP50: {metrics_aggregated.get('map50', 0):.4f}")
            print(f"   mAP50-95: {metrics_aggregated.get('map50-95', 0):.4f}")
            print(f"   Precision: {metrics_aggregated.get('precision', 0):.4f}")
            print(f"   Recall: {metrics_aggregated.get('recall', 0):.4f}")
            
            # Save metrics to JSON
            metrics_file = SERVER_WEIGHTS_DIR / "metrics_history.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            print(f"💾 Metrics saved to {metrics_file}")
        
        return loss_aggregated, metrics_aggregated
    
    def _save_global_weights(self, parameters: Parameters, round_num: int):
        """
        Save the aggregated global model weights.
        
        Args:
            parameters: Aggregated parameters from FedAvg
            round_num: Current federated learning round
        """
        try:
            # Convert Parameters to list of numpy arrays
            weights: NDArrays = fl.common.parameters_to_ndarrays(parameters)
            
            # Convert to PyTorch tensors and save
            weights_dict = {f"layer_{i}": torch.from_numpy(w) for i, w in enumerate(weights)}
            
            save_path = SERVER_WEIGHTS_DIR / f"global_model_round_{round_num}.pt"
            torch.save(weights_dict, save_path)
            
            print(f"💾 SERVER: Global model weights saved to {save_path}")
            
        except Exception as e:
            print(f"❌ SERVER: Error saving weights: {e}")

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def fit_config(server_round: int):
    """
    Send configuration to clients for training.
    Includes current round number and total rounds.
    """
    return {
        "round": server_round,
        "num_rounds": 5  # Match your ServerConfig
    }

def evaluate_config(server_round: int):
    """
    Send configuration to clients for evaluation.
    """
    return {
        "round": server_round
    }

# ============================================================================
# MAIN SERVER SETUP
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Flower server with weight tracking...")
    print(f"📁 Weights will be saved to: {SERVER_WEIGHTS_DIR.absolute()}")
    
    # Initialize strategy
    strategy = SaveWeightsFedAvg(
        fraction_fit=1.0,              # Use 100% of available clients for training
        fraction_evaluate=1.0,         # Use 100% of available clients for evaluation
        min_fit_clients=6,             # Minimum clients needed for training
        min_evaluate_clients=6,        # Minimum clients needed for evaluation
        min_available_clients=6,       # Minimum clients that must connect
        evaluate_metrics_aggregation_fn=weighted_average,  # Custom metric aggregation
        on_fit_config_fn=fit_config,   # Send config to clients during training
        on_evaluate_config_fn=evaluate_config,  # Send config during evaluation
    )
    
    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=5),  # Total federated learning rounds
        strategy=strategy
    )
    
    print("\n🎉 Federated learning complete!")
    print(f"📊 Final metrics saved in: {SERVER_WEIGHTS_DIR / 'metrics_history.json'}")
    print(f"💾 Global model weights saved for each round in: {SERVER_WEIGHTS_DIR}")