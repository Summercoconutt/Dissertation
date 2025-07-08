# Extension Helpers for DAO Governance Simulation System
# This module provides helper functions and templates for easily extending the system

from typing import List, Dict, Callable, Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel

# ===================================================================
# 🔧 VOTING MECHANISM HELPERS
# ===================================================================

class VotingMechanismTemplate:
    """Template for creating new voting mechanisms"""
    
    @staticmethod
    def create_mechanism(name: str, 
                        power_function: Callable[[List[float]], List[float]],
                        requires_reputation: bool = False,
                        description: str = ""):
        """
        Create a new voting mechanism function
        
        Args:
            name: Name of the mechanism
            power_function: Function that transforms input values to voting power
            requires_reputation: Whether this mechanism needs reputation data
            description: Description of how the mechanism works
            
        Returns:
            Dictionary with mechanism metadata and function
        """
        return {
            "name": name,
            "function": power_function,
            "requires_reputation": requires_reputation,
            "description": description
        }

# Pre-built mechanism templates
def linear_cap_voting(tokens: List[float], cap: float = 1000.0) -> List[float]:
    """Linear voting with a cap - prevents whale dominance"""
    return [min(t, cap) for t in tokens]

def logarithmic_voting(tokens: List[float], base: float = 2.0) -> List[float]:
    """Logarithmic voting - diminishing returns"""
    return [np.log(1 + t) / np.log(base) for t in tokens]

def stake_weighted_quadratic(tokens: List[float], reputations: List[float], 
                           stake_weight: float = 0.7) -> List[float]:
    """Combines stake and quadratic voting with custom weighting"""
    qv_power = [np.sqrt(t) for t in tokens]
    rep_power = [np.log1p(r) for r in reputations]
    return [stake_weight * q + (1 - stake_weight) * r 
            for q, r in zip(qv_power, rep_power)]

# Registry for custom mechanisms
CUSTOM_MECHANISMS = {}

def register_mechanism(name: str, mechanism_dict: Dict):
    """Register a new voting mechanism"""
    CUSTOM_MECHANISMS[name] = mechanism_dict
    print(f"✅ Registered new mechanism: {name}")

# ===================================================================
# 🤖 AGENT HELPERS
# ===================================================================

class AgentTemplate(BaseModel, ABC):
    """Abstract base class for creating new agent types"""
    
    @abstractmethod
    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare training data from raw DataFrame"""
        pass
    
    @abstractmethod
    def train(self):
        """Train the agent model"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the trained model"""
        pass

class SimpleRuleBasedAgent(AgentTemplate):
    """Template for rule-based agents (no ML training needed)"""
    
    def __init__(self, rules: Dict[str, Any]):
        super().__init__()
        self.rules = rules
        self.is_trained = True
    
    def prepare_training_data(self, df: pd.DataFrame):
        """Rule-based agents don't need training data"""
        pass
    
    def train(self):
        """Rule-based agents don't need training"""
        pass
    
    def predict(self, input_data: Any) -> Dict:
        """Apply rules to make predictions"""
        # Implement your rule logic here
        return {"predicted_vote": 0, "confidence": 1.0}

# ===================================================================
# 📊 METRICS HELPERS
# ===================================================================

class MetricTemplate:
    """Template for creating new evaluation metrics"""
    
    @staticmethod
    def create_metric(name: str,
                     calculation_function: Callable,
                     description: str = "",
                     optimal_range: tuple = (0, 1),
                     higher_is_better: bool = True):
        """
        Create a new evaluation metric
        
        Args:
            name: Name of the metric
            calculation_function: Function that calculates the metric
            description: What this metric measures
            optimal_range: Expected range of values (min, max)
            higher_is_better: Whether higher values are better
            
        Returns:
            Dictionary with metric metadata and function
        """
        return {
            "name": name,
            "function": calculation_function,
            "description": description,
            "optimal_range": optimal_range,
            "higher_is_better": higher_is_better
        }

# Example custom metrics
def calculate_decentralization_index(voting_power: List[float]) -> float:
    """Measure how decentralized voting power is (inverse of concentration)"""
    from evaluation import calculate_hhi
    hhi = calculate_hhi(voting_power)
    return 1 - hhi  # Higher = more decentralized

def calculate_effective_voters(voting_power: List[float], threshold: float = 0.01) -> float:
    """Count voters with meaningful voting power (above threshold)"""
    total_power = sum(voting_power)
    effective = sum(1 for power in voting_power if power/total_power >= threshold)
    return effective / len(voting_power)

def calculate_whale_influence(voting_power: List[float], whale_threshold: float = 0.1) -> float:
    """Measure influence of whale voters (top holders above threshold)"""
    total_power = sum(voting_power)
    whale_power = sum(power for power in voting_power 
                     if power/total_power >= whale_threshold)
    return whale_power / total_power

# Registry for custom metrics
CUSTOM_METRICS = {}

def register_metric(name: str, metric_dict: Dict):
    """Register a new evaluation metric"""
    CUSTOM_METRICS[name] = metric_dict
    print(f"✅ Registered new metric: {name}")

# ===================================================================
# 🔄 SIMULATION HELPERS
# ===================================================================

class SimulationBuilder:
    """Helper class to build custom simulation configurations"""
    
    def __init__(self):
        self.config = {
            "mechanisms": ["token", "quadratic", "reputation", "hybrid"],
            "participation_rates": [0.3],
            "num_rounds": 100,
            "custom_mechanisms": {},
            "custom_metrics": {},
            "output_dir": "./results/"
        }
    
    def add_mechanism(self, name: str, mechanism_dict: Dict):
        """Add a custom mechanism to the simulation"""
        self.config["custom_mechanisms"][name] = mechanism_dict
        if name not in self.config["mechanisms"]:
            self.config["mechanisms"].append(name)
        return self
    
    def add_metric(self, name: str, metric_dict: Dict):
        """Add a custom metric to the simulation"""
        self.config["custom_metrics"][name] = metric_dict
        return self
    
    def set_participation_rates(self, rates: List[float]):
        """Set different participation rates to test"""
        self.config["participation_rates"] = rates
        return self
    
    def set_rounds(self, num_rounds: int):
        """Set number of simulation rounds"""
        self.config["num_rounds"] = num_rounds
        return self
    
    def get_config(self) -> Dict:
        """Get the final configuration"""
        return self.config

# ===================================================================
# 📈 VISUALIZATION HELPERS
# ===================================================================

def create_comparison_dataset(results: pd.DataFrame, 
                            metrics: List[str] = None) -> pd.DataFrame:
    """Prepare results for visualization"""
    if metrics is None:
        metrics = ['gini', 'hhi', 'participation', 'utility']
    
    # Normalize metrics to 0-1 scale for better comparison
    normalized_results = results.copy()
    
    for metric in metrics:
        if metric in results.columns:
            min_val = results[metric].min()
            max_val = results[metric].max()
            if max_val > min_val:
                normalized_results[f"{metric}_normalized"] = (
                    (results[metric] - min_val) / (max_val - min_val)
                )
            else:
                normalized_results[f"{metric}_normalized"] = 0.5
    
    return normalized_results

# ===================================================================
# 🚀 QUICK START TEMPLATES
# ===================================================================

def quick_add_mechanism_example():
    """Example of how to quickly add a new voting mechanism"""
    
    # Define your mechanism function
    def capped_quadratic_voting(tokens: List[float], cap: float = 10000.0) -> List[float]:
        """Quadratic voting with a cap to prevent extreme whale influence"""
        capped_tokens = [min(t, cap) for t in tokens]
        return [np.sqrt(t) for t in capped_tokens]
    
    # Create mechanism metadata
    mechanism = VotingMechanismTemplate.create_mechanism(
        name="capped_quadratic",
        power_function=lambda tokens: capped_quadratic_voting(tokens, cap=10000.0),
        requires_reputation=False,
        description="Quadratic voting with token cap to reduce whale influence"
    )
    
    # Register it
    register_mechanism("capped_quadratic", mechanism)
    
    return mechanism

def quick_add_metric_example():
    """Example of how to quickly add a new evaluation metric"""
    
    # Create metric metadata
    metric = MetricTemplate.create_metric(
        name="decentralization_index",
        calculation_function=calculate_decentralization_index,
        description="Measures how decentralized voting power is (1 - HHI)",
        optimal_range=(0, 1),
        higher_is_better=True
    )
    
    # Register it
    register_metric("decentralization_index", metric)
    
    return metric

def create_voter_pool_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to create voter pool from raw data"""
    
    # Calculate reputation as number of votes cast
    df['reputation'] = df.groupby('voter_address')['vote_choice'].transform('count')
    
    # Create voting history lists
    vote_map = {'for': 0, 'against': 1, 'abstain': 2, 'other': 3}
    df['vote_label'] = df['vote_choice'].map(vote_map)
    
    voter_pool = df.groupby('voter_address').agg({
        'token_amount': 'mean',  # Average tokens across votes
        'reputation': 'first',   # Vote count (same for all rows per user)
        'vote_label': lambda x: list(x)  # Voting history
    }).reset_index()
    
    voter_pool.rename(columns={'vote_label': 'history'}, inplace=True)
    
    return voter_pool

# ===================================================================
# 🎯 USAGE EXAMPLES
# ===================================================================

def example_usage():
    """Complete example of extending the system"""
    
    print("🚀 DAO Simulation Extension Example")
    print("=" * 50)
    
    # 1. Add a new voting mechanism
    print("\n1. Adding new voting mechanism...")
    new_mechanism = quick_add_mechanism_example()
    
    # 2. Add a new evaluation metric  
    print("\n2. Adding new evaluation metric...")
    new_metric = quick_add_metric_example()
    
    # 3. Build a custom simulation
    print("\n3. Building custom simulation...")
    sim_builder = SimulationBuilder()
    sim_config = (sim_builder
                  .add_mechanism("capped_quadratic", new_mechanism)
                  .add_metric("decentralization_index", new_metric)
                  .set_participation_rates([0.2, 0.3, 0.5])
                  .set_rounds(50)
                  .get_config())
    
    print("✅ Custom simulation configuration created:")
    for key, value in sim_config.items():
        print(f"   {key}: {value}")
    
    return sim_config

if __name__ == "__main__":
    example_usage()