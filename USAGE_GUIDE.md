# 🚀 DAO Governance Simulation - Complete Usage Guide

This guide shows you how to use the refined DAO governance simulation system with all its components and helper functions.

## 🎯 Quick Start (5 Minutes)

### 1. Basic Simulation Run

```python
# Run the basic simulation with default settings
from simulator import main

# This will:
# - Load data from merged_snapshot.csv
# - Train the UserBehaviorAgent
# - Run simulations for all 4 mechanisms
# - Display results
main()
```

### 2. View Results Visualization

```python
from plots import plot_bar_comparison, plot_radar_chart
import pandas as pd

# Load your results (after running simulation)
results = pd.read_csv("results.csv")  # Or use the results from main()

# Create visualizations
plot_bar_comparison(results, ["gini", "hhi", "participation", "utility"])
plot_radar_chart(results, ["gini", "hhi", "participation", "utility"])
```

---

## 🔧 Customizing the System

### Adding a New Voting Mechanism

```python
from extension_helpers import VotingMechanismTemplate, register_mechanism
import numpy as np

# Step 1: Define your mechanism function
def conviction_voting(tokens, convictions, decay_rate=0.9):
    """Voting power based on conviction (how long tokens are locked)"""
    conviction_weights = [decay_rate ** (max_time - t) for t in convictions]
    return [t * w for t, w in zip(tokens, conviction_weights)]

# Step 2: Create mechanism metadata
conviction_mechanism = VotingMechanismTemplate.create_mechanism(
    name="conviction",
    power_function=lambda tokens, **kwargs: conviction_voting(
        tokens, kwargs.get('convictions', [1]*len(tokens))
    ),
    requires_reputation=False,
    description="Voting power increases with conviction (lock time)"
)

# Step 3: Register the mechanism
register_mechanism("conviction", conviction_mechanism)

# Step 4: Update mechanism_simulation.py to include your mechanism
# Add this to the simulate_proposal() function:
# elif mechanism == "conviction":
#     power = conviction_voting(tokens, convictions)
```

### Adding a New Evaluation Metric

```python
from extension_helpers import MetricTemplate, register_metric

# Step 1: Define your metric function
def calculate_voter_diversity(voting_power, vote_choices):
    """Measure how diverse the voting choices are"""
    from collections import Counter
    choice_counts = Counter(vote_choices)
    total_votes = len(vote_choices)
    
    # Shannon diversity index
    diversity = 0
    for count in choice_counts.values():
        if count > 0:
            p = count / total_votes
            diversity -= p * np.log(p)
    
    return diversity

# Step 2: Create metric metadata
diversity_metric = MetricTemplate.create_metric(
    name="voter_diversity",
    calculation_function=calculate_voter_diversity,
    description="Shannon diversity of voting choices",
    optimal_range=(0, 2),  # log(4) for 4 vote choices
    higher_is_better=True
)

# Step 3: Register the metric
register_metric("voter_diversity", diversity_metric)
```

### Creating a Custom Agent

```python
from extension_helpers import AgentTemplate
from agent_base import UserVoteHistory, VotePrediction
import random

class RandomAgent(AgentTemplate):
    """Simple random voting agent for testing"""
    
    def __init__(self, vote_probabilities=None):
        super().__init__()
        self.vote_probabilities = vote_probabilities or [0.4, 0.3, 0.2, 0.1]
        self.is_trained = True
    
    def prepare_training_data(self, df):
        """Random agent doesn't need training data"""
        pass
    
    def train(self):
        """Random agent doesn't need training"""
        pass
    
    def predict(self, history: UserVoteHistory) -> VotePrediction:
        """Return random prediction based on probabilities"""
        vote = random.choices(
            range(4), 
            weights=self.vote_probabilities
        )[0]
        
        return VotePrediction(
            predicted_vote=vote,
            confidence=max(self.vote_probabilities)
        )

# Usage:
random_agent = RandomAgent(vote_probabilities=[0.5, 0.3, 0.1, 0.1])
```

---

## 🎛️ Building Custom Simulations

### Using the SimulationBuilder

```python
from extension_helpers import SimulationBuilder
from mechanism_simulation import run_simulation
from data_loader import load_snapshot_data
from agent_base import UserBehaviorAgent

# Step 1: Set up your simulation configuration
sim_builder = SimulationBuilder()

config = (sim_builder
    .add_mechanism("conviction", conviction_mechanism)
    .add_metric("voter_diversity", diversity_metric)
    .set_participation_rates([0.2, 0.3, 0.4, 0.5])  # Test different participation
    .set_rounds(200)  # More rounds for better accuracy
    .get_config())

# Step 2: Load data and train agent
df = load_snapshot_data("merged_snapshot.csv")
agent = UserBehaviorAgent(window_size=3)
agent.prepare_training_data(df)
agent.train()

# Step 3: Create voter pool
from extension_helpers import create_voter_pool_from_data
voter_pool = create_voter_pool_from_data(df)

# Step 4: Run custom simulation
results = []
for mechanism in config["mechanisms"]:
    for participation_rate in config["participation_rates"]:
        # Run simulation for this combination
        result = run_simulation(agent, voter_pool, mechanism, config["num_rounds"])
        result["participation_rate"] = participation_rate
        results.append(result)

final_results = pd.concat(results, ignore_index=True)
```

### Advanced Configuration Example

```python
# Complete custom simulation with multiple agents and mechanisms
def run_advanced_simulation():
    
    # 1. Load and prepare data
    df = load_snapshot_data("merged_snapshot.csv")
    voter_pool = create_voter_pool_from_data(df)
    
    # 2. Set up multiple agents
    ml_agent = UserBehaviorAgent(window_size=3)
    ml_agent.prepare_training_data(df)
    ml_agent.train()
    
    random_agent = RandomAgent()
    
    agents = {
        "ml_agent": ml_agent,
        "random_agent": random_agent
    }
    
    # 3. Set up multiple mechanisms
    mechanisms = {
        "token": "standard token voting",
        "quadratic": "quadratic voting",
        "conviction": "conviction voting (custom)",
        "capped_quadratic": "capped quadratic (custom)"
    }
    
    # 4. Run comprehensive comparison
    all_results = []
    
    for agent_name, agent in agents.items():
        for mechanism_name in mechanisms.keys():
            for participation in [0.2, 0.3, 0.5]:
                
                result = run_simulation(
                    agent, voter_pool, mechanism_name, num_rounds=100
                )
                
                result["agent_type"] = agent_name
                result["participation_rate"] = participation
                all_results.append(result)
    
    return pd.concat(all_results, ignore_index=True)

# Run the advanced simulation
advanced_results = run_advanced_simulation()
```

---

## 📊 Analysis and Insights

### Comparing Results

```python
# Group results by different dimensions
mechanism_comparison = advanced_results.groupby('mechanism').agg({
    'gini': 'mean',
    'hhi': 'mean', 
    'participation': 'mean',
    'utility': 'mean'
}).round(3)

agent_comparison = advanced_results.groupby('agent_type').agg({
    'gini': 'mean',
    'hhi': 'mean',
    'participation': 'mean', 
    'utility': 'mean'
}).round(3)

print("📊 Mechanism Performance:")
print(mechanism_comparison)

print("\n🤖 Agent Performance:")
print(agent_comparison)
```

### Statistical Analysis

```python
from scipy import stats

# Test if mechanism differences are statistically significant
token_utility = advanced_results[advanced_results['mechanism'] == 'token']['utility']
quadratic_utility = advanced_results[advanced_results['mechanism'] == 'quadratic']['utility']

t_stat, p_value = stats.ttest_ind(token_utility, quadratic_utility)
print(f"Token vs Quadratic utility difference: p={p_value:.4f}")

# Calculate effect sizes
def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size"""
    pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                         (len(group2) - 1) * group2.var()) / 
                         (len(group1) + len(group2) - 2))
    return (group1.mean() - group2.mean()) / pooled_std

effect_size = calculate_effect_size(token_utility, quadratic_utility)
print(f"Effect size (Cohen's d): {effect_size:.3f}")
```

### Advanced Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create multi-dimensional heatmap
pivot_data = advanced_results.pivot_table(
    values='utility',
    index='mechanism', 
    columns='participation_rate',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_data, annot=True, cmap='viridis', 
            cbar_kws={'label': 'Utility Score'})
plt.title('Utility Score by Mechanism and Participation Rate')
plt.tight_layout()
plt.savefig('utility_heatmap.png')
plt.show()

# Box plot comparison
plt.figure(figsize=(12, 8))
sns.boxplot(data=advanced_results, x='mechanism', y='utility', hue='agent_type')
plt.title('Utility Distribution by Mechanism and Agent Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('utility_boxplot.png')
plt.show()
```

---

## 🔍 Debugging and Validation

### Validating Your Extensions

```python
def validate_mechanism(mechanism_func, test_tokens):
    """Test if a voting mechanism behaves correctly"""
    
    try:
        # Test with sample data
        result = mechanism_func(test_tokens)
        
        # Basic validation
        assert len(result) == len(test_tokens), "Output length mismatch"
        assert all(r >= 0 for r in result), "Negative voting power"
        assert any(r > 0 for r in result), "No positive voting power"
        
        print("✅ Mechanism validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Mechanism validation failed: {e}")
        return False

# Test your custom mechanism
test_tokens = [100, 500, 1000, 50]
validate_mechanism(conviction_voting, test_tokens)
```

### Performance Monitoring

```python
import time

def benchmark_simulation(agent, voter_pool, mechanism, num_rounds=10):
    """Benchmark simulation performance"""
    
    start_time = time.time()
    
    results = run_simulation(agent, voter_pool, mechanism, num_rounds)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"⏱️  Simulation completed in {execution_time:.2f} seconds")
    print(f"📊 Average time per round: {execution_time/num_rounds:.3f} seconds")
    
    return results, execution_time

# Benchmark your simulation
benchmark_results, timing = benchmark_simulation(
    ml_agent, voter_pool, "quadratic", num_rounds=50
)
```

---

## 🎯 Best Practices

### 1. **Data Preparation**
- Always validate your CSV data format
- Check for missing values and outliers
- Ensure timestamp formatting is consistent

### 2. **Agent Training**
- Use sufficient training data (>1000 samples recommended)
- Validate agent predictions make sense
- Consider cross-validation for model selection

### 3. **Simulation Design**
- Run multiple rounds for statistical significance
- Test different participation rates
- Include sensitivity analysis

### 4. **Results Interpretation**
- Compare relative performance, not absolute values
- Look for statistically significant differences
- Consider practical significance vs statistical significance

### 5. **Extension Development**
- Test new components thoroughly
- Document your extensions clearly
- Follow the template patterns provided

---

## 🚀 Next Steps

### Advanced Features to Add

1. **Dynamic Mechanisms**: Mechanisms that adapt based on historical data
2. **Multi-Round Dependencies**: Proposals that depend on previous outcomes
3. **Network Effects**: Voter influence based on social networks
4. **Economic Models**: Token economics and staking mechanisms
5. **Real-time Integration**: Live data from actual DAOs

### Research Applications

1. **Governance Optimization**: Find optimal mechanism parameters
2. **Attack Analysis**: Simulate governance attacks and defenses
3. **Fairness Studies**: Deep dive into voting equality metrics
4. **Efficiency Analysis**: Optimize for decision-making speed

### System Scaling

1. **Parallel Processing**: Use multiprocessing for large simulations
2. **Database Integration**: Store results in proper databases
3. **Web Interface**: Create a web UI for easier interaction
4. **API Development**: Expose simulation as REST API

---

## 📚 Reference

- **System Architecture**: See `SYSTEM_LOGIC_FLOW.md`
- **Visual Flow Diagram**: See `VISUAL_FLOW_DIAGRAM.md`
- **Extension Helpers**: See `extension_helpers.py`
- **Core Components**: Individual `.py` files in the project

## 🆘 Getting Help

If you encounter issues:

1. Check the documentation files first
2. Validate your data format matches expected schema
3. Test components individually before integration
4. Use the validation functions provided in `extension_helpers.py`

Happy simulating! 🎉