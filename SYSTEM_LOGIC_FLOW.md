# DAO Governance Simulation - System Logic Flow

## 🎯 System Overview

This is a multi-agent DAO governance simulation system that uses machine learning to predict user voting behavior and evaluates different voting mechanisms for fairness and efficiency.

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │ ── │   Agent Layer    │ ── │ Simulation Layer│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Evaluation Layer│    │ Mechanism Layer  │    │Visualization Lyr│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🔄 Component Flow Diagram

### 1. Data Layer (`data_loader.py`)

**Purpose**: Load and preprocess Snapshot governance data

#### Input:
- `file_path`: String (default: "merged_snapshot.csv")
- CSV columns expected:
  - `voter_address`: String (wallet address)
  - `vote_choice`: String ('for', 'against', 'abstain', 'other')
  - `timestamp`: String/DateTime
  - `token_amount`: Float (voting tokens owned)

#### Output:
```python
pd.DataFrame:
    - voter_address: str
    - vote_choice: str  
    - timestamp: pd.DateTime
    - token_amount: float
```

#### Functions:
- `load_snapshot_data(file_path)` → DataFrame
- `get_user_profiles(df)` → DataFrame with aggregated user stats

---

### 2. Agent Layer (`agent_base.py`)

**Purpose**: Machine learning agent that predicts user voting behavior

#### Input Data Models:
```python
class UserVoteHistory(BaseModel):
    voter_address: str
    proposal_ids: List[str]  
    vote_labels: List[int]   # [0=for, 1=against, 2=abstain, 3=other]
```

#### Output Data Models:
```python
class VotePrediction(BaseModel):
    predicted_vote: int      # 0-3 (voting choice)
    confidence: float        # 0.0-1.0 (model confidence)
```

#### Agent Class: `UserBehaviorAgent`

**Configuration:**
- `window_size`: int = 3 (sliding window for behavior patterns)
- `model`: RandomForestClassifier (ML model)
- `label_map`: dict = {'for': 0, 'against': 1, 'abstain': 2, 'other': 3}

**Methods Flow:**

1. **`prepare_training_data(df: pd.DataFrame)`**
   - **Input**: Raw voting DataFrame
   - **Process**: 
     - Maps vote choices to numeric labels
     - Creates sliding windows of user behavior
     - Generates training samples: [history_window] → [next_vote]
   - **Output**: Sets `self.X` (features) and `self.y` (targets)

2. **`train()`**
   - **Input**: Uses prepared `self.X` and `self.y`
   - **Process**: Trains RandomForest with 80/20 train/test split
   - **Output**: Trained model + classification report

3. **`predict(history: UserVoteHistory)`**
   - **Input**: UserVoteHistory object
   - **Process**: Uses trained model to predict next vote
   - **Output**: VotePrediction object

---

### 3. Mechanism Layer (`mechanism_simulation.py`)

**Purpose**: Implements different voting mechanisms and their power calculations

#### Voting Mechanisms:

1. **`token_voting(tokens: List[float])`**
   - **Input**: List of token amounts
   - **Output**: Same list (1:1 token = 1 vote)

2. **`quadratic_voting(tokens: List[float])`**
   - **Input**: List of token amounts  
   - **Output**: List of √(token) values

3. **`reputation_voting(reputations: List[float])`**
   - **Input**: List of reputation scores
   - **Output**: List of log(1 + reputation) values

4. **`hybrid_voting(tokens, reputations, alpha=0.5)`**
   - **Input**: Token amounts + reputation scores + weighting factor
   - **Output**: Weighted combination: α×√(tokens) + (1-α)×log(1+reputation)

#### Single Simulation Function:

**`simulate_proposal(agent, mechanism, voter_pool, participation_rate=0.3)`**

**Input:**
- `agent`: Trained UserBehaviorAgent
- `mechanism`: str ('token', 'quadratic', 'reputation', 'hybrid')
- `voter_pool`: DataFrame with columns [voter_address, token_amount, reputation, history]
- `participation_rate`: float (0.0-1.0)

**Process Flow:**
1. Randomly sample voters from pool based on participation_rate
2. For each voter:
   - Extract recent voting history (window_size)
   - Use agent to predict next vote
   - Get voter's tokens/reputation
3. Apply chosen voting mechanism to calculate voting power
4. Determine proposal outcome (for_power vs against_power)
5. Calculate fairness metrics

**Output:**
```python
Dict:
    gini: float           # Gini coefficient (inequality)
    hhi: float            # Herfindahl-Hirschman Index (concentration)
    participation: float  # Actual participation rate
    passed: bool          # Whether proposal passed
    utility: float        # Combined utility score
```

---

### 4. Simulation Layer (`simulator.py`)

**Purpose**: Orchestrates the entire simulation workflow

#### Main Simulation Flow:

**`run_simulation(agent, voter_pool, mechanism, num_rounds=100)`**

**Input:**
- `agent`: Trained UserBehaviorAgent
- `voter_pool`: DataFrame with voter profiles
- `mechanism`: str (voting mechanism name)
- `num_rounds`: int (number of proposals to simulate)

**Process:**
1. Runs `simulate_proposal()` for specified rounds
2. Collects results from each round
3. Calculates average metrics across all rounds

**Output:**
```python
pd.DataFrame:
    mechanism: str
    gini: float       # Average Gini across rounds
    hhi: float        # Average HHI across rounds  
    participation: float  # Average participation
    passed: float     # Proportion of passed proposals
    utility: float    # Average utility score
```

#### Complete System Workflow:

```python
def main():
    # Step 1: Data Loading
    df = load_snapshot_data("merged_snapshot.csv")
    
    # Step 2: Agent Training
    agent = UserBehaviorAgent(window_size=3)
    agent.prepare_training_data(df)
    agent.train()
    
    # Step 3: Voter Pool Creation
    voter_pool = create_voter_pool(df)
    
    # Step 4: Multi-Mechanism Simulation
    results = []
    for mechanism in ["token", "quadratic", "reputation", "hybrid"]:
        summary = run_simulation(agent, voter_pool, mechanism, num_rounds=100)
        results.append(summary)
    
    # Step 5: Results Analysis
    final_results = pd.concat(results, ignore_index=True)
```

---

### 5. Evaluation Layer (`evaluation.py`)

**Purpose**: Calculate fairness and efficiency metrics

#### Metric Functions:

1. **`calculate_gini(voting_power: List[float])`**
   - **Input**: List of voting power values
   - **Output**: Gini coefficient (0.0-1.0, higher = more unequal)

2. **`calculate_hhi(voting_power: List[float])`**
   - **Input**: List of voting power values  
   - **Output**: HHI value (0.0-1.0, higher = more concentrated)

3. **`calculate_participation_rate(num_voters, total_users)`**
   - **Input**: Number of actual voters, total possible voters
   - **Output**: Participation rate (0.0-1.0)

4. **`calculate_pass_rate(proposal_results: List[bool])`**
   - **Input**: List of proposal outcomes
   - **Output**: Pass rate (0.0-1.0)

5. **`calculate_utility(gini, participation, alpha=1.0, beta=1.0)`**
   - **Input**: Gini coefficient, participation rate, weight parameters
   - **Formula**: U = α × participation - β × gini
   - **Output**: Utility score (higher = better)

---

### 6. Visualization Layer (`plots.py`)

**Purpose**: Create visual comparisons of mechanism performance

#### Visualization Functions:

1. **`plot_bar_comparison(df, metrics, title)`**
   - **Input**: Results DataFrame, list of metric names, chart title
   - **Output**: Bar chart comparing mechanisms across metrics
   - **Saves**: "bar_plot_metrics.png"

2. **`plot_radar_chart(df, metrics, title)`**
   - **Input**: Results DataFrame, list of metric names, chart title  
   - **Output**: Radar/spider chart showing mechanism profiles
   - **Saves**: "radar_plot_metrics.png"

---

## 🚀 Adding New Components

### Adding a New Voting Mechanism:

1. **Add mechanism function to `mechanism_simulation.py`:**
```python
def my_new_mechanism(tokens: List[float], **kwargs) -> List[float]:
    # Your logic here
    return transformed_voting_power
```

2. **Update `simulate_proposal()` mechanism selection:**
```python
elif mechanism == "my_new_mechanism":
    power = my_new_mechanism(tokens, **extra_params)
```

### Adding a New Agent Type:

1. **Create new agent class in `agent_base.py`:**
```python
class MyNewAgent(BaseModel):
    # Define input/output models
    # Implement prepare_training_data(), train(), predict() methods
```

2. **Update simulation workflow to use new agent**

### Adding New Metrics:

1. **Add metric function to `evaluation.py`:**
```python
def calculate_my_metric(data: List[float]) -> float:
    # Your calculation logic
    return metric_value
```

2. **Update `simulate_proposal()` to calculate and return new metric**

### Adding New Visualizations:

1. **Add plot function to `plots.py`:**
```python
def plot_my_visualization(df: pd.DataFrame, **kwargs):
    # Your plotting logic
    plt.savefig("my_plot.png")
    plt.show()
```

---

## 📈 Data Flow Summary

```
CSV Data → Data Loader → Training Data → ML Agent → Predictions
    ↓           ↓            ↓           ↓           ↓
Voter Pool → Mechanism → Voting Power → Proposal → Metrics → Visualization
```

## 🔧 Key Extension Points

1. **New Voting Mechanisms**: Add to `mechanism_simulation.py`
2. **New ML Models**: Replace RandomForest in `UserBehaviorAgent`
3. **New Metrics**: Add to `evaluation.py`
4. **New Data Sources**: Modify `data_loader.py`
5. **New Agent Types**: Extend base classes in `agent_base.py`
6. **New Visualizations**: Add to `plots.py`

This modular design makes it easy to experiment with different approaches while maintaining clean separation of concerns.