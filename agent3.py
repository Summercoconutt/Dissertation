"""
Agent3: DAO Governance Mechanism Simulation and Optimization

This program implements a fully functional DAO governance simulation system that:
1. Samples proposals and voters from real data
2. Calculates voting power using configurable formulas
3. Predicts votes using heuristic models
4. Simulates voting outcomes across different parameter combinations
5. Performs Monte Carlo analysis for statistical robustness
6. Outputs comprehensive results and statistics

Author: Agent3 Team
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Proposal:
    """Represents a governance proposal."""
    title: str
    body: str
    created_time: str
    proposal_id: str
    
    def __post_init__(self):
        if not self.proposal_id:
            self.proposal_id = f"prop_{hash(self.title) % 10000}"

@dataclass
class Voter:
    """Represents a voter with their attributes."""
    voter_id: str
    token_holdings: float  # T_i
    reputation_score: float  # R_i
    whale_indicator: bool  # W_i
    average_choice: float  # Average voting pattern
    alignment_score: float  # Alignment with majority
    cluster_id: int
    voting_power: float = 0.0  # Computed voting power

@dataclass
class VotingRule:
    """Represents a voting rule with parameters."""
    alpha: float  # Reputation weight
    beta: float   # Whale penalty weight
    rule_name: str
    
    def __str__(self):
        return f"{self.rule_name}_α{self.alpha}_β{self.beta}"

@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    rule: VotingRule
    proposal_id: str
    passed: bool
    support_weight: float
    opposition_weight: float
    abstention_weight: float
    total_participants: int
    pass_rate: float

class DataSampler:
    """Handles data sampling from the CSV file."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file."""
        logger.info(f"Loading data from {self.csv_path}")
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.data)} records")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def sample_proposals(self, n_proposals: int = 100) -> List[Proposal]:
        """Randomly sample N proposals from the dataset."""
        unique_proposals = self.data.drop_duplicates(subset=['Proposal Title'])
        
        if len(unique_proposals) < n_proposals:
            logger.warning(f"Only {len(unique_proposals)} unique proposals available, sampling all")
            n_proposals = len(unique_proposals)
        
        sampled = unique_proposals.sample(n=n_proposals, random_state=42)
        
        proposals = []
        for _, row in sampled.iterrows():
            proposal = Proposal(
                title=row['Proposal Title'],
                body=row['Proposal Body'],
                created_time=row['Created Time'],
                proposal_id=f"prop_{hash(row['Proposal Title']) % 10000}"
            )
            proposals.append(proposal)
        
        logger.info(f"Sampled {len(proposals)} proposals")
        return proposals
    
    def sample_voters(self, n_voters: int = 1000) -> List[Voter]:
        """Randomly sample N voters and compute their attributes."""
        unique_voters = self.data.drop_duplicates(subset=['Voter'])
        
        if len(unique_voters) < n_voters:
            logger.warning(f"Only {len(unique_voters)} unique voters available, sampling all")
            n_voters = len(unique_voters)
        
        sampled = unique_voters.sample(n=n_voters, random_state=42)
        
        # Calculate max votes per voter for normalization
        max_votes_per_voter = self.data.groupby('Voter').size().max()
        
        voters = []
        for _, row in sampled.iterrows():
            # Compute voter attributes from their voting history
            voter_data = self.data[self.data['Voter'] == row['Voter']]
            
            # Token holdings (Voting Power)
            token_holdings = float(row['Voting Power'])
            
            # Reputation score (based on participation and alignment)
            participation_rate = len(voter_data) / max_votes_per_voter
            alignment_rate = voter_data['Aligned With Majority'].mean()
            reputation_score = (participation_rate + alignment_rate) / 2
            
            # Whale indicator
            whale_indicator = bool(row['Is Whale'])
            
            # Average choice (1=FOR, 2=AGAINST, 3=ABSTAIN)
            choice_mapping = {'FOR': 1, 'AGAINST': 2, 'ABSTAIN': 3}
            choices = voter_data['Choice'].map(choice_mapping).fillna(2)
            average_choice = choices.mean()
            
            # Alignment score
            alignment_score = float(row['Aligned With Majority'])
            
            # Cluster ID (simplified clustering based on voting power)
            if token_holdings > 1000000:  # Large holders
                cluster_id = 0
            elif token_holdings > 10000:  # Medium holders
                cluster_id = 1
            else:  # Small holders
                cluster_id = 2
            
            voter = Voter(
                voter_id=row['Voter'],
                token_holdings=token_holdings,
                reputation_score=reputation_score,
                whale_indicator=whale_indicator,
                average_choice=average_choice,
                alignment_score=alignment_score,
                cluster_id=cluster_id
            )
            voters.append(voter)
        
        logger.info(f"Sampled {len(voters)} voters")
        return voters

class VotingPowerCalculator:
    """Calculates voting power using the specified formula."""
    
    @staticmethod
    def calculate_voting_power(voter: Voter, rule: VotingRule) -> float:
        """
        Calculate voting power using the formula:
        Voting_Power_i = sqrt(T_i) + α  R_i - β  W_i
        """
        T_i = voter.token_holdings
        R_i = voter.reputation_score
        W_i = 1.0 if voter.whale_indicator else 0.0
        
        voting_power = np.sqrt(T_i) + rule.alpha * R_i - rule.beta * W_i
        
        # Ensure non-negative voting power
        return max(0.0, voting_power)

class VotePredictor:
    """Predicts voter choices using heuristic models."""
    
    def __init__(self):
        self.random_state = np.random.RandomState(42)
    
    def predict_vote(self, voter: Voter, proposal: Proposal) -> str:
        """
        Predict voter choice using a heuristic model.
        This is a placeholder that can be replaced with a more sophisticated ML model.
        """
        # Base prediction on voter's historical patterns
        base_choice = voter.average_choice
        
        # Add some randomness to simulate proposal-specific factors
        noise = self.random_state.normal(0, 0.3)
        predicted_choice = base_choice + noise
        
        # Map to choice strings
        if predicted_choice <= 1.5:
            return "FOR"
        elif predicted_choice <= 2.5:
            return "AGAINST"
        else:
            return "ABSTAIN"

class VotingSimulator:
    """Simulates voting outcomes for proposals."""
    
    def __init__(self, predictor: VotePredictor):
        self.predictor = predictor
    
    def simulate_proposal(self, proposal: Proposal, voters: List[Voter], rule: VotingRule) -> SimulationResult:
        """Simulate voting for a single proposal."""
        support_weight = 0.0
        opposition_weight = 0.0
        abstention_weight = 0.0
        total_participants = 0
        
        for voter in voters:
            # Calculate voting power
            voting_power = VotingPowerCalculator.calculate_voting_power(voter, rule)
            voter.voting_power = voting_power
            
            # Predict vote
            predicted_vote = self.predictor.predict_vote(voter, proposal)
            
            # Aggregate weights
            if predicted_vote == "FOR":
                support_weight += voting_power
            elif predicted_vote == "AGAINST":
                opposition_weight += voting_power
            else:  # ABSTAIN
                abstention_weight += voting_power
            
            total_participants += 1
        
        # Determine if proposal passes (simple majority rule)
        total_votes = support_weight + opposition_weight
        passed = support_weight > opposition_weight if total_votes > 0 else False
        pass_rate = support_weight / total_votes if total_votes > 0 else 0.0
        
        return SimulationResult(
            rule=rule,
            proposal_id=proposal.proposal_id,
            passed=passed,
            support_weight=support_weight,
            opposition_weight=opposition_weight,
            abstention_weight=abstention_weight,
            total_participants=total_participants,
            pass_rate=pass_rate
        )

class MonteCarloAnalyzer:
    """Performs Monte Carlo analysis on voting simulations."""
    
    def __init__(self, n_iterations: int = 100):
        self.n_iterations = n_iterations
        self.results = []
    
    def run_analysis(self, proposals: List[Proposal], voters: List[Voter], 
                    rules: List[VotingRule], simulator: VotingSimulator) -> Dict[str, Any]:
        """Run Monte Carlo analysis across multiple iterations."""
        logger.info(f"Starting Monte Carlo analysis with {self.n_iterations} iterations")
        
        all_results = []
        
        for iteration in range(self.n_iterations):
            if iteration % 10 == 0:
                logger.info(f"Running iteration {iteration + 1}/{self.n_iterations}")
            
            # Re-sample data for each iteration (with different random seed)
            random.seed(42 + iteration)
            np.random.seed(42 + iteration)
            
            # Sample new proposals and voters for this iteration
            sampler = DataSampler(self.csv_path)
            iter_proposals = sampler.sample_proposals(len(proposals))
            iter_voters = sampler.sample_voters(len(voters))
            
            # Run simulations for each rule
            for rule in rules:
                for proposal in iter_proposals:
                    result = simulator.simulate_proposal(proposal, iter_voters, rule)
                    all_results.append(result)
        
        # Aggregate results
        return self.aggregate_results(all_results, rules)
    
    def aggregate_results(self, results: List[SimulationResult], rules: List[VotingRule]) -> Dict[str, Any]:
        """Aggregate results across all iterations."""
        aggregated = {}
        
        for rule in rules:
            rule_results = [r for r in results if r.rule == rule]
            
            if not rule_results:
                continue
            
            # Calculate statistics
            pass_rates = [r.pass_rate for r in rule_results]
            passed_count = sum(1 for r in rule_results if r.passed)
            
            aggregated[str(rule)] = {
                'total_simulations': len(rule_results),
                'pass_rate_mean': np.mean(pass_rates),
                'pass_rate_std': np.std(pass_rates),
                'pass_rate_min': np.min(pass_rates),
                'pass_rate_max': np.max(pass_rates),
                'proposals_passed': passed_count,
                'proposals_failed': len(rule_results) - passed_count,
                'support_weight_mean': np.mean([r.support_weight for r in rule_results]),
                'opposition_weight_mean': np.mean([r.opposition_weight for r in rule_results]),
                'abstention_weight_mean': np.mean([r.abstention_weight for r in rule_results]),
            }
        
        return aggregated

class ResultsExporter:
    """Exports simulation results to various formats."""
    
    def __init__(self, output_dir: str = "agent3_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_results(self, results: Dict[str, Any], proposals: List[Proposal], 
                      voters: List[Voter], rules: List[VotingRule]):
        """Export all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export aggregated results
        self.export_aggregated_results(results, timestamp)
        
        # Export rule comparison
        self.export_rule_comparison(results, timestamp)
        
        # Export voter statistics
        self.export_voter_statistics(voters, timestamp)
        
        # Export proposal statistics
        self.export_proposal_statistics(proposals, timestamp)
        
        logger.info(f"Results exported to {self.output_dir}")
    
    def export_aggregated_results(self, results: Dict[str, Any], timestamp: str):
        """Export aggregated simulation results."""
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'voting_rule'
        df.reset_index(inplace=True)
        
        filename = self.output_dir / f"aggregated_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Exported aggregated results to {filename}")
    
    def export_rule_comparison(self, results: Dict[str, Any], timestamp: str):
        """Export rule comparison summary."""
        comparison_data = []
        
        for rule_name, stats in results.items():
            comparison_data.append({
                'rule': rule_name,
                'pass_rate_mean': stats['pass_rate_mean'],
                'pass_rate_std': stats['pass_rate_std'],
                'proposals_passed': stats['proposals_passed'],
                'proposals_failed': stats['proposals_failed'],
                'support_weight_mean': stats['support_weight_mean'],
                'opposition_weight_mean': stats['opposition_weight_mean']
            })
        
        df = pd.DataFrame(comparison_data)
        filename = self.output_dir / f"rule_comparison_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Exported rule comparison to {filename}")
    
    def export_voter_statistics(self, voters: List[Voter], timestamp: str):
        """Export voter statistics."""
        voter_data = []
        for voter in voters:
            voter_data.append({
                'voter_id': voter.voter_id,
                'token_holdings': voter.token_holdings,
                'reputation_score': voter.reputation_score,
                'whale_indicator': voter.whale_indicator,
                'average_choice': voter.average_choice,
                'alignment_score': voter.alignment_score,
                'cluster_id': voter.cluster_id
            })
        
        df = pd.DataFrame(voter_data)
        filename = self.output_dir / f"voter_statistics_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Exported voter statistics to {filename}")
    
    def export_proposal_statistics(self, proposals: List[Proposal], timestamp: str):
        """Export proposal statistics."""
        proposal_data = []
        for proposal in proposals:
            proposal_data.append({
                'proposal_id': proposal.proposal_id,
                'title': proposal.title,
                'created_time': proposal.created_time,
                'body_length': len(proposal.body)
            })
        
        df = pd.DataFrame(proposal_data)
        filename = self.output_dir / f"proposal_statistics_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Exported proposal statistics to {filename}")

class Agent3:
    """Main Agent3 class that orchestrates the entire simulation."""
    
    def __init__(self, csv_path: str, output_dir: str = "agent3_results"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.sampler = DataSampler(csv_path)
        self.predictor = VotePredictor()
        self.simulator = VotingSimulator(self.predictor)
        self.exporter = ResultsExporter(output_dir)
    
    def run_simulation(self, n_proposals: int = 100, n_voters: int = 1000, 
                      n_iterations: int = 100, alpha_range: Tuple[float, float, float] = (0.0, 1.0, 0.2),
                      beta_range: Tuple[float, float, float] = (0.0, 1.0, 0.2)):
        """Run the complete simulation pipeline."""
        logger.info("Starting Agent3 simulation")
        
        # 1. Sample data
        logger.info("Step 1: Sampling proposals and voters")
        proposals = self.sampler.sample_proposals(n_proposals)
        voters = self.sampler.sample_voters(n_voters)
        
        # 2. Generate voting rules
        logger.info("Step 2: Generating voting rules")
        rules = self.generate_voting_rules(alpha_range, beta_range)
        
        # 3. Run Monte Carlo analysis
        logger.info("Step 3: Running Monte Carlo analysis")
        analyzer = MonteCarloAnalyzer(n_iterations)
        results = analyzer.run_analysis(proposals, voters, rules, self.simulator)
        
        # 4. Export results
        logger.info("Step 4: Exporting results")
        self.exporter.export_results(results, proposals, voters, rules)
        
        # 5. Print summary
        self.print_summary(results, rules)
        
        logger.info("Agent3 simulation completed successfully")
        return results
    
    def generate_voting_rules(self, alpha_range: Tuple[float, float, float], 
                            beta_range: Tuple[float, float, float]) -> List[VotingRule]:
        """Generate voting rules with different parameter combinations."""
        alpha_start, alpha_end, alpha_step = alpha_range
        beta_start, beta_end, beta_step = beta_range
        
        rules = []
        rule_id = 0
        
        alpha_values = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
        beta_values = np.arange(beta_start, beta_end + beta_step, beta_step)
        
        for alpha in alpha_values:
            for beta in beta_values:
                rule = VotingRule(
                    alpha=round(alpha, 2),
                    beta=round(beta, 2),
                    rule_name=f"Rule_{rule_id:03d}"
                )
                rules.append(rule)
                rule_id += 1
        
        logger.info(f"Generated {len(rules)} voting rules")
        return rules
    
    def print_summary(self, results: Dict[str, Any], rules: List[VotingRule]):
        """Print a summary of the simulation results."""
        print("\n" + "="*80)
        print("AGENT3 SIMULATION SUMMARY")
        print("="*80)
        
        print(f"Total voting rules tested: {len(rules)}")
        print(f"Total simulations run: {sum(stats['total_simulations'] for stats in results.values())}")
        
        # Find best performing rule
        best_rule = None
        best_pass_rate = -1
        
        for rule_name, stats in results.items():
            if stats['pass_rate_mean'] > best_pass_rate:
                best_pass_rate = stats['pass_rate_mean']
                best_rule = rule_name
        
        if best_rule:
            print(f"\nBest performing rule: {best_rule}")
            print(f"Average pass rate: {best_pass_rate:.3f}")
            print(f"Pass rate std: {results[best_rule]['pass_rate_std']:.3f}")
        
        print("\nTop 5 performing rules:")
        sorted_rules = sorted(results.items(), key=lambda x: x[1]['pass_rate_mean'], reverse=True)
        for i, (rule_name, stats) in enumerate(sorted_rules[:5]):
            print(f"{i+1}. {rule_name}: {stats['pass_rate_mean']:.3f}  {stats['pass_rate_std']:.3f}")
        
        print("="*80)

def main():
    """Main function to run Agent3 simulation."""
    # Configuration
    CSV_PATH = "snapshot_votes_with_timestamp_choice.csv"
    OUTPUT_DIR = "agent3_results"
    
    # Simulation parameters
    N_PROPOSALS = 50  # Reduced for faster execution
    N_VOTERS = 500    # Reduced for faster execution
    N_ITERATIONS = 10 # Reduced for faster execution
    
    # Parameter ranges
    ALPHA_RANGE = (0.0, 1.0, 0.5)  # (start, end, step)
    BETA_RANGE = (0.0, 1.0, 0.5)   # (start, end, step)
    
    try:
        # Initialize and run Agent3
        agent3 = Agent3(CSV_PATH, OUTPUT_DIR)
        results = agent3.run_simulation(
            n_proposals=N_PROPOSALS,
            n_voters=N_VOTERS,
            n_iterations=N_ITERATIONS,
            alpha_range=ALPHA_RANGE,
            beta_range=BETA_RANGE
        )
        
        print(f"\nSimulation completed! Results saved to {OUTPUT_DIR}/")
        
    except Exception as e:
        logger.error(f"Error running Agent3 simulation: {e}")
        raise

if __name__ == "__main__":
    main()
