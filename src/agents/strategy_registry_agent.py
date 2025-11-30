"""
🌙 Moon Dev's Strategy Registry Agent
Manages the lifecycle of trading strategies in the marketplace
Built with love by Moon Dev 🚀
"""

import os
import sys
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from termcolor import cprint
import shutil

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.agents.base_agent import BaseAgent


class StrategyRegistryAgent(BaseAgent):
    """Manages strategy registration, validation, and lifecycle in the marketplace"""
    
    def __init__(self):
        super().__init__(agent_type='marketplace', use_exchange_manager=False)
        self.registry_path = os.path.join(project_root, "src", "data", "marketplace", "registry.json")
        self.strategies_path = os.path.join(project_root, "src", "data", "marketplace", "strategies")
        self.metrics_path = os.path.join(project_root, "src", "data", "marketplace", "metrics")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        os.makedirs(self.strategies_path, exist_ok=True)
        os.makedirs(self.metrics_path, exist_ok=True)
        
        # Load or initialize registry
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load the strategy registry from disk"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"strategies": {}, "categories": {}, "authors": {}}
    
    def _save_registry(self):
        """Save the strategy registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_strategy(self, 
                         name: str,
                         description: str,
                         author: str,
                         code_path: str,
                         category: List[str],
                         timeframes: List[str],
                         instruments: List[str],
                         min_capital: float = 100.0,
                         risk_level: str = "medium",
                         dependencies: List[str] = None) -> Dict[str, Any]:
        """
        Register a new strategy in the marketplace
        
        Args:
            name: Strategy name
            description: Strategy description
            author: Author name/identifier
            code_path: Path to strategy Python file
            category: List of categories (momentum, mean_reversion, etc.)
            timeframes: Supported timeframes
            instruments: Supported instruments/tokens
            min_capital: Minimum capital required
            risk_level: Risk level (low, medium, high)
            dependencies: Required dependencies
            
        Returns:
            Strategy metadata with assigned ID
        """
        # Generate strategy ID
        strategy_id = str(uuid.uuid4())
        
        # Validate strategy code
        if not self._validate_strategy(code_path):
            raise ValueError("Strategy validation failed")
        
        # Calculate file hash for versioning
        file_hash = self._calculate_file_hash(code_path)
        
        # Copy strategy to marketplace
        strategy_filename = f"{strategy_id}_{name.replace(' ', '_')}.py"
        dest_path = os.path.join(self.strategies_path, strategy_filename)
        shutil.copy2(code_path, dest_path)
        
        # Create strategy metadata
        metadata = {
            "strategy_id": strategy_id,
            "name": name,
            "description": description,
            "author": author,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "category": category,
            "timeframes": timeframes,
            "instruments": instruments,
            "min_capital": min_capital,
            "risk_level": risk_level,
            "dependencies": dependencies or [],
            "file_path": dest_path,
            "file_hash": file_hash,
            "performance_summary": {},
            "rating": {"average": 0.0, "count": 0},
            "downloads": 0,
            "status": "under_review"
        }
        
        # Add to registry
        self.registry["strategies"][strategy_id] = metadata
        
        # Update category index
        for cat in category:
            if cat not in self.registry["categories"]:
                self.registry["categories"][cat] = []
            self.registry["categories"][cat].append(strategy_id)
        
        # Update author index
        if author not in self.registry["authors"]:
            self.registry["authors"][author] = []
        self.registry["authors"][author].append(strategy_id)
        
        # Save registry
        self._save_registry()
        
        cprint(f"✅ Strategy '{name}' registered successfully with ID: {strategy_id}", "green")
        return metadata
    
    def _validate_strategy(self, code_path: str) -> bool:
        """
        Validate strategy code for safety and compatibility
        
        Args:
            code_path: Path to strategy file
            
        Returns:
            True if valid, False otherwise
        """
        # Check file exists
        if not os.path.exists(code_path):
            cprint(f"❌ Strategy file not found: {code_path}", "red")
            return False
        
        # Read strategy code
        with open(code_path, 'r') as f:
            code = f.read()
        
        # Basic safety checks - be more specific to avoid false positives
        forbidden_patterns = [
            'subprocess', 
            'eval(', 
            'exec(', 
            '__import__',
            'os.system',
            'os.popen',
            'os.remove',
            'os.rmdir',
            'open(.*w.*)',  # Writing files
        ]
        for pattern in forbidden_patterns:
            if pattern in code:
                cprint(f"❌ Forbidden pattern found: {pattern}", "red")
                return False
        
        # Check for Strategy inheritance (either BaseStrategy or backtesting.Strategy)
        if "BaseStrategy" not in code and "Strategy" not in code:
            cprint("❌ Strategy must inherit from BaseStrategy or backtesting.Strategy", "red")
            return False
        
        # Check for required methods based on strategy type
        if "BaseStrategy" in code:
            # For BaseStrategy, check for generate_signals
            if "def generate_signals" not in code:
                cprint("❌ Required method missing: generate_signals", "red")
                return False
        elif "Strategy" in code and "backtesting" in code:
            # For backtesting.Strategy, check for init and next
            if "def init" not in code or "def next" not in code:
                cprint("❌ Required methods missing: init and/or next", "red")
                return False
        
        return True
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def search_strategies(self,
                         query: str = "",
                         category: Optional[str] = None,
                         author: Optional[str] = None,
                         min_rating: float = 0.0,
                         risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for strategies based on criteria
        
        Args:
            query: Text search in name/description
            category: Filter by category
            author: Filter by author
            min_rating: Minimum rating
            risk_level: Filter by risk level
            
        Returns:
            List of matching strategies
        """
        results = []
        
        for strategy_id, metadata in self.registry["strategies"].items():
            # Text search
            if query and query.lower() not in metadata["name"].lower() and \
               query.lower() not in metadata["description"].lower():
                continue
            
            # Category filter
            if category and category not in metadata["category"]:
                continue
            
            # Author filter
            if author and metadata["author"] != author:
                continue
            
            # Rating filter
            if metadata["rating"]["average"] < min_rating:
                continue
            
            # Risk level filter
            if risk_level and metadata["risk_level"] != risk_level:
                continue
            
            results.append(metadata)
        
        # Sort by rating and downloads
        results.sort(key=lambda x: (x["rating"]["average"], x["downloads"]), reverse=True)
        
        return results
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy metadata by ID"""
        return self.registry["strategies"].get(strategy_id)
    
    def update_performance(self, strategy_id: str, performance_data: Dict[str, Any]):
        """
        Update strategy performance summary
        
        Args:
            strategy_id: Strategy ID
            performance_data: Performance metrics
        """
        if strategy_id not in self.registry["strategies"]:
            cprint(f"❌ Strategy not found: {strategy_id}", "red")
            return
        
        self.registry["strategies"][strategy_id]["performance_summary"] = performance_data
        self.registry["strategies"][strategy_id]["updated_at"] = datetime.now().isoformat()
        self._save_registry()
        
        cprint(f"✅ Updated performance data for strategy: {strategy_id}", "green")
    
    def increment_downloads(self, strategy_id: str):
        """Increment download counter for a strategy"""
        if strategy_id in self.registry["strategies"]:
            self.registry["strategies"][strategy_id]["downloads"] += 1
            self._save_registry()
    
    def update_rating(self, strategy_id: str, rating: float):
        """
        Update strategy rating
        
        Args:
            strategy_id: Strategy ID
            rating: New rating (1-5)
        """
        if strategy_id not in self.registry["strategies"]:
            return
        
        current = self.registry["strategies"][strategy_id]["rating"]
        new_count = current["count"] + 1
        new_average = ((current["average"] * current["count"]) + rating) / new_count
        
        self.registry["strategies"][strategy_id]["rating"] = {
            "average": round(new_average, 2),
            "count": new_count
        }
        self._save_registry()
    
    def approve_strategy(self, strategy_id: str):
        """Approve a strategy for public use"""
        if strategy_id in self.registry["strategies"]:
            self.registry["strategies"][strategy_id]["status"] = "active"
            self.registry["strategies"][strategy_id]["updated_at"] = datetime.now().isoformat()
            self._save_registry()
            cprint(f"✅ Strategy {strategy_id} approved", "green")
    
    def deprecate_strategy(self, strategy_id: str, reason: str = ""):
        """Mark a strategy as deprecated"""
        if strategy_id in self.registry["strategies"]:
            self.registry["strategies"][strategy_id]["status"] = "deprecated"
            self.registry["strategies"][strategy_id]["deprecation_reason"] = reason
            self.registry["strategies"][strategy_id]["updated_at"] = datetime.now().isoformat()
            self._save_registry()
            cprint(f"⚠️ Strategy {strategy_id} deprecated", "yellow")
    
    def export_strategy(self, strategy_id: str, output_path: str) -> bool:
        """
        Export a strategy with its metadata
        
        Args:
            strategy_id: Strategy ID to export
            output_path: Directory to export to
            
        Returns:
            True if successful
        """
        metadata = self.get_strategy(strategy_id)
        if not metadata:
            cprint(f"❌ Strategy not found: {strategy_id}", "red")
            return False
        
        # Create export directory
        export_dir = os.path.join(output_path, f"strategy_{metadata['name'].replace(' ', '_')}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy strategy file
        strategy_dest = os.path.join(export_dir, f"{metadata['name'].replace(' ', '_')}.py")
        shutil.copy2(metadata["file_path"], strategy_dest)
        
        # Save metadata
        metadata_path = os.path.join(export_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme_path = os.path.join(export_dir, "README.md")
        readme_content = f"""# {metadata['name']}

## Description
{metadata['description']}

## Author
{metadata['author']}

## Performance Summary
{json.dumps(metadata.get('performance_summary', {}), indent=2)}

## Requirements
- Minimum Capital: ${metadata['min_capital']}
- Risk Level: {metadata['risk_level']}
- Timeframes: {', '.join(metadata['timeframes'])}
- Instruments: {', '.join(metadata['instruments'])}

## Installation
1. Copy the strategy file to your `src/strategies/` directory
2. Install required dependencies: {', '.join(metadata['dependencies'])}
3. Run with your preferred agent or backtesting system
"""
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        cprint(f"✅ Strategy exported to: {export_dir}", "green")
        return True
    
    def run(self):
        """Main execution loop for registry maintenance"""
        cprint("\n🏪 Strategy Registry Agent Starting...", "white", "on_blue")
        
        # Print registry statistics
        total_strategies = len(self.registry["strategies"])
        active_strategies = sum(1 for s in self.registry["strategies"].values() 
                              if s["status"] == "active")
        
        cprint(f"\n📊 Registry Statistics:", "cyan")
        cprint(f"Total Strategies: {total_strategies}", "white")
        cprint(f"Active Strategies: {active_strategies}", "white")
        cprint(f"Categories: {len(self.registry['categories'])}", "white")
        cprint(f"Authors: {len(self.registry['authors'])}", "white")
        
        # Show top strategies by downloads
        if self.registry["strategies"]:
            cprint(f"\n🔥 Top Strategies by Downloads:", "cyan")
            sorted_strategies = sorted(
                self.registry["strategies"].values(),
                key=lambda x: x["downloads"],
                reverse=True
            )[:5]
            
            for strategy in sorted_strategies:
                cprint(f"- {strategy['name']}: {strategy['downloads']} downloads, "
                      f"⭐ {strategy['rating']['average']}/5.0", "white")


if __name__ == "__main__":
    # Example usage
    agent = StrategyRegistryAgent()
    
    # Run the agent
    agent.run()
    
    # Example: Register a dummy strategy
    # metadata = agent.register_strategy(
    #     name="RSI Mean Reversion",
    #     description="Classic RSI oversold/overbought strategy",
    #     author="moon_dev",
    #     code_path="path/to/strategy.py",
    #     category=["mean_reversion", "technical"],
    #     timeframes=["15m", "1H"],
    #     instruments=["BTC", "ETH", "SOL"],
    #     min_capital=100.0,
    #     risk_level="low"
    # )