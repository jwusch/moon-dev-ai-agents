"""
🌙 Moon Dev's Strategy Export/Import Tool
Handles packaging and sharing of trading strategies
Built with love by Moon Dev 🚀
"""

import os
import sys
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class StrategyExporter:
    """Handles export and import of trading strategies"""
    
    def __init__(self, registry_agent=None):
        self.registry_agent = registry_agent
        self.export_version = "1.0.0"
    
    def export_strategy_package(self, 
                               strategy_id: str, 
                               output_path: str,
                               include_backtest_data: bool = False,
                               include_performance: bool = True) -> str:
        """
        Export a complete strategy package as a ZIP file
        
        Args:
            strategy_id: Strategy ID to export
            output_path: Directory to save the package
            include_backtest_data: Include historical backtest results
            include_performance: Include performance metrics
            
        Returns:
            Path to the created package file
        """
        if not self.registry_agent:
            raise ValueError("Registry agent required for export")
        
        # Get strategy metadata
        metadata = self.registry_agent.get_strategy(strategy_id)
        if not metadata:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Create temporary directory for package contents
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / f"strategy_{metadata['name'].replace(' ', '_')}"
            package_dir.mkdir()
            
            # 1. Copy strategy code
            strategy_file = package_dir / f"{metadata['name'].replace(' ', '_')}.py"
            shutil.copy2(metadata['file_path'], strategy_file)
            
            # 2. Create enhanced metadata
            export_metadata = {
                **metadata,
                "export_version": self.export_version,
                "exported_at": datetime.now().isoformat(),
                "package_contents": {
                    "strategy_code": True,
                    "performance_data": include_performance,
                    "backtest_data": include_backtest_data
                }
            }
            
            # Remove internal paths
            export_metadata.pop('file_path', None)
            
            # 3. Save metadata
            with open(package_dir / "metadata.json", 'w') as f:
                json.dump(export_metadata, f, indent=2)
            
            # 4. Include performance data if requested
            if include_performance:
                metrics_file = Path(project_root) / "src" / "data" / "marketplace" / "metrics" / f"{strategy_id}_metrics.json"
                if metrics_file.exists():
                    shutil.copy2(metrics_file, package_dir / "performance.json")
            
            # 5. Include backtest data if requested
            if include_backtest_data:
                backtest_dir = package_dir / "backtests"
                backtest_dir.mkdir()
                # TODO: Copy relevant backtest files
            
            # 6. Create README
            self._create_readme(package_dir, metadata, include_performance)
            
            # 7. Create requirements file
            self._create_requirements(package_dir, metadata)
            
            # 8. Create installation script
            self._create_install_script(package_dir, metadata)
            
            # 9. Create package manifest
            manifest = self._create_manifest(package_dir)
            with open(package_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # 10. Zip the package
            package_name = f"strategy_{metadata['name'].replace(' ', '_')}_{metadata['version'].replace('.', '_')}.zip"
            output_file = Path(output_path) / package_name
            
            with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_dir)
                        zipf.write(file_path, arcname)
            
            return str(output_file)
    
    def import_strategy_package(self, package_path: str, validate: bool = True) -> Dict[str, Any]:
        """
        Import a strategy package
        
        Args:
            package_path: Path to the strategy package ZIP file
            validate: Whether to validate the package contents
            
        Returns:
            Import result with strategy metadata
        """
        if not os.path.exists(package_path):
            raise FileNotFoundError(f"Package not found: {package_path}")
        
        # Extract to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Unzip package
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Find strategy directory
            strategy_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
            if not strategy_dirs:
                raise ValueError("Invalid package structure")
            
            strategy_dir = strategy_dirs[0]
            
            # Load metadata
            metadata_file = strategy_dir / "metadata.json"
            if not metadata_file.exists():
                raise ValueError("Missing metadata.json")
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate package if requested
            if validate:
                validation_result = self._validate_package(strategy_dir, metadata)
                if not validation_result['valid']:
                    raise ValueError(f"Package validation failed: {validation_result['errors']}")
            
            # Find strategy Python file
            py_files = list(strategy_dir.glob("*.py"))
            if not py_files:
                raise ValueError("No strategy Python file found")
            
            strategy_file = py_files[0]
            
            # Import to marketplace if registry agent available
            if self.registry_agent:
                # Register the strategy
                new_metadata = self.registry_agent.register_strategy(
                    name=metadata['name'],
                    description=metadata['description'],
                    author=metadata['author'],
                    code_path=str(strategy_file),
                    category=metadata['category'],
                    timeframes=metadata['timeframes'],
                    instruments=metadata['instruments'],
                    min_capital=metadata.get('min_capital', 100),
                    risk_level=metadata.get('risk_level', 'medium'),
                    dependencies=metadata.get('dependencies', [])
                )
                
                # Import performance data if available
                perf_file = strategy_dir / "performance.json"
                if perf_file.exists():
                    with open(perf_file, 'r') as f:
                        perf_data = json.load(f)
                        if 'metrics' in perf_data:
                            self.registry_agent.update_performance(
                                new_metadata['strategy_id'],
                                perf_data['metrics']
                            )
                
                return {
                    "success": True,
                    "strategy_id": new_metadata['strategy_id'],
                    "metadata": new_metadata,
                    "message": f"Successfully imported strategy: {metadata['name']}"
                }
            else:
                # Just return the extracted files location
                dest_dir = Path(project_root) / "src" / "strategies" / "imported"
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy strategy file
                dest_file = dest_dir / strategy_file.name
                shutil.copy2(strategy_file, dest_file)
                
                return {
                    "success": True,
                    "strategy_file": str(dest_file),
                    "metadata": metadata,
                    "message": f"Strategy extracted to: {dest_file}"
                }
    
    def _create_readme(self, package_dir: Path, metadata: Dict[str, Any], include_performance: bool):
        """Create comprehensive README for the package"""
        readme_content = f"""# {metadata['name']}

## Description
{metadata['description']}

## Strategy Information
- **Author**: {metadata['author']}
- **Version**: {metadata['version']}
- **Created**: {metadata['created_at'][:10]}
- **Category**: {', '.join(metadata['category'])}
- **Risk Level**: {metadata['risk_level']}

## Requirements
- **Minimum Capital**: ${metadata['min_capital']}
- **Supported Timeframes**: {', '.join(metadata['timeframes'])}
- **Supported Instruments**: {', '.join(metadata['instruments'])}
- **Python Dependencies**: {', '.join(metadata['dependencies']) if metadata['dependencies'] else 'None'}

"""
        
        if include_performance and metadata.get('performance_summary'):
            perf = metadata['performance_summary']
            readme_content += f"""## Performance Summary
- **Total Return**: {perf.get('total_return', 'N/A')}%
- **Sharpe Ratio**: {perf.get('sharpe_ratio', 'N/A')}
- **Win Rate**: {perf.get('win_rate', 'N/A')}%
- **Max Drawdown**: {perf.get('max_drawdown', 'N/A')}%

"""
        
        readme_content += """## Installation

### Automatic Installation
```bash
python install.py
```

### Manual Installation
1. Copy the strategy file to your `src/strategies/` directory
2. Install required dependencies: `pip install -r requirements.txt`
3. Run with your preferred agent or backtesting system

## Usage

### With Moon Dev Trading System
```python
from src.agents.strategy_agent import StrategyAgent

agent = StrategyAgent()
agent.run()
```

### Standalone Backtesting
```python
from backtesting import Backtest
from your_strategy import YourStrategy

bt = Backtest(data, YourStrategy, cash=10000, commission=.002)
stats = bt.run()
print(stats)
```

## Support
- Discord: https://discord.gg/8UPuVZ53bh
- YouTube: Moon Dev Channel

## License
This strategy is provided as-is for educational purposes.
"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    def _create_requirements(self, package_dir: Path, metadata: Dict[str, Any]):
        """Create requirements.txt file"""
        base_requirements = [
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "backtesting>=0.3.3"
        ]
        
        # Add strategy-specific dependencies
        all_requirements = base_requirements + metadata.get('dependencies', [])
        
        with open(package_dir / "requirements.txt", 'w') as f:
            f.write('\n'.join(all_requirements))
    
    def _create_install_script(self, package_dir: Path, metadata: Dict[str, Any]):
        """Create installation script"""
        install_script = f"""#!/usr/bin/env python
\"\"\"
Installation script for {metadata['name']} strategy
\"\"\"

import os
import shutil
import sys
from pathlib import Path

def install_strategy():
    # Get Moon Dev project root
    project_root = os.environ.get('MOONDEV_PROJECT_ROOT')
    if not project_root:
        print("Please set MOONDEV_PROJECT_ROOT environment variable")
        return False
    
    # Strategy source file
    strategy_file = Path(__file__).parent / "{metadata['name'].replace(' ', '_')}.py"
    
    # Destination
    dest_dir = Path(project_root) / "src" / "strategies"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = dest_dir / strategy_file.name
    
    # Copy strategy
    shutil.copy2(strategy_file, dest_file)
    print(f"Strategy installed to: {{dest_file}}")
    
    # Install dependencies
    print("Installing dependencies...")
    os.system("pip install -r requirements.txt")
    
    print("Installation complete!")
    return True

if __name__ == "__main__":
    install_strategy()
"""
        
        with open(package_dir / "install.py", 'w') as f:
            f.write(install_script)
    
    def _create_manifest(self, package_dir: Path) -> Dict[str, Any]:
        """Create package manifest with checksums"""
        manifest = {
            "export_version": self.export_version,
            "created_at": datetime.now().isoformat(),
            "files": {}
        }
        
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(package_dir)
                
                # Calculate checksum
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                
                manifest["files"][str(relative_path)] = {
                    "size": file_path.stat().st_size,
                    "sha256": sha256_hash.hexdigest()
                }
        
        return manifest
    
    def _validate_package(self, package_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate package contents"""
        errors = []
        warnings = []
        
        # Check required files
        required_files = ['metadata.json', 'README.md']
        for req_file in required_files:
            if not (package_dir / req_file).exists():
                errors.append(f"Missing required file: {req_file}")
        
        # Check for strategy Python file
        py_files = list(package_dir.glob("*.py"))
        if not py_files:
            errors.append("No Python strategy file found")
        
        # Validate metadata structure
        required_fields = ['name', 'description', 'author', 'version']
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")
        
        # Check manifest if exists
        manifest_file = package_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                
            # Verify checksums
            for file_path, file_info in manifest.get('files', {}).items():
                full_path = package_dir / file_path
                if full_path.exists():
                    # Recalculate checksum
                    sha256_hash = hashlib.sha256()
                    with open(full_path, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
                    
                    if sha256_hash.hexdigest() != file_info['sha256']:
                        warnings.append(f"Checksum mismatch for: {file_path}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


if __name__ == "__main__":
    # Example usage
    from src.agents.strategy_registry_agent import StrategyRegistryAgent
    
    registry = StrategyRegistryAgent()
    exporter = StrategyExporter(registry)
    
    # Example export (would need actual strategy registered first)
    # package_path = exporter.export_strategy_package(
    #     "strategy_id_here",
    #     "/tmp/",
    #     include_performance=True
    # )
    # print(f"Package created: {package_path}")
    
    print("Strategy Exporter ready for use")