"""
Bandit Security Analyzer

Detects common Python security issues using Bandit static analysis tool
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class BanditAnalyzer:
    """
    Bandit static analysis for Python security vulnerabilities
    
    Detects:
    - SQL injection (B608)
    - Command injection (B602, B605, B607)
    - Deserialization (B301, B403)
    - Weak cryptography (B303, B324)
    - Path traversal (B108)
    """
    
    # CWE mapping for Bandit test IDs
    CWE_MAPPING = {
        'B608': 'CWE-89',   # SQL injection
        'B602': 'CWE-78',   # Shell injection
        'B605': 'CWE-78',   # Shell injection (shell=True)
        'B607': 'CWE-78',   # Partial shell path
        'B301': 'CWE-502',  # Pickle usage
        'B403': 'CWE-502',  # Import pickle
        'B303': 'CWE-327',  # MD5 usage
        'B324': 'CWE-327',  # Weak hash
        'B108': 'CWE-22',   # Hardcoded temp directory
    }
    
    def __init__(self, confidence_level: str = "MEDIUM"):
        """
        Initialize Bandit analyzer
        
        Args:
            confidence_level: Minimum confidence (LOW, MEDIUM, HIGH)
        """
        self.confidence_level = confidence_level
        logger.info(f"Initialized BanditAnalyzer (confidence: {confidence_level})")
    
    def analyze(self, code: str) -> Dict:
        """
        Analyze Python code for security vulnerabilities
        
        Args:
            code: Python code to analyze
        
        Returns:
            Dict with keys:
            - cwes: List of detected CWE IDs
            - issues: List of issue details
            - raw_output: Raw Bandit JSON output
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run Bandit
            result = subprocess.run(
                [
                    'bandit',
                    '-f', 'json',
                    '-ll',  # Low confidence minimum
                    temp_file
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse JSON output
            if result.stdout:
                output = json.loads(result.stdout)
                issues = output.get('results', [])
                
                # Extract CWEs
                cwes = set()
                for issue in issues:
                    test_id = issue.get('test_id', '')
                    if test_id in self.CWE_MAPPING:
                        cwes.add(self.CWE_MAPPING[test_id])
                
                return {
                    'cwes': sorted(list(cwes)),
                    'issues': issues,
                    'raw_output': output
                }
            
            return {'cwes': [], 'issues': [], 'raw_output': {}}
        
        except subprocess.TimeoutExpired:
            logger.error("Bandit analysis timed out")
            return {'cwes': [], 'issues': [], 'error': 'timeout'}
        
        except Exception as e:
            logger.error(f"Bandit analysis failed: {e}")
            return {'cwes': [], 'issues': [], 'error': str(e)}
        
        finally:
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
