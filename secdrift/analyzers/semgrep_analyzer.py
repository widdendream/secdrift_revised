"""
Semgrep Security Analyzer

Detects security vulnerabilities using Semgrep pattern matching
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class SemgrepAnalyzer:
    """
    Semgrep static analysis for security vulnerabilities
    
    Uses p/security-audit ruleset covering:
    - SQL injection
    - Command injection
    - Deserialization
    - Path traversal
    - Weak cryptography
    """
    
    # CWE mapping for common Semgrep rules
    CWE_MAPPING = {
        'sqli': 'CWE-89',
        'sql-injection': 'CWE-89',
        'command-injection': 'CWE-78',
        'dangerous-subprocess': 'CWE-78',
        'pickle': 'CWE-502',
        'deserialization': 'CWE-502',
        'path-traversal': 'CWE-22',
        'weak-crypto': 'CWE-327',
        'md5': 'CWE-327',
        'sha1': 'CWE-327',
    }
    
    def __init__(self, ruleset: str = "auto", timeout: int = 120):
        """
        Initialize Semgrep analyzer
        
        Args:
            ruleset: Semgrep ruleset to use (default: auto for language detection)
            timeout: Timeout in seconds for semgrep execution
        """
        self.ruleset = ruleset
        self.timeout = timeout
        logger.info(f"Initialized SemgrepAnalyzer (ruleset: {ruleset}, timeout: {timeout}s)")
    
    def analyze(self, code: str) -> Dict:
        """
        Analyze Python code for security vulnerabilities
        
        Args:
            code: Python code to analyze
        
        Returns:
            Dict with keys:
            - cwes: List of detected CWE IDs
            - findings: List of finding details
            - raw_output: Raw Semgrep JSON output
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run Semgrep with increased timeout and optimizations
            result = subprocess.run(
                [
                    'semgrep',
                    '--config', self.ruleset,
                    '--json',
                    '--quiet',
                    '--no-git-ignore',  # Don't check .gitignore
                    '--metrics', 'off',  # Disable metrics collection
                    temp_file
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse JSON output
            if result.stdout:
                output = json.loads(result.stdout)
                findings = output.get('results', [])
                
                # Extract CWEs
                cwes = set()
                for finding in findings:
                    # Check rule ID for CWE indicators
                    rule_id = finding.get('check_id', '').lower()
                    for keyword, cwe in self.CWE_MAPPING.items():
                        if keyword in rule_id:
                            cwes.add(cwe)
                            break
                    
                    # Check metadata for CWE
                    metadata = finding.get('extra', {}).get('metadata', {})
                    if 'cwe' in metadata:
                        cwe_list = metadata['cwe']
                        if isinstance(cwe_list, list):
                            cwes.update([f"CWE-{c}" if not str(c).startswith('CWE') else str(c) 
                                        for c in cwe_list])
                
                return {
                    'cwes': sorted(list(cwes)),
                    'findings': findings,
                    'raw_output': output
                }
            
            return {'cwes': [], 'findings': [], 'raw_output': {}}
        
        except subprocess.TimeoutExpired:
            logger.error("Semgrep analysis timed out")
            return {'cwes': [], 'findings': [], 'error': 'timeout'}
        
        except Exception as e:
            logger.error(f"Semgrep analysis failed: {e}")
            return {'cwes': [], 'findings': [], 'error': str(e)}
        
        finally:
            # Clean up temp file
            Path(temp_file).unlink(missing_ok=True)
