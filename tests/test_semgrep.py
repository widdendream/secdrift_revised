#!/usr/bin/env python3
"""Test semgrep analyzer performance"""

from secdrift.analyzers.semgrep_analyzer import SemgrepAnalyzer
import time

# Test code with SQL injection
test_code = """
import sqlite3

def search_users(name):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Vulnerable: SQL injection
    query = f"SELECT * FROM users WHERE name = '{name}'"
    cursor.execute(query)
    return cursor.fetchall()
"""

print("Testing Semgrep analyzer...")
print("=" * 50)

analyzer = SemgrepAnalyzer()

start = time.time()
result = analyzer.analyze(test_code)
elapsed = time.time() - start

print(f"Analysis completed in {elapsed:.2f} seconds")
print(f"CWEs detected: {result.get('cwes', [])}")
print(f"Findings: {len(result.get('findings', []))}")

if 'error' in result:
    print(f"❌ ERROR: {result['error']}")
elif elapsed > 60:
    print(f"⚠️  WARNING: Analysis took {elapsed:.2f}s (>60s)")
else:
    print(f"✅ SUCCESS: Analysis completed in {elapsed:.2f}s")
