#!/usr/bin/env python3
"""Test error handling in todo CLI"""

import subprocess
import os
import tempfile

def run_cli(args, input_text=None):
    """Run CLI command and return exit code and output"""
    cmd = ['python', 'todo_cli.py'] + args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_text
    )
    
    return result.returncode, result.stdout, result.stderr

def test_error_handling():
    """Test various error conditions"""
    print("Testing CLI error handling...\n")
    
    # Test 1: Missing required argument
    print("1. Testing missing required argument:")
    code, stdout, stderr = run_cli(['add'])
    print(f"   Exit code: {code}")
    print(f"   Error contains 'required': {'required' in stderr}")
    
    # Test 2: Invalid command
    print("\n2. Testing invalid command:")
    code, stdout, stderr = run_cli(['invalid'])
    print(f"   Exit code: {code}")
    print(f"   Error contains 'invalid choice': {'invalid choice' in stderr}")
    
    # Test 3: Non-existent task ID
    print("\n3. Testing non-existent task ID:")
    code, stdout, stderr = run_cli(['complete', 'nonexistent'])
    print(f"   Exit code: {code}")
    print(f"   Error: {stderr.strip()}")
    
    # Test 4: Invalid priority
    print("\n4. Testing invalid priority:")
    code, stdout, stderr = run_cli(['add', 'Test', '-p', 'invalid'])
    print(f"   Exit code: {code}")
    print(f"   Error contains 'invalid choice': {'invalid choice' in stderr}")
    
    # Test 5: File permission error (create read-only file)
    print("\n5. Testing file permission error:")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        readonly_file = f.name
        f.write('{}')
    
    os.chmod(readonly_file, 0o444)  # Read-only
    code, stdout, stderr = run_cli(['-f', readonly_file, 'add', 'Test'])
    print(f"   Exit code: {code}")
    print(f"   Error contains 'Error': {'Error' in stderr}")
    
    # Cleanup
    os.chmod(readonly_file, 0o644)
    os.unlink(readonly_file)
    
    # Test 6: Corrupted JSON file
    print("\n6. Testing corrupted JSON file:")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        corrupt_file = f.name
        f.write('{ invalid json ]')
    
    code, stdout, stderr = run_cli(['-f', corrupt_file, 'list'])
    print(f"   Exit code: {code}")
    print(f"   Handles corrupted file: {code == 0}")
    
    os.unlink(corrupt_file)
    
    print("\nAll error handling tests completed.")

if __name__ == '__main__':
    test_error_handling()