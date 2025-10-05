#!/usr/bin/env python3
"""
AI Loop Automation Script for Crypto ML Trading Dashboard

This script implements an autonomous patch workflow that:
1. Reads task description from workflow_state.md
2. Uses Perplexity API to generate requirements and patches
3. Applies patches and runs tests
4. Iterates until success or max attempts reached

Usage:
    python ai_loop.py [--dry-run] [--max-iterations N]
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_loop.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PerplexityClient:
    """Client for Perplexity API with retry logic and rate limiting."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def call_api(self, messages: List[Dict], model: str = "claude-3-5-sonnet-20241022") -> Optional[str]:
        """Call Perplexity API with retry logic."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.1
        }
        
        try:
            response = self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            return None

class WorkflowManager:
    """Manages the autonomous patch workflow."""
    
    def __init__(self, dry_run: bool = False, max_iterations: int = 5):
        self.dry_run = dry_run
        self.max_iterations = max_iterations
        self.project_root = Path.cwd()
        self.workflow_state_path = self.project_root / "workflow_state.md"
        self.brief_path = self.project_root / "docs" / "brief.md"
        self.patch_path = self.project_root / "changes.patch"
        
        # Initialize Perplexity client
        api_key = os.getenv("PPLX_API_KEY")
        if not api_key:
            raise ValueError("PPLX_API_KEY environment variable not set")
        self.client = PerplexityClient(api_key)
        
        logger.info(f"Initialized WorkflowManager (dry_run={dry_run}, max_iterations={max_iterations})")
    
    def read_workflow_state(self) -> Dict[str, str]:
        """Read workflow_state.md and extract sections."""
        if not self.workflow_state_path.exists():
            raise FileNotFoundError("workflow_state.md not found")
        
        content = self.workflow_state_path.read_text()
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def generate_brief(self) -> bool:
        """Generate docs/brief.md using Perplexity API."""
        logger.info("Generating requirements brief...")
        
        workflow_state = self.read_workflow_state()
        goal = workflow_state.get("Goal", "Complete the current development task")
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert software architect. Generate a comprehensive requirements brief for the given goal."
            },
            {
                "role": "user",
                "content": f"""Generate a requirements brief for this goal: {goal}

Please create a structured brief with these sections:

## Goal
Clear statement of what needs to be accomplished

## Key APIs/libraries
List the main APIs and libraries to be used with brief descriptions

## Implementation approach
Step-by-step approach for implementing the solution

## Acceptance criteria
Specific, testable criteria for success

Format as markdown suitable for docs/brief.md"""
            }
        ]
        
        response = self.client.call_api(messages)
        if not response:
            logger.error("Failed to generate brief")
            return False
        
        # Ensure docs directory exists
        self.brief_path.parent.mkdir(exist_ok=True)
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would write brief to {self.brief_path}")
            logger.info(f"Brief content preview:\n{response[:200]}...")
            return True
        
        self.brief_path.write_text(response)
        logger.info(f"Generated brief at {self.brief_path}")
        return True
    
    def generate_patch(self) -> bool:
        """Generate unified git patch using Perplexity API."""
        logger.info("Generating implementation patch...")
        
        if not self.brief_path.exists():
            logger.error("Brief not found, generating it first...")
            if not self.generate_brief():
                return False
        
        brief_content = self.brief_path.read_text()
        workflow_content = self.workflow_state_path.read_text()
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert software developer. Generate a unified git patch that implements the requirements."
            },
            {
                "role": "user",
                "content": f"""Based on these requirements and current project state, generate a unified git patch:

REQUIREMENTS BRIEF:
{brief_content}

CURRENT PROJECT STATE:
{workflow_content}

Please generate a unified git patch that:
1. Implements the requirements from the brief
2. Follows the project's coding standards and patterns
3. Includes necessary test files
4. Uses minimal, targeted changes (patch-only approach)
5. Maintains existing functionality

Output ONLY the unified diff format, no explanations."""
            }
        ]
        
        response = self.client.call_api(messages)
        if not response:
            logger.error("Failed to generate patch")
            return False
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would write patch to {self.patch_path}")
            logger.info(f"Patch preview:\n{response[:500]}...")
            return True
        
        self.patch_path.write_text(response)
        logger.info(f"Generated patch at {self.patch_path}")
        return True
    
    def apply_patch(self) -> bool:
        """Apply the generated patch using git."""
        if self.dry_run:
            logger.info("DRY RUN: Would apply patch")
            return True
        
        if not self.patch_path.exists():
            logger.error("Patch file not found")
            return False
        
        try:
            # Test patch first
            result = subprocess.run(
                ["git", "apply", "--check", str(self.patch_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                logger.error(f"Patch check failed: {result.stderr}")
                return False
            
            # Apply patch
            result = subprocess.run(
                ["git", "apply", str(self.patch_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                logger.error(f"Patch application failed: {result.stderr}")
                return False
            
            logger.info("Patch applied successfully")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"Error applying patch: {e}")
            return False
    
    def run_tests(self) -> Tuple[bool, List[str]]:
        """Run tests and return success status and failure messages."""
        logger.info("Running tests...")
        
        test_commands = [
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            ["python", "-m", "flake8", ".", "--max-line-length=100", "--ignore=E203,W503"],
            ["python", "-c", "import streamlit, pandas, numpy; print('Imports OK')"]
        ]
        
        failures = []
        all_passed = True
        
        for cmd in test_commands:
            try:
                if self.dry_run:
                    logger.info(f"DRY RUN: Would run {' '.join(cmd)}")
                    continue
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode != 0:
                    all_passed = False
                    failure_msg = f"Command {' '.join(cmd)} failed:\n{result.stderr}"
                    failures.append(failure_msg)
                    logger.error(failure_msg)
                else:
                    logger.info(f"✓ {' '.join(cmd)} passed")
                    
            except subprocess.SubprocessError as e:
                all_passed = False
                failure_msg = f"Error running {' '.join(cmd)}: {e}"
                failures.append(failure_msg)
                logger.error(failure_msg)
        
        return all_passed, failures
    
    def update_failing_tests(self, failures: List[str]):
        """Update workflow_state.md with failing tests."""
        if not failures:
            return
        
        content = self.workflow_state_path.read_text()
        
        # Find Failing tests section
        lines = content.split('\n')
        failing_tests_start = -1
        for i, line in enumerate(lines):
            if line.startswith('## Failing tests'):
                failing_tests_start = i
                break
        
        if failing_tests_start == -1:
            # Add new section
            new_content = content + f"\n\n## Failing tests\n- [ ] {datetime.now().isoformat()}: " + "\n- [ ] ".join(failures)
        else:
            # Update existing section
            new_failures = [f"- [ ] {datetime.now().isoformat()}: {failure}" for failure in failures]
            lines.insert(failing_tests_start + 2, "\n".join(new_failures))
            new_content = "\n".join(lines)
        
        if not self.dry_run:
            self.workflow_state_path.write_text(new_content)
        
        logger.info(f"Updated failing tests section with {len(failures)} failures")
    
    def commit_changes(self):
        """Commit successful changes."""
        if self.dry_run:
            logger.info("DRY RUN: Would commit changes")
            return
        
        try:
            subprocess.run(
                ["git", "add", "."],
                check=True,
                cwd=self.project_root
            )
            
            subprocess.run(
                ["git", "commit", "-m", f"[AI LOOP] Automated patch application - {datetime.now().isoformat()}"],
                check=True,
                cwd=self.project_root
            )
            
            logger.info("Changes committed successfully")
            
        except subprocess.SubprocessError as e:
            logger.error(f"Error committing changes: {e}")
    
    def update_log(self, message: str):
        """Update workflow_state.md log section."""
        content = self.workflow_state_path.read_text()
        
        # Find Log section
        lines = content.split('\n')
        log_start = -1
        for i, line in enumerate(lines):
            if line.startswith('## Log'):
                log_start = i
                break
        
        if log_start == -1:
            new_content = content + f"\n\n## Log\n**{datetime.now().isoformat()}**: {message}"
        else:
            new_log_entry = f"**{datetime.now().isoformat()}**: {message}"
            lines.insert(log_start + 2, new_log_entry)
            new_content = "\n".join(lines)
        
        if not self.dry_run:
            self.workflow_state_path.write_text(new_content)
        
        logger.info(f"Updated log: {message}")
    
    def run_loop(self):
        """Execute the main automation loop."""
        logger.info("Starting AI automation loop...")
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n=== ITERATION {iteration}/{self.max_iterations} ===")
            
            try:
                # Step 1: Generate brief
                if not self.generate_brief():
                    logger.error("Failed to generate brief")
                    continue
                
                # Step 2: Generate patch
                if not self.generate_patch():
                    logger.error("Failed to generate patch")
                    continue
                
                # Step 3: Apply patch
                if not self.apply_patch():
                    logger.error("Failed to apply patch")
                    continue
                
                # Step 4: Run tests
                tests_passed, failures = self.run_tests()
                
                if tests_passed:
                    logger.info("🎉 All tests passed!")
                    self.commit_changes()
                    self.update_log(f"SUCCESS: Completed in {iteration} iterations")
                    return True
                else:
                    logger.warning(f"Tests failed in iteration {iteration}")
                    self.update_failing_tests(failures)
                    
                    # Brief pause before retry
                    time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                continue
        
        logger.error(f"Failed to complete after {self.max_iterations} iterations")
        self.update_log(f"FAILED: Completed {self.max_iterations} iterations without success")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Loop Automation Script")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations (default: 5)")
    
    args = parser.parse_args()
    
    try:
        workflow = WorkflowManager(
            dry_run=args.dry_run,
            max_iterations=args.max_iterations
        )
        
        success = workflow.run_loop()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
