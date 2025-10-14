"""
    architecture for the "fundamentals retrieval and storage system" with automation

        work-flow:  scheduler (orchestrator) -> data collector -> database (arangodb)
"""

import schedule

def run_job(job: str):
    schedule.every().day.at("09:00").do(job)

