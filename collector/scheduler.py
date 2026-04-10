"""
Automated Collection Scheduler with Logging
Runs the collector every 3 hours and logs everything
"""
import schedule
import time
import logging
import os
from datetime import datetime
from continuous_collector import run_collection

# Setup dedicated scheduler log
os.makedirs('logs', exist_ok=True)
scheduler_logger = logging.getLogger('scheduler')
scheduler_logger.setLevel(logging.INFO)

# File handler for scheduler-specific logs
scheduler_handler = logging.FileHandler('logs/scheduler.log', encoding='utf-8')
scheduler_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

scheduler_logger.addHandler(scheduler_handler)
scheduler_logger.addHandler(console_handler)


def scheduled_collection():
    """Wrapper function for scheduled runs"""
    scheduler_logger.info("="*60)
    scheduler_logger.info(f"[SCHEDULED RUN] Starting collection")
    scheduler_logger.info("="*60)
    
    start_time = time.time()
    
    try:
        run_collection()
        duration = time.time() - start_time
        scheduler_logger.info(f"[SUCCESS] Collection completed in {duration:.1f} seconds")
        
    except Exception as e:
        duration = time.time() - start_time
        scheduler_logger.error(f"[ERROR] Collection failed after {duration:.1f} seconds: {e}")
        scheduler_logger.error("[RETRY] Will try again at next scheduled time")
        
        # Log the full traceback to file only
        import traceback
        with open('logs/scheduler_errors.log', 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Error at {datetime.now()}\n")
            f.write(traceback.format_exc())
            f.write(f"{'='*60}\n")


def main():
    """Main scheduler loop"""
    scheduler_logger.info("="*60)
    scheduler_logger.info("AUTOMATED REDDIT COLLECTOR STARTED")
    scheduler_logger.info("="*60)
    scheduler_logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    scheduler_logger.info("Collection interval: Every 3 hours")
    scheduler_logger.info("Logs: logs/scheduler.log and logs/collector.log")
    scheduler_logger.info("Press Ctrl+C to stop")
    scheduler_logger.info("="*60)
    
    # Schedule the job
    schedule.every(3).hours.do(scheduled_collection)
    
    # Run immediately on startup
    scheduler_logger.info("\n[IMMEDIATE] Running first collection now...")
    scheduled_collection()
    
    # Calculate next run time
    next_run = schedule.next_run()
    scheduler_logger.info(f"\n[SCHEDULED] Next run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    scheduler_logger.info("[WAITING] Scheduler is running... (minimize this window)")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        scheduler_logger.info("\n" + "="*60)
        scheduler_logger.info("[STOPPED] Scheduler stopped by user (Ctrl+C)")
        scheduler_logger.info(f"Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        scheduler_logger.info("="*60)
        scheduler_logger.info("Goodbye! ")