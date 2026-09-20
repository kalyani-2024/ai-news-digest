import os
import sys

from dotenv import load_dotenv

load_dotenv()

from app.database.create_tables import create_tables
from app.daily_runner import run_daily_pipeline


def main() -> int:
    hours = int(os.getenv("PIPELINE_HOURS", "24"))
    top_n = int(os.getenv("PIPELINE_TOP_N", "10"))

    create_tables()
    result = run_daily_pipeline(hours=hours, top_n=top_n)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
