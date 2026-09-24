import argparse

from app.daily_runner import run_daily_pipeline


def main(hours: int = 24, top_n: int = 10, dry_run: bool = False):
    return run_daily_pipeline(hours=hours, top_n=top_n, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily AI news digest pipeline")
    parser.add_argument("hours", nargs="?", type=int, default=24, help="lookback window in hours")
    parser.add_argument("top_n", nargs="?", type=int, default=10, help="articles to include in the email")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the email to output/digest_preview.html instead of sending it")
    parser.add_argument("--graph", action="store_true",
                        help="print the pipeline graph as a Mermaid diagram and exit")
    args = parser.parse_args()

    if args.graph:
        from app.graph.pipeline import build_pipeline_graph
        print(build_pipeline_graph().get_graph(xray=True).draw_mermaid())
        raise SystemExit(0)

    result = main(hours=args.hours, top_n=args.top_n, dry_run=args.dry_run)
    exit(0 if result["success"] else 1)
