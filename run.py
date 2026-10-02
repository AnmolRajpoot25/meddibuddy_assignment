import os
import sys
import argparse
import uvicorn
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

load_dotenv()

def start_server():
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    print("\n" + "=" * 65)
    print("  WEATHER-ADVISORY SUPPORT BOT - SERVER LAUNCHER")
    print("=" * 65)
    print(f" Web UI:        http://{host}:{port}")
    print(f" API Docs:      http://{host}:{port}/docs")
    print(f" OpenRouter:    {'CONFIGURED' if os.getenv('OPENROUTER_API_KEY') else 'NOT CONFIGURED (Using Grounded Safety Formatter)'}")
    print("=" * 65 + "\n")
    uvicorn.run("frontend.app:app", host=host, port=port, reload=True)

def run_eval():
    from eval.eval_suite import run_evaluation_suite
    success = run_evaluation_suite()
    sys.exit(0 if success else 1)

def run_cli():
    from graph.workflow import run_advisory_agent
    import uuid
    session_id = f"cli_session_{uuid.uuid4().hex[:6]}"
    print("\n" + "=" * 65)
    print("  WEATHER-ADVISORY SUPPORT BOT - INTERACTIVE CLI")
    print(f"  Session ID: {session_id}")
    print("  Type 'exit' or 'quit' to terminate session.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_msg = input("\nYou > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ["exit", "quit"]:
                print("Session terminated. Goodbye!")
                break
            
            res = run_advisory_agent(session_id=session_id, user_input=user_msg)
            print("\nBot > " + res.get("final_response", ""))
            if res.get("primary_sop"):
                sop = res["primary_sop"]
                print(f"\n[Traceability: Matched {sop['id']} ({sop['severity']}) | Location: {res.get('location_name')}]")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather-Advisory Support Bot Launcher")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI web server and UI (Default)")
    parser.add_argument("--eval", action="store_true", help="Run automated test evaluation suite")
    parser.add_argument("--cli", action="store_true", help="Launch interactive command-line session")

    args = parser.parse_args()

    if args.eval:
        run_eval()
    elif args.cli:
        run_cli()
    else:
        start_server()
