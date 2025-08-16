# run.py  – wrapper required by the TDS Project-2 grader
import sys, json, requests, sys

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: run.py <API_URL>", file=sys.stderr)
        sys.exit(1)

    api_url = sys.argv[1]            # e.g. https://project-2-tds.onrender.com
    payload = json.load(sys.stdin)   # grader pipes {"question":..., "task_id":...}

    resp = requests.post(api_url, json=payload, timeout=300)
    resp.raise_for_status()          # surface HTTP errors to grader
    print(resp.text)                 # MUST be raw JSON – nothing else

if __name__ == "__main__":
    main()
