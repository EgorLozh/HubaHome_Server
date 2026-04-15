import json
import sys
import urllib.error
import urllib.request


BASE_URL = "http://localhost:8000"
PATHS = ["/health", "/config", "/diagnostics", "/metrics"]


def fetch(path: str) -> tuple[int, str]:
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, body


def main() -> int:
    failures = []
    for path in PATHS:
        try:
            status, body = fetch(path)
            if status != 200:
                failures.append(f"{path}: unexpected status {status}")
                continue
            if path != "/metrics":
                json.loads(body)
            print(f"[ok] {path}")
        except (urllib.error.URLError, TimeoutError) as err:
            failures.append(f"{path}: request failed ({err})")
        except json.JSONDecodeError as err:
            failures.append(f"{path}: invalid json ({err})")

    if failures:
        print("[fail] smoke checks failed:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("[ok] all smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
