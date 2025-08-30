import argparse, os, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    urllib.request.urlretrieve(args.url, args.out)
    print(f"Downloaded to {args.out}")

if __name__ == "__main__":
    main()