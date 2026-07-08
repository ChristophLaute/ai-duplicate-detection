import csv
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

# 🔧 SET THIS PATH CORRECTLY
FAILED_FILE = Path("/Users/christophlaute/Thesis/DuplicateDetectionThesis/VCSL_Testing/failed_downloads.csv")

def categorize(reason: str) -> str:
    r = (reason or "").lower()

    if "private video" in r:
        return "Private"
    elif "video unavailable" in r:
        return "Unavailable"
    elif "404" in r or "not found" in r:
        return "404 / Not Found"
    elif "403" in r or "forbidden" in r:
        return "403 / Forbidden"
    elif "age" in r:
        return "Age Restricted"
    elif "timeout" in r:
        return "Timeout"
    elif "format not available" in r or "requested format" in r:
        return "Format Issue"
    elif "unable to download webpage" in r:
        return "Webpage Error"
    elif "geo" in r or "country" in r:
        return "Geo Restricted"
    else:
        return "Other"

counter = Counter()

with FAILED_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        reason = row.get("reason", "")
        counter[categorize(reason)] += 1

# ---- plot ----
labels = list(counter.keys())
sizes = list(counter.values())

plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("VCSL Download Failure Reasons")
plt.show()