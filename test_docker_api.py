"""Test Docker API by uploading file and checking accuracy."""
import requests

url = "http://localhost:5000/upload"
file_path = "c:/other/defy/data/uploads/Batch1withGroundTruth.xlsx"

with open(file_path, 'rb') as f:
    files = {'file': ('Batch1withGroundTruth.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    response = requests.post(url, files=files)

print(f"Status code: {response.status_code}")
print(f"URL: {response.url}")

# Save HTML response
with open("c:/other/defy/docker_response.html", "w", encoding="utf-8") as f:
    f.write(response.text)

# Extract accuracy from response
if "Accuracy:" in response.text:
    import re
    match = re.search(r'Accuracy:\s*([\d.]+)%', response.text)
    if match:
        print(f"\n=== ACCURACY: {match.group(1)}% ===")
elif "accuracy" in response.text.lower():
    import re
    match = re.search(r'(\d+\.?\d*)%', response.text)
    if match:
        print(f"\n=== Found percentage: {match.group(1)}% ===")

print("\nResponse saved to docker_response.html")
