import requests
import json

BASE_URL = "http://127.0.0.1:8000/api"


# -----------------------------------------
# 1. Upload a dummy legal document
# -----------------------------------------
def test_upload():
    print("\n--- UPLOADING TEST DOCUMENT ---\n")

    text = """
    This Agreement is made between Party A and Party B.
    Party A agrees to deliver services every month.
    Payment must be completed within 15 days of invoice.
    Termination may occur with 30 days written notice.
    Penalties apply for late payments.
    """

    files = {
        "file": ("test_doc.txt", text, "text/plain")
    }

    response = requests.post(f"{BASE_URL}/upload", files=files)

    print("Status:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=4))

    return response.json()


# -----------------------------------------
# 2. Query the uploaded document
# -----------------------------------------
def test_query():
    print("\n--- QUERYING DOCUMENT ---\n")

    payload = {
        "question": "What are the payment terms?",
        "top_k": 5
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        f"{BASE_URL}/query",
        data=json.dumps(payload),
        headers=headers
    )

    print("Status:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=4))


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    uploaded = test_upload()
    test_query()

