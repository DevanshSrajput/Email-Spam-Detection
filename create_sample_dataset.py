import pandas as pd

# Example data (very small sample just for testing)
emails = [
    {"text": "Congratulations! You've won $1,000,000. Click here to claim now!", "label": "spam"},
    {"text": "URGENT: Your account has been compromised. Verify your details at: http://bit.ly/2kFR", "label": "spam"},
    {"text": "Get 90% off on all products! Limited time offer!!!", "label": "spam"},
    {"text": "Meeting scheduled for tomorrow at 10 AM. Please prepare the quarterly report.", "label": "ham"},
    {"text": "Hi John, can you send me the project files when you get a chance?", "label": "ham"},
    {"text": "The invoice for your recent purchase is attached. Thank you for your business.", "label": "ham"}
]

# Create DataFrame and save to CSV
df = pd.DataFrame(emails)
df.to_csv("spam_ham_dataset.csv", index=False)
print("Sample dataset created!")