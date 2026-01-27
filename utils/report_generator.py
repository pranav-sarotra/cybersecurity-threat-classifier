"""
Report generation module with Shodan support.
"""

import json
from datetime import datetime
from typing import Dict, List


class ReportGenerator:
    @staticmethod
    def generate_json_report(result: Dict, pretty: bool = True) -> str:
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'threat_analysis',
                'version': '3.0.0'
            },
            'analysis_result': result
        }
        return json.dumps(report, indent=2 if pretty else None, default=str)

    @staticmethod
    def generate_text_report(result: Dict) -> str:
        lines = [
            "=" * 60,
            "XYZ COMPANY SECURITY THREAT ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 60,
            "CLASSIFICATION SUMMARY",
            "-" * 60,
            f"Classification: {result.get('classification', 'Unknown')}",
            f"Confidence: {result.get('confidence', 0)}%",
            f"Threat Score: {result.get('threat_score', 0)}%",
            f"Threat Level: {result.get('threat_level', 'Unknown')}",
            "",
            "-" * 60,
            "DETECTED INDICATORS",
            "-" * 60,
        ]
        
        for indicator in result.get('indicators', []):
            lines.append(f"  • {indicator}")
        
        if not result.get('indicators'):
            lines.append("  No suspicious indicators detected")
        
        lines.extend([
            "",
            "-" * 60,
            "RECOMMENDATIONS",
            "-" * 60
        ])
        
        for rec in result.get('recommendations', []):
            lines.append(f"  {rec}")
        
        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return "\n".join(lines)

    @staticmethod
    def generate_ip_report(ip_info: Dict) -> str:
        lines = [
            "=" * 60,
            "IP ADDRESS THREAT INTELLIGENCE REPORT",
            "=" * 60,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 60,
            "IP INFORMATION",
            "-" * 60,
            f"IP Address: {ip_info.get('ip', 'Unknown')}",
            f"Country: {ip_info.get('country', 'Unknown')}",
            f"City: {ip_info.get('city', 'Unknown')}",
            f"ISP: {ip_info.get('isp', 'Unknown')}",
            "",
            "-" * 60,
            "THREAT ASSESSMENT",
            "-" * 60,
            f"Threat Score: {ip_info.get('threat_score', 0)}%",
            "",
            "Security Flags:",
            f"  VPN: {'Yes' if ip_info.get('is_vpn') else 'No'}",
            f"  Proxy: {'Yes' if ip_info.get('is_proxy') else 'No'}",
            f"  Tor: {'Yes' if ip_info.get('is_tor') else 'No'}",
            "",
        ]
        
        # Shodan Data
        if ip_info.get('open_ports'):
            lines.extend([
                "-" * 60,
                "SHODAN - OPEN PORTS",
                "-" * 60,
            ])
            for port in ip_info.get('open_ports', []):
                lines.append(f"  • Port {port}")
        
        if ip_info.get('vulns'):
            lines.extend([
                "",
                "-" * 60,
                "SHODAN - VULNERABILITIES (CVEs)",
                "-" * 60,
            ])
            for vuln in ip_info.get('vulns', []):
                lines.append(f"  ⚠️ {vuln}")
        
        if ip_info.get('hostnames'):
            lines.extend([
                "",
                "-" * 60,
                "ASSOCIATED HOSTNAMES",
                "-" * 60,
            ])
            for hostname in ip_info.get('hostnames', []):
                lines.append(f"  • {hostname}")
        
        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60
        ])
        
        return "\n".join(lines)