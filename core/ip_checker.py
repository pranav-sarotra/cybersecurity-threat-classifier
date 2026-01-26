"""
IP Address Checker Module.
"""

import re
import ipaddress
import socket
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class IPInfo:
    """IP address information container."""
    ip: str
    is_valid: bool = False
    ip_type: str = ""
    ip_class: str = ""
    country: str = ""
    country_code: str = ""
    region: str = ""
    city: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    timezone: str = ""
    isp: str = ""
    org: str = ""
    asn: str = ""
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    is_known_attacker: bool = False
    is_known_abuser: bool = False
    is_threat: bool = False
    threat_score: float = 0.0
    reputation_score: float = 100.0
    abuse_confidence_score: float = 0.0
    blacklists: List[str] = field(default_factory=list)
    blacklist_count: int = 0
    reverse_dns: str = ""
    check_timestamp: str = ""
    data_sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class IPChecker:
    """Comprehensive IP address analysis tool."""

    KNOWN_MALICIOUS_RANGES = ["185.220.101.0/24", "45.33.32.0/24"]
    DATACENTER_ASNS = ["AS14061", "AS16509", "AS15169", "AS8075", "AS13335", "AS16276", "AS24940"]
    VPN_ASNS = ["AS9009", "AS60068", "AS206092"]

    def __init__(self, virustotal_api_key: str = "", abuseipdb_api_key: str = "", timeout: int = 10):
        self.virustotal_key = virustotal_api_key
        self.abuseipdb_key = abuseipdb_api_key
        self.timeout = timeout
        self.custom_blacklist: set = set()

    def check_ip(self, ip: str, deep_scan: bool = True) -> IPInfo:
        info = IPInfo(ip=ip.strip(), check_timestamp=datetime.now().isoformat())

        if not self._validate_ip(ip, info):
            return info

        self._classify_ip(ip, info)
        self._reverse_dns(ip, info)
        self._check_local_blacklists(ip, info)
        self._check_malicious_ranges(ip, info)

        if deep_scan and REQUESTS_AVAILABLE:
            self._geolocation_lookup(ip, info)
            if self.abuseipdb_key:
                self._abuseipdb_check(ip, info)

        self._calculate_threat_score(info)
        return info

    def _validate_ip(self, ip: str, info: IPInfo) -> bool:
        try:
            ip_obj = ipaddress.ip_address(ip.strip())
            info.is_valid = True
            info.ip_type = "IPv4" if isinstance(ip_obj, ipaddress.IPv4Address) else "IPv6"
            return True
        except ValueError:
            info.is_valid = False
            info.errors.append(f"Invalid IP address format: {ip}")
            return False

    def _classify_ip(self, ip: str, info: IPInfo) -> None:
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                info.ip_class = "private"
            elif ip_obj.is_loopback:
                info.ip_class = "loopback"
            elif ip_obj.is_global:
                info.ip_class = "public"
            else:
                info.ip_class = "unknown"
        except Exception as e:
            info.errors.append(f"Classification error: {str(e)}")

    def _reverse_dns(self, ip: str, info: IPInfo) -> None:
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            info.reverse_dns = hostname
        except:
            info.reverse_dns = ""

    def _check_local_blacklists(self, ip: str, info: IPInfo) -> None:
        if ip in self.custom_blacklist:
            info.is_known_abuser = True
            info.blacklists.append("custom_blacklist")
            info.blacklist_count += 1

    def _check_malicious_ranges(self, ip: str, info: IPInfo) -> None:
        try:
            ip_obj = ipaddress.ip_address(ip)
            for range_str in self.KNOWN_MALICIOUS_RANGES:
                network = ipaddress.ip_network(range_str, strict=False)
                if ip_obj in network:
                    info.is_threat = True
                    info.blacklists.append(f"known_malicious_range:{range_str}")
                    info.blacklist_count += 1
        except Exception as e:
            info.errors.append(f"Range check error: {str(e)}")

    def _geolocation_lookup(self, ip: str, info: IPInfo) -> None:
        try:
            response = requests.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "status,country,countryCode,regionName,city,lat,lon,timezone,isp,org,as,proxy,hosting"},
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    info.country = data.get("country", "")
                    info.country_code = data.get("countryCode", "")
                    info.region = data.get("regionName", "")
                    info.city = data.get("city", "")
                    info.latitude = data.get("lat", 0.0)
                    info.longitude = data.get("lon", 0.0)
                    info.timezone = data.get("timezone", "")
                    info.isp = data.get("isp", "")
                    info.org = data.get("org", "")
                    info.asn = data.get("as", "").split()[0] if data.get("as") else ""
                    info.is_proxy = data.get("proxy", False)
                    info.is_datacenter = data.get("hosting", False)
                    info.data_sources.append("ip-api.com")

                    if info.asn in self.DATACENTER_ASNS:
                        info.is_datacenter = True
                    if info.asn in self.VPN_ASNS:
                        info.is_vpn = True
        except Exception as e:
            info.errors.append(f"Geolocation API error: {str(e)}")

    def _abuseipdb_check(self, ip: str, info: IPInfo) -> None:
        try:
            headers = {"Key": self.abuseipdb_key, "Accept": "application/json"}
            response = requests.get(
                "https://api.abuseipdb.com/api/v2/check",
                headers=headers,
                params={"ipAddress": ip, "maxAgeInDays": 90},
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                info.abuse_confidence_score = data.get("abuseConfidenceScore", 0)
                info.is_tor = data.get("isTor", False)
                if info.abuse_confidence_score > 25:
                    info.is_known_abuser = True
                if info.abuse_confidence_score > 75:
                    info.is_known_attacker = True
                info.data_sources.append("AbuseIPDB")
        except Exception as e:
            info.errors.append(f"AbuseIPDB API error: {str(e)}")

    def _calculate_threat_score(self, info: IPInfo) -> None:
        score = info.abuse_confidence_score * 0.4
        score += min(info.blacklist_count * 10, 30)
        if info.is_known_attacker:
            score += 25
        if info.is_known_abuser:
            score += 15
        if info.is_threat:
            score += 20
        if info.is_tor:
            score += 10
        if info.is_vpn:
            score += 5
        if info.is_proxy:
            score += 5
        info.threat_score = min(score, 100)
        info.reputation_score = max(0, 100 - info.threat_score)

    def extract_ips_from_text(self, text: str) -> List[str]:
        ipv4_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        matches = re.findall(ipv4_pattern, text)
        valid_ips = []
        seen = set()
        for ip in matches:
            if ip not in seen:
                try:
                    ipaddress.ip_address(ip)
                    valid_ips.append(ip)
                    seen.add(ip)
                except ValueError:
                    continue
        return valid_ips

    def get_threat_summary(self, info: IPInfo) -> Dict:
        if not info.is_valid:
            return {"status": "INVALID", "message": "Invalid IP address format", "threat_level": "UNKNOWN"}
        if info.ip_class == "private":
            return {"status": "PRIVATE", "message": "Private IP address", "threat_level": "N/A"}

        if info.threat_score >= 75:
            level, color = "CRITICAL", "#ff0000"
        elif info.threat_score >= 50:
            level, color = "HIGH", "#ff4444"
        elif info.threat_score >= 25:
            level, color = "MEDIUM", "#ffaa00"
        elif info.threat_score > 0:
            level, color = "LOW", "#ffcc00"
        else:
            level, color = "NONE", "#00ff88"

        flags = []
        if info.is_known_attacker:
            flags.append("Known Attacker")
        if info.is_tor:
            flags.append("Tor Exit Node")
        if info.is_vpn:
            flags.append("VPN")
        if info.is_proxy:
            flags.append("Proxy")

        return {
            "status": "ANALYZED",
            "threat_level": level,
            "threat_level_color": color,
            "threat_score": info.threat_score,
            "reputation_score": info.reputation_score,
            "flags": flags,
            "location": f"{info.city}, {info.country}" if info.city else info.country,
            "isp": info.isp,
            "blacklist_count": info.blacklist_count
        }