import os
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP


class Mailer:
    smtp_server: SMTP = None

    def __init__(self):
        self.smtp_server = SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT")))
        self.smtp_server.starttls(context=ssl.create_default_context())
        self.smtp_server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))

    def disconnect(self):
        if self.smtp_server is not None:
            self.smtp_server.close()

    def send_email(self, from_addr: str, to_addr: str, subject: str, msg: str, html: str = None):
        # If SMTP is disabled
        if not self.smtp_server:
            return

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = from_addr
        message["To"] = to_addr

        message.attach(MIMEText(msg, "plain"))
        if html is not None:
            message.attach(MIMEText(html, "html"))

        self.smtp_server.sendmail(from_addr, to_addr, message.as_string())
