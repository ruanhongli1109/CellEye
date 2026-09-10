"""告警推送。

支持企业微信 / 钉钉机器人 webhook 与邮件。
默认 enabled=false —— 调试阶段最忌讳被自己的告警刷屏，
真要联调时再把开关打开。

带冷却时间：同一条告警在 cooldown_minutes 内只发一次。
培养箱异常通常是持续状态而非瞬时事件，不冷却就会把手机炸掉。
"""

from __future__ import annotations

import json
import smtplib
import time
import urllib.request
from email.mime.text import MIMEText

from .config import Config

LEVEL_ICON = {"ok": "✅", "warn": "⚠️", "alert": "🚨", "unknown": "❔"}


class Alerter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.get("alert.enabled", False))
        self.channel = str(cfg.get("alert.channel", "wecom"))
        self.webhook = str(cfg.get("alert.webhook", "") or "")
        self.email_to = str(cfg.get("alert.email_to", "") or "")
        self.cooldown_s = float(cfg.get("alert.cooldown_minutes", 60)) * 60
        self._last_sent: dict[str, float] = {}

    def _cooled_down(self, key: str) -> bool:
        now = time.time()
        last = self._last_sent.get(key, 0.0)
        if now - last < self.cooldown_s:
            return False
        self._last_sent[key] = now
        return True

    def send(self, level: str, title: str, message: str, key: str | None = None) -> bool:
        """发送一条告警。返回是否真的推了出去。"""
        key = key or title
        if not self._cooled_down(key):
            return False

        icon = LEVEL_ICON.get(level, "")
        text = f"{icon} **{title}**\n\n{message}"

        if not self.enabled:
            print(f"[alert/dry-run] {level.upper()} | {title} | {message}")
            return False

        try:
            if self.channel in ("wecom", "dingtalk"):
                self._send_webhook(text)
            elif self.channel == "email":
                self._send_email(title, text)
            else:
                print(f"[alert] 未知通道: {self.channel}")
                return False
            return True
        except Exception as exc:  # 告警失败绝不能影响采集主流程
            print(f"[alert] 发送失败: {exc}")
            return False

    def _send_webhook(self, text: str) -> None:
        if not self.webhook:
            raise ValueError("alert.webhook 为空")
        if self.channel == "wecom":
            payload = {"msgtype": "markdown", "markdown": {"content": text}}
        else:
            payload = {"msgtype": "markdown", "markdown": {"title": "CellEye", "text": text}}

        req = urllib.request.Request(
            self.webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "ignore")
        if '"errcode":0' not in body and '"errcode": 0' not in body:
            print(f"[alert] webhook 返回异常: {body[:200]}")

    def _send_email(self, subject: str, text: str) -> None:
        import os

        host = os.environ.get("CELLEYE_SMTP_HOST", "")
        user = os.environ.get("CELLEYE_SMTP_USER", "")
        password = os.environ.get("CELLEYE_SMTP_PASS", "")
        if not (host and user and self.email_to):
            raise ValueError("邮件告警需要设置 CELLEYE_SMTP_HOST / USER / PASS 与 alert.email_to")

        msg = MIMEText(text, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = self.email_to
        with smtplib.SMTP_SSL(host, 465, timeout=15) as s:
            s.login(user, password)
            s.sendmail(user, [self.email_to], msg.as_string())
